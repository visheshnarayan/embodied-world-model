"""Throughput benchmark: Tier 2 vs Tier 3 vs Tier 4 rollout collection.

Measures environment steps per second for three progressively more JAX-native
rollout implementations, holding total steps fixed so wall-clock times are
directly comparable.

Tiers
-----
  Tier 2  NumPy env   + Python for-loop over envs + Python for-loop over H
          (PushCubeEnv, matches train_ppo.py)
  Tier 3  JAX env     + Python for-loop over H  (vmap over envs, no scan)
  Tier 4  JAX env     + lax.scan over H         (fully compiled, matches train_ppo_scan.py)

Usage:
    python scripts/benchmark_throughput.py
    python scripts/benchmark_throughput.py --steps 2_000_000 --envs 256
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from ewm.env import PushCubeEnv
from ewm import jax_env


# ── Shared network ────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs: jax.Array):
        x = nn.tanh(nn.Dense(128)(obs))
        x = nn.tanh(nn.Dense(128)(x))
        mean  = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)[..., 0]
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        return mean, log_std, value


def gaussian_log_prob(mean, log_std, action):
    scale = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * (((action - mean) / scale) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)),
        axis=-1,
    )


# ── Tier 2: NumPy env, Python loops ──────────────────────────────────────────

def make_tier2_rollout(net, params, num_envs: int, horizon: int):
    """One rollout step — Python for-loop over envs and timesteps."""
    rng = np.random.default_rng(0)
    # net.apply is JIT'd separately so the network itself is compiled
    _apply = jax.jit(net.apply)

    def rollout(num_steps: int) -> int:
        """num_steps = total (env × timestep) pairs to process."""
        envs = [PushCubeEnv() for _ in range(num_envs)]
        obs  = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)])
        steps_done = 0
        while steps_done < num_steps:
            for _ in range(horizon):
                # Network forward (batched, JIT'd)
                mean, log_std, _ = _apply(params, jnp.asarray(obs))
                noise  = rng.normal(size=mean.shape).astype(np.float32)
                raw    = np.asarray(mean) + np.exp(np.asarray(log_std)) * noise
                action = np.tanh(raw).astype(np.float32)
                # Python for-loop over envs
                next_obs = []
                for i, env in enumerate(envs):
                    nxt, _, term, trunc, _ = env.step(action[i])
                    if term or trunc:
                        nxt, _ = env.reset(seed=i)
                    next_obs.append(nxt)
                obs = np.stack(next_obs)
            steps_done += num_envs * horizon   # count total env×timestep pairs
        return steps_done

    return rollout


# ── Tier 3: JAX env, Python loop over horizon ─────────────────────────────────

def make_tier3_rollout(net, params, num_envs: int, horizon: int):
    """vmap over envs, Python for-loop over timesteps."""
    batch_step  = jax.jit(jax.vmap(jax_env.step))
    batch_reset = jax.jit(jax.vmap(jax_env.reset))
    batch_obs   = jax.jit(jax.vmap(jax_env.get_obs))
    _apply      = jax.jit(net.apply)

    def _where(done, fresh, current):
        mask = done
        while mask.ndim < fresh.ndim:
            mask = mask[..., None]
        return jnp.where(mask, fresh, current)

    def rollout(num_steps: int, key: jax.Array) -> int:
        """num_steps = total (env × timestep) pairs to process."""
        env_keys  = jax.random.split(key, num_envs)
        env_state = jax.jit(jax.vmap(jax_env.reset))(env_keys)
        steps_done = 0

        while steps_done < num_steps:
            for _ in range(horizon):
                key, act_key, rst_key = jax.random.split(key, 3)
                obs = batch_obs(env_state)
                mean, log_std, _ = _apply(params, obs)
                noise = jax.random.normal(act_key, mean.shape)
                raw   = mean + jnp.exp(log_std) * noise
                act   = jnp.tanh(raw)

                next_state, _, done, _ = batch_step(env_state, act)
                rst_keys = jax.random.split(rst_key, num_envs)
                fresh    = batch_reset(rst_keys)
                env_state = jax.tree_util.tree_map(
                    lambda fr, nx: _where(done, fr, nx), fresh, next_state
                )
            steps_done += num_envs * horizon   # count total env×timestep pairs
            jax.block_until_ready(env_state.hand)   # prevent async inflation
        return steps_done

    return rollout


# ── Tier 4: JAX env, lax.scan ────────────────────────────────────────────────

def make_tier4_rollout(net, params, num_envs: int, horizon: int):
    """vmap over envs, lax.scan over horizon — single XLA kernel per rollout."""
    batch_step  = jax.vmap(jax_env.step)
    batch_reset = jax.vmap(jax_env.reset)
    batch_obs   = jax.vmap(jax_env.get_obs)

    def _where(done, fresh, current):
        mask = done
        while mask.ndim < fresh.ndim:
            mask = mask[..., None]
        return jnp.where(mask, fresh, current)

    @jax.jit
    def _one_rollout(env_state, key):
        def scan_step(carry, _):
            env_state, key = carry
            key, act_key, rst_key = jax.random.split(key, 3)
            obs  = batch_obs(env_state)
            mean, log_std, _ = net.apply(params, obs)
            noise = jax.random.normal(act_key, mean.shape)
            raw   = mean + jnp.exp(log_std) * noise
            act   = jnp.tanh(raw)
            next_state, _, done, _ = batch_step(env_state, act)
            rst_keys  = jax.random.split(rst_key, num_envs)
            fresh     = batch_reset(rst_keys)
            final     = jax.tree_util.tree_map(
                lambda fr, nx: _where(done, fr, nx), fresh, next_state
            )
            return (final, key), None   # we only care about throughput, discard traj
        (env_state, key), _ = jax.lax.scan(scan_step, (env_state, key), None, length=horizon)
        return env_state, key

    def rollout(num_steps: int, key: jax.Array) -> int:
        env_keys  = jax.random.split(key, num_envs)
        env_state = jax.vmap(jax_env.reset)(env_keys)
        steps_done = 0
        while steps_done < num_steps:
            env_state, key = _one_rollout(env_state, key)
            jax.block_until_ready(env_state.hand)
            steps_done += num_envs * horizon
        return steps_done

    return rollout


# ── Timing harness ────────────────────────────────────────────────────────────

def time_rollout(
    name: str,
    fn: Callable,
    target_steps: int,
    warmup_steps: int,
    *args,
) -> dict:
    """Warmup, then time `fn(target_steps, *args)` and return stats."""
    print(f"  [{name}] warming up ({warmup_steps:,} steps)...")
    fn(warmup_steps, *args)

    print(f"  [{name}] timing ({target_steps:,} steps)...")
    t0 = time.perf_counter()
    actual = fn(target_steps, *args)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    sps = actual / elapsed
    print(f"  [{name}] {sps:>12,.0f} steps/s  ({elapsed:.2f}s)\n")
    return {"name": name, "steps_per_second": sps, "wall_time_s": elapsed, "steps": actual}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=2_000_000,
                        help="Total env×timestep pairs to process per tier")
    parser.add_argument("--warmup",  type=int, default=200_000,
                        help="Warmup env×timestep pairs (covers JIT compilation)")
    parser.add_argument("--envs",    type=int, default=256,
                        help="Parallel envs (same for all tiers)")
    parser.add_argument("--horizon", type=int, default=128,
                        help="Rollout horizon per iteration")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    print(f"JAX backend : {jax.default_backend()}")
    print(f"JAX devices : {jax.devices()}")
    print(f"Parallel envs: {args.envs}   horizon: {args.horizon}")
    print(f"Target steps : {args.steps:,}   warmup: {args.warmup:,}\n")

    # Shared network and params
    net    = ActorCritic()
    key    = jax.random.PRNGKey(args.seed)
    params = net.init(key, jnp.zeros((1, 8)))

    results = []

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    tier2 = make_tier2_rollout(net, params, args.envs, args.horizon)
    print("Tier 2: NumPy env, Python for-loops")
    r2 = time_rollout("tier2", tier2, args.steps, args.warmup)
    results.append(r2)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    tier3_key = jax.random.PRNGKey(args.seed + 1)
    tier3 = make_tier3_rollout(net, params, args.envs, args.horizon)
    print("Tier 3: JAX env (vmap), Python for-loop over horizon")
    r3 = time_rollout("tier3", tier3, args.steps, args.warmup, tier3_key)
    results.append(r3)

    # ── Tier 4 ────────────────────────────────────────────────────────────────
    tier4_key = jax.random.PRNGKey(args.seed + 2)
    tier4 = make_tier4_rollout(net, params, args.envs, args.horizon)
    print("Tier 4: JAX env (vmap) + lax.scan — fully compiled")
    r4 = time_rollout("tier4", tier4, args.steps, args.warmup, tier4_key)
    results.append(r4)

    # ── Summary table ─────────────────────────────────────────────────────────
    baseline = results[0]["steps_per_second"]
    print("=" * 55)
    print(f"{'Tier':<10} {'Steps/s':>14} {'Speedup vs T2':>16}")
    print("-" * 55)
    for r in results:
        speedup = r["steps_per_second"] / baseline
        print(f"{r['name']:<10} {r['steps_per_second']:>14,.0f} {speedup:>14.1f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
