"""
benchmark_microduck.py — Throughput benchmark for MicroduckWalkEnv.

Compares:
  Tier 2  MicroduckWalkEnv (NumPy) + Python for-loops
  Auto    gym_to_jax(MicroduckWalkEnv) → compile_ppo → lax.scan (fully compiled)

Usage:
    python scripts/benchmark_microduck.py
    python scripts/benchmark_microduck.py --steps 500000 --envs 64
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ewm.microduck_env import MicroduckWalkEnv, OBS_DIM, NUM_JOINTS
from ewm.gym_to_jax import gym_to_jax
from ewm.jax_convert import compile_ppo, PPOConfig


# ── Shared network ────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    action_dim: int = NUM_JOINTS

    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(128)(obs))
        x = nn.tanh(nn.Dense(128)(x))
        mean    = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        value   = nn.Dense(1)(x)[..., 0]
        return mean, log_std, value


# ── Tier 2: NumPy env, Python for-loops ──────────────────────────────────────

def bench_tier2(num_envs: int, horizon: int, target_steps: int, warmup_steps: int) -> dict:
    net    = ActorCritic()
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _apply = jax.jit(net.apply)
    rng    = np.random.default_rng(0)

    def rollout(n_steps: int) -> int:
        envs = [MicroduckWalkEnv(max_steps=horizon) for _ in range(num_envs)]
        obs  = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)])
        done = 0
        while done < n_steps:
            for _ in range(horizon):
                mean, log_std, _ = _apply(params, jnp.asarray(obs))
                noise  = rng.standard_normal(mean.shape).astype(np.float32)
                action = np.tanh(np.asarray(mean) + np.exp(np.asarray(log_std)) * noise)
                next_obs = []
                for i, env in enumerate(envs):
                    o, _, term, trunc, _ = env.step(action[i])
                    if term or trunc:
                        o, _ = env.reset(seed=i)
                    next_obs.append(o)
                obs = np.stack(next_obs)
            done += num_envs * horizon
        return done

    print(f"  [tier2-microduck] warmup ({warmup_steps:,} steps)...")
    rollout(warmup_steps)
    print(f"  [tier2-microduck] timing ({target_steps:,} steps)...")
    t0 = time.perf_counter()
    actual = rollout(target_steps)
    elapsed = time.perf_counter() - t0
    sps = actual / elapsed
    print(f"  [tier2-microduck] {sps:>12,.0f} steps/s  ({elapsed:.2f}s)\n")
    return {"name": "Tier 2 (Python + NumPy)", "steps_per_second": sps, "wall_time_s": elapsed}


# ── Auto (gym_to_jax → compile_ppo → lax.scan) ───────────────────────────────

def bench_auto(num_envs: int, horizon: int, target_steps: int, warmup_steps: int) -> dict:
    reset_fn, step_fn = gym_to_jax(MicroduckWalkEnv)
    net     = ActorCritic()
    trainer = compile_ppo(reset_fn, step_fn, net)

    updates_target = max(1, target_steps // (num_envs * horizon))
    updates_warmup = max(1, warmup_steps  // (num_envs * horizon))
    cfg = PPOConfig(num_envs=num_envs, horizon=horizon, updates=updates_target)

    print(f"  [auto-microduck] warmup ({updates_warmup} updates)...")
    trainer.train(PPOConfig(num_envs=num_envs, horizon=horizon, updates=updates_warmup), seed=1)

    print(f"  [auto-microduck] timing ({updates_target} updates = {updates_target*num_envs*horizon:,} steps)...")
    result = trainer.train(cfg, seed=0)
    actual = result["total_steps"]
    elapsed = result["wall_time_s"]
    sps = result["steps_per_second"]
    print(f"  [auto-microduck] {sps:>12,.0f} steps/s  ({elapsed:.2f}s)\n")
    return {"name": "Auto (gym_to_jax + compile_ppo)", "steps_per_second": sps, "wall_time_s": elapsed}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=200_000)
    parser.add_argument("--warmup",  type=int, default=20_000)
    parser.add_argument("--envs",    type=int, default=16)
    parser.add_argument("--horizon", type=int, default=128)
    args = parser.parse_args()

    print(f"JAX backend  : {jax.default_backend()}")
    print(f"JAX devices  : {jax.devices()}")
    print(f"Env          : MicroduckWalkEnv (obs={OBS_DIM}, act={NUM_JOINTS})")
    print(f"Parallel envs: {args.envs}   horizon: {args.horizon}")
    print(f"Target steps : {args.steps:,}   warmup: {args.warmup:,}\n")

    r2   = bench_tier2(args.envs, args.horizon, args.steps, args.warmup)
    auto = bench_auto( args.envs, args.horizon, args.steps, args.warmup)

    speedup = auto["steps_per_second"] / r2["steps_per_second"]

    print("=" * 60)
    print(f"{'Method':<35} {'Steps/s':>12}  {'Speedup':>8}")
    print("-" * 60)
    print(f"{r2['name']:<35} {r2['steps_per_second']:>12,.0f}  {'1.0×':>8}")
    print(f"{auto['name']:<35} {auto['steps_per_second']:>12,.0f}  {speedup:>7.1f}×")
    print("=" * 60)


if __name__ == "__main__":
    main()
