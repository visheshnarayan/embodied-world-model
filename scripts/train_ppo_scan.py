"""Tier 4 — Full scan-based PPO on the pure-JAX PushCubeEnv.

Every bottleneck that slows traditional RL is compiled away:

  Tier 2 (train_ppo.py)       | Tier 4 (this file)
  ----------------------------|-----------------------------------
  Python for-loop over N envs | jax.vmap — one XLA kernel
  Python for-loop over H steps| jax.lax.scan — one XLA kernel
  Python GAE backward loop    | jax.lax.scan(reverse=True)
  Python minibatch loop       | jax.lax.scan per epoch
  NumPy environment           | Pure-JAX jax_env (no Python calls)

Usage:
    python scripts/train_ppo_scan.py --num-envs 1024 --updates 300
    python scripts/train_ppo_scan.py --num-envs 1024 --dtype bfloat16

The bfloat16 flag uses mixed precision: fp32 params/optimizer, bf16 activations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from rl2xla import jax_env


# ── Network ───────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    action_dim: int = 2
    hidden: int = 128
    # compute_dtype controls activation precision; params always stored in fp32
    compute_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs: jax.Array):
        x = obs.astype(self.compute_dtype)
        x = nn.tanh(nn.Dense(self.hidden, dtype=self.compute_dtype)(x))
        x = nn.tanh(nn.Dense(self.hidden, dtype=self.compute_dtype)(x))
        mean = nn.Dense(self.action_dim, dtype=self.compute_dtype)(x).astype(jnp.float32)
        value = nn.Dense(1, dtype=self.compute_dtype)(x)[..., 0].astype(jnp.float32)
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.action_dim,)
        )
        return mean, log_std, value


def gaussian_log_prob(mean: jax.Array, log_std: jax.Array, action: jax.Array) -> jax.Array:
    """Log-prob of action under N(mean, exp(log_std)). action is the pre-tanh sample."""
    scale = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * (((action - mean) / scale) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)),
        axis=-1,
    )


# ── Auto-reset helper ─────────────────────────────────────────────────────────

def _select(done: jax.Array, fresh: jax.Array, current: jax.Array) -> jax.Array:
    """Where done[i]=True pick fresh[i], else current[i]. Broadcasts over trailing dims."""
    mask = done
    while mask.ndim < fresh.ndim:
        mask = mask[..., None]
    return jnp.where(mask, fresh, current)


# ── JIT'd rollout collection ──────────────────────────────────────────────────

def make_collect_rollout(net: ActorCritic, num_envs: int, horizon: int):
    """Return a JIT'd function that collects one (H, N, ...) trajectory via lax.scan."""
    batch_step  = jax.vmap(jax_env.step)
    batch_reset = jax.vmap(jax_env.reset)
    batch_obs   = jax.vmap(jax_env.get_obs)

    @jax.jit
    def collect(
        params, env_state: jax_env.EnvState, key: jax.Array
    ) -> tuple[tuple, jax_env.EnvState, jax.Array]:

        def scan_step(carry, _):
            env_state, key = carry
            key, act_key, rst_key = jax.random.split(key, 3)

            obs = batch_obs(env_state)                        # (N, 8)
            mean, log_std, value = net.apply(params, obs)    # (N,2), (2,), (N,)

            noise = jax.random.normal(act_key, mean.shape)
            raw_action = mean + jnp.exp(log_std) * noise     # pre-tanh  (N, 2)
            action = jnp.tanh(raw_action)                    # bounded   (N, 2)
            logp = gaussian_log_prob(mean, log_std, raw_action)  # (N,)

            next_state, reward, done, _ = batch_step(env_state, action)

            # Auto-reset: replace done envs with fresh random states
            rst_keys = jax.random.split(rst_key, num_envs)
            fresh = batch_reset(rst_keys)
            final_state = jax.tree_util.tree_map(
                lambda fr, nx: _select(done, fr, nx), fresh, next_state
            )

            transition = (obs, raw_action, logp, value, reward, done)
            return (final_state, key), transition

        (final_state, key), traj = jax.lax.scan(
            scan_step, (env_state, key), None, length=horizon
        )
        # traj: tuple of (H, N, ...) arrays
        return traj, final_state, key

    return collect


# ── JIT'd GAE (reverse scan) ──────────────────────────────────────────────────

@jax.jit
def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """GAE via lax.scan(reverse=True). All inputs (H, N), last_value (N,)."""

    def gae_step(carry, t):
        next_gae, next_value = carry
        reward, value, done = t
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * next_gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, advantages + values   # advantages, returns


# ── JIT'd PPO epoch (minibatch scan) ─────────────────────────────────────────

def make_run_epoch(
    net: ActorCritic,
    optimizer,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    minibatch_size: int,
):
    """Return a JIT'd function that runs one PPO epoch (scanned over minibatches)."""

    def _update_mb(carry, batch):
        params, opt_state = carry
        obs, raw_actions, old_logp, returns, advantages = batch

        def loss_fn(p):
            mean, log_std, values = net.apply(p, obs)
            logp = gaussian_log_prob(mean, log_std, raw_actions)
            ratio = jnp.exp(logp - old_logp)
            clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = -jnp.mean(jnp.minimum(ratio * adv_norm, clipped * adv_norm))
            value_loss  = 0.5 * jnp.mean((returns - values) ** 2)
            entropy     = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
            total = policy_loss + value_coef * value_loss - entropy_coef * entropy
            return total, (policy_loss, value_loss, entropy)

        (total_loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt), total_loss

    @jax.jit
    def run_epoch(params, opt_state, perm, f_obs, f_act, f_logp, f_ret, f_adv):
        n = f_obs.shape[0]
        num_mb = n // minibatch_size

        # Shuffle via permutation index
        obs_s  = f_obs[perm];  act_s  = f_act[perm];  logp_s = f_logp[perm]
        ret_s  = f_ret[perm];  adv_s  = f_adv[perm]

        # Reshape into (num_mb, minibatch_size, ...)
        def mb(x):
            return x.reshape(num_mb, minibatch_size, *x.shape[1:])

        batches = (mb(obs_s), mb(act_s), mb(logp_s), mb(ret_s), mb(adv_s))
        (params, opt_state), losses = jax.lax.scan(_update_mb, (params, opt_state), batches)
        return params, opt_state, losses.mean()

    return run_epoch


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(params, net: ActorCritic, episodes: int, seed: int) -> dict:
    """Greedy evaluation (tanh(mean)) on the JAX env."""
    successes, returns, distances = [], [], []
    for ep in range(episodes):
        key = jax.random.PRNGKey(seed + ep)
        state = jax_env.reset(key)
        total = 0.0
        done = False
        while not done:
            obs = jax_env.get_obs(state)[None]       # (1, 8)
            mean, _, _ = net.apply(params, obs)
            action = jnp.tanh(mean[0])
            state, reward, done, success = jax_env.step(state, action)
            total += float(reward)
        successes.append(float(success))
        returns.append(total)
        distances.append(float(jnp.linalg.norm(state.cube - state.target)))
    return {
        "success_rate": float(np.mean(successes)),
        "return_mean":  float(np.mean(returns)),
        "return_std":   float(np.std(returns)),
        "final_distance_mean": float(np.mean(distances)),
        "episodes": episodes,
        "seed": seed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 4 scan-based PPO")
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--updates",        type=int,   default=300,
                        help="Number of rollout+update iterations")
    parser.add_argument("--num-envs",       type=int,   default=1024,
                        help="Parallel environments (vmapped)")
    parser.add_argument("--horizon",        type=int,   default=128,
                        help="Rollout length (scanned over)")
    parser.add_argument("--epochs",         type=int,   default=4)
    parser.add_argument("--minibatch-size", type=int,   default=1024)
    parser.add_argument("--eval-episodes",  type=int,   default=50)
    parser.add_argument("--dtype",          default="float32",
                        choices=["float32", "bfloat16"],
                        help="Network activation dtype (params always fp32)")
    parser.add_argument("--output",         default="artifacts/ppo_scan.json")
    args = parser.parse_args()

    flat_size = args.num_envs * args.horizon
    assert flat_size % args.minibatch_size == 0, (
        f"num_envs*horizon ({flat_size}) must be divisible by minibatch_size ({args.minibatch_size})"
    )

    compute_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    net = ActorCritic(compute_dtype=compute_dtype)

    key = jax.random.PRNGKey(args.seed)
    key, init_key, env_key = jax.random.split(key, 3)

    params    = net.init(init_key, jnp.zeros((1, 8)))
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    env_keys  = jax.random.split(env_key, args.num_envs)
    env_state = jax.vmap(jax_env.reset)(env_keys)

    collect_rollout = make_collect_rollout(net, args.num_envs, args.horizon)
    run_epoch       = make_run_epoch(net, optimizer, 0.2, 0.5, 0.01, args.minibatch_size)
    batch_obs       = jax.vmap(jax_env.get_obs)

    # Warmup: compile all JIT'd functions before timing
    print("Compiling... (first update includes JIT overhead)")
    traj, env_state, key = collect_rollout(params, env_state, key)
    obs_t, act_t, logp_t, val_t, rew_t, done_t = traj
    last_obs    = batch_obs(env_state)
    _, _, last_val = net.apply(params, last_obs)
    advantages, returns = compute_gae(rew_t, val_t, done_t, last_val)
    flat = lambda x: x.reshape(flat_size, *x.shape[2:]) if x.ndim > 2 else x.reshape(flat_size)
    f_obs  = flat(obs_t); f_act = flat(act_t); f_logp = flat(logp_t)
    f_ret  = flat(returns); f_adv = flat(advantages)
    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, flat_size)
    params, opt_state, _ = run_epoch(params, opt_state, perm, f_obs, f_act, f_logp, f_ret, f_adv)
    jax.block_until_ready(params)
    print("Compilation done. Starting timed training.\n")

    total_steps = 0
    start_time  = time.perf_counter()

    for update_idx in range(args.updates):
        key, rollout_key = jax.random.split(key)
        traj, env_state, _ = collect_rollout(params, env_state, rollout_key)
        obs_t, act_t, logp_t, val_t, rew_t, done_t = traj

        last_obs = batch_obs(env_state)
        _, _, last_val = net.apply(params, last_obs)
        advantages, returns = compute_gae(rew_t, val_t, done_t, last_val)

        flat = lambda x: x.reshape(flat_size, *x.shape[2:]) if x.ndim > 2 else x.reshape(flat_size)
        f_obs  = flat(obs_t); f_act = flat(act_t); f_logp = flat(logp_t)
        f_ret  = flat(returns); f_adv = flat(advantages)

        for _ in range(args.epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, flat_size)
            params, opt_state, _ = run_epoch(
                params, opt_state, perm, f_obs, f_act, f_logp, f_ret, f_adv
            )

        total_steps += flat_size
        if (update_idx + 1) % max(1, args.updates // 10) == 0:
            jax.block_until_ready(params)   # flush async ops before timing
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed
            print(f"update={update_idx+1:4d}/{args.updates}  "
                  f"steps={total_steps:>10,}  steps/s={sps:>10,.0f}")

    jax.block_until_ready(params)
    elapsed = time.perf_counter() - start_time
    sps = total_steps / elapsed

    result = evaluate(params, net, args.eval_episodes, args.seed)
    result.update({
        "method":           f"ppo_scan_{args.dtype}",
        "updates":          args.updates,
        "num_envs":         args.num_envs,
        "horizon":          args.horizon,
        "total_steps":      total_steps,
        "wall_time_s":      round(elapsed, 2),
        "steps_per_second": round(sps, 0),
        "dtype":            args.dtype,
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
