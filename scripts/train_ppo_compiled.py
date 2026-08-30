"""
train_ppo_compiled.py — PPO via the compile_ppo framework (Tier 5).

Demonstrates one-call compilation: the caller provides two pure JAX functions
(reset_with_obs, step_with_obs) and a Flax policy; the framework compiles
vmap, lax.scan rollout, reverse-scan GAE, and scanned minibatch updates
automatically.

Usage:
    python scripts/train_ppo_compiled.py --seed 1 --envs 16 --updates 300
    python scripts/train_ppo_compiled.py --seed 1 --envs 256 --updates 100
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn

from ewm import jax_env
from ewm.env import PushCubeEnv
from ewm.jax_convert import PPOConfig, compile_ppo


# ── Policy (identical architecture to Tier 2 / Tier 4 baselines) ─────────────

class ActorCritic(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs):
        x     = nn.tanh(nn.Dense(128)(obs))
        x     = nn.tanh(nn.Dense(128)(x))
        mean  = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)[..., 0]
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.5), (self.action_dim,)
        )
        return mean, log_std, value


# ── Evaluation (uses NumPy env for fair comparison with Tier 2) ───────────────

def evaluate(params, episodes: int, seed: int) -> dict:
    net       = ActorCritic()
    successes = []
    for ep in range(episodes):
        env = PushCubeEnv()
        obs, _ = env.reset(seed=seed + ep)
        while True:
            mean, _, _ = net.apply(params, jnp.asarray(obs)[None])
            action     = np.tanh(np.asarray(mean[0])).astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(float(info["success"]))
                break
    return {
        "success_rate": float(np.mean(successes)),
        "eval_episodes": episodes,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int, default=1)
    parser.add_argument("--envs",          type=int, default=16)
    parser.add_argument("--updates",       type=int, default=300)
    parser.add_argument("--horizon",       type=int, default=128)
    parser.add_argument("--epochs",        type=int, default=4)
    parser.add_argument("--minibatch",     type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output",        default="artifacts/ppo_framework.json")
    args = parser.parse_args()

    print(f"compile_ppo framework | envs={args.envs} updates={args.updates} seed={args.seed}")

    # ── One-call compilation ──────────────────────────────────────────────────
    trainer = compile_ppo(
        reset_fn   = jax_env.reset_with_obs,
        step_fn    = jax_env.step_with_obs,
        net        = ActorCritic(),
        obs_dim    = 8,
        action_dim = 2,
    )

    config = PPOConfig(
        num_envs       = args.envs,
        horizon        = args.horizon,
        updates        = args.updates,
        epochs         = args.epochs,
        minibatch_size = args.minibatch,
    )

    result = trainer.train(config, seed=args.seed)
    params = result.pop("params")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_result = evaluate(params, args.eval_episodes, args.seed)

    output_dict = {
        "method":  "ppo_framework",
        "envs":    args.envs,
        "updates": args.updates,
        "horizon": args.horizon,
        **result,
        **eval_result,
    }

    print(json.dumps(output_dict, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output_dict, indent=2) + "\n")


if __name__ == "__main__":
    main()
