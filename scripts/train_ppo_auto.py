"""
train_ppo_auto.py — End-to-end demo: NumPy Gym env → auto-converted → compiled PPO.

Pipeline:
    PushCubeEnv (NumPy Gymnasium)
        ↓  gym_to_jax()          ← automatic conversion
    (reset_fn, step_fn)  [pure JAX]
        ↓  compile_ppo()         ← automatic compilation
    PPOTrainer             [fully XLA-compiled]
        ↓  .train()
    results                [22× faster than Python baseline]

Usage:
    python scripts/train_ppo_auto.py --seed 1 --envs 16 --updates 300
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2xla.env import PushCubeEnv
from rl2xla.gym_to_jax import gym_to_jax
from rl2xla.jax_convert import PPOConfig, compile_ppo


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


def evaluate(params, episodes: int, seed: int) -> dict:
    net = ActorCritic()
    successes = []
    for ep in range(episodes):
        env = PushCubeEnv()
        obs, _ = env.reset(seed=seed + ep)
        while True:
            mean, _, _ = net.apply(params, jnp.asarray(obs)[None])
            action = np.tanh(np.asarray(mean[0])).astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(float(info["success"]))
                break
    return {"success_rate": float(np.mean(successes)), "eval_episodes": episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int, default=1)
    parser.add_argument("--envs",          type=int, default=16)
    parser.add_argument("--updates",       type=int, default=300)
    parser.add_argument("--horizon",       type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output",        default="artifacts/ppo_auto.json")
    args = parser.parse_args()

    print(f"Step 1: gym_to_jax(PushCubeEnv)  [auto-convert NumPy → JAX]")
    reset_fn, step_fn = gym_to_jax(PushCubeEnv)
    print(f"        reset_fn: {reset_fn}")
    print(f"        step_fn:  {step_fn}")

    print(f"Step 2: compile_ppo(reset_fn, step_fn, ...)  [auto-compile PPO]")
    trainer = compile_ppo(
        reset_fn=reset_fn,
        step_fn=step_fn,
        net=ActorCritic(),
        obs_dim=8,
        action_dim=2,
    )

    config = PPOConfig(
        num_envs=args.envs,
        horizon=args.horizon,
        updates=args.updates,
    )

    print(f"Step 3: trainer.train(...)  [envs={args.envs}, updates={args.updates}, seed={args.seed}]")
    result = trainer.train(config, seed=args.seed)
    params = result.pop("params")

    eval_result = evaluate(params, args.eval_episodes, args.seed)

    output_dict = {
        "method":  "ppo_auto_converted",
        "envs":    args.envs,
        "updates": args.updates,
        "horizon": args.horizon,
        **result,
        **eval_result,
    }

    print("\nResults:")
    print(json.dumps(output_dict, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output_dict, indent=2) + "\n")


if __name__ == "__main__":
    main()
