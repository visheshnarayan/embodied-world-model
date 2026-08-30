"""Train a small continuous-control PPO baseline on PushCubeEnv."""
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

from rl2xla.env import PushCubeEnv


class ActorCritic(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(128)(obs))
        x = nn.tanh(nn.Dense(128)(x))
        mean = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)[..., 0]
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        return mean, log_std, value


def gaussian_log_prob(mean, log_std, action):
    scale = jnp.exp(log_std)
    return jnp.sum(-0.5 * (((action - mean) / scale) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)), axis=-1)


def make_update(net, optimizer, clip_ratio, value_coef, entropy_coef):
    def update(params, opt_state, obs, actions, old_logp, returns, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def loss_fn(p):
            mean, log_std, values = net.apply(p, obs)
            logp = gaussian_log_prob(mean, log_std, actions)
            ratio = jnp.exp(logp - old_logp)
            clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
            value_loss = 0.5 * jnp.mean((returns - values) ** 2)
            entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
            return policy_loss + value_coef * value_loss - entropy_coef * entropy, (policy_loss, value_loss, entropy)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, jnp.array([loss, *metrics])

    return jax.jit(update)


def evaluate(params, episodes, seed):
    net = ActorCritic()
    successes, returns, distances = [], [], []
    for episode in range(episodes):
        env = PushCubeEnv()
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            mean, _, _ = net.apply(params, jnp.asarray(obs)[None])
            action = np.tanh(np.asarray(mean[0])).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                successes.append(float(info["success"]))
                returns.append(total)
                distances.append(info["distance"])
                break
    return {
        "success_rate": float(np.mean(successes)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "final_distance_mean": float(np.mean(distances)),
        "episodes": episodes,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output", default="artifacts/ppo.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    envs = [PushCubeEnv() for _ in range(args.envs)]
    obs = np.stack([env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)])
    net = ActorCritic()
    params = net.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 8), jnp.float32))
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    update = make_update(net, optimizer, 0.2, 0.5, 0.01)

    start_time = time.perf_counter()
    for update_idx in range(args.updates):
        observations, actions, rewards, values, logps, dones = [], [], [], [], [], []
        for _ in range(args.horizon):
            mean, log_std, value = net.apply(params, jnp.asarray(obs))
            noise = rng.normal(size=mean.shape).astype(np.float32)
            raw_action = np.asarray(mean) + np.exp(np.asarray(log_std)) * noise
            action = np.tanh(raw_action).astype(np.float32)
            logp = np.asarray(gaussian_log_prob(mean, log_std, jnp.asarray(raw_action)))
            observations.append(obs.copy()); actions.append(raw_action.astype(np.float32)); values.append(np.asarray(value)); logps.append(logp)
            next_obs, step_rewards, step_dones = [], [], []
            for i, env in enumerate(envs):
                nxt, reward, terminated, truncated, _ = env.step(action[i])
                done = terminated or truncated
                next_obs.append(env.reset(seed=args.seed + update_idx * args.envs + i)[0] if done else nxt)
                step_rewards.append(reward); step_dones.append(done)
            rewards.append(np.asarray(step_rewards, np.float32)); dones.append(np.asarray(step_dones, np.float32)); obs = np.stack(next_obs)

        _, _, last_value = net.apply(params, jnp.asarray(obs))
        observations = np.asarray(observations); actions = np.asarray(actions); rewards = np.asarray(rewards)
        values = np.asarray(values); logps = np.asarray(logps); dones = np.asarray(dones)
        advantages = np.zeros_like(rewards); gae = np.zeros(args.envs, np.float32)
        next_values = np.asarray(last_value)
        for t in range(args.horizon - 1, -1, -1):
            mask = 1.0 - dones[t]
            delta = rewards[t] + 0.99 * next_values * mask - values[t]
            gae = delta + 0.99 * 0.95 * mask * gae
            advantages[t] = gae; next_values = values[t]
        returns = advantages + values
        flat = lambda x: x.reshape((-1, x.shape[-1])) if x.ndim == 3 else x.reshape(-1)
        train_obs, train_actions = flat(observations), flat(actions)
        train_logps, train_returns, train_adv = flat(logps), flat(returns), flat(advantages)
        indices = np.arange(len(train_obs))
        for _ in range(args.epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), args.minibatch_size):
                idx = indices[start:start + args.minibatch_size]
                params, opt_state, _ = update(params, opt_state, train_obs[idx], train_actions[idx], train_logps[idx], train_returns[idx], train_adv[idx])
        if (update_idx + 1) % max(1, args.updates // 10) == 0:
            print(f"update={update_idx + 1}/{args.updates}")

    elapsed = time.perf_counter() - start_time
    total_steps = args.updates * args.envs * args.horizon

    result = evaluate(params, args.eval_episodes, args.seed)
    result.update({
        "method": "ppo_tier2",
        "updates": args.updates,
        "envs": args.envs,
        "horizon": args.horizon,
        "total_steps": total_steps,
        "wall_time_s": round(elapsed, 2),
        "steps_per_second": round(total_steps / elapsed, 0),
    })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
