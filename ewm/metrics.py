from __future__ import annotations

import numpy as np

from .env import PushCubeEnv
from .planner import mpc_action


def evaluate_controller(controller, episodes=20, seed=0, candidates=64, horizon=8):
    """Evaluate a controller and return metrics suitable for a CSV row."""
    rng = np.random.default_rng(seed)
    returns, final_distances, lengths, contacts, successes = [], [], [], [], []
    for episode in range(episodes):
        env = PushCubeEnv()
        env.reset(seed=seed + episode)
        done, total_return, contact_count = False, 0.0, 0
        while not done:
            if controller == "random":
                action = rng.uniform(-1, 1, 2).astype(np.float32)
            else:
                action = mpc_action(env, controller, rng, candidates, horizon)
            _, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            contact_count += int(info["contact"])
            done = terminated or truncated
        returns.append(total_return)
        final_distances.append(info["distance"])
        lengths.append(env.steps)
        contacts.append(contact_count / max(1, env.steps))
        successes.append(float(info["success"]))
    return {
        "success_rate": float(np.mean(successes)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "final_distance_mean": float(np.mean(final_distances)),
        "episode_length_mean": float(np.mean(lengths)),
        "contact_rate": float(np.mean(contacts)),
        "episodes": episodes,
        "seed": seed,
    }


def one_step_model_mse(model, data):
    predictions = [model.predict(o, a)[0] for o, a in zip(data["obs"], data["action"])]
    return float(np.mean((np.asarray(predictions) - data["next_obs"]) ** 2))

