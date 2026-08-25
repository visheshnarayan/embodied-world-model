from __future__ import annotations
import numpy as np

def fit_standardizer(data):
    return {
        "obs_mean": data["obs"].mean(0), "obs_std": np.maximum(data["obs"].std(0), 1e-6),
        "action_mean": data["action"].mean(0), "action_std": np.maximum(data["action"].std(0), 1e-6),
        "next_mean": data["next_obs"].mean(0), "next_std": np.maximum(data["next_obs"].std(0), 1e-6),
    }

def transform(data, stats, target="absolute", target_stats=None):
    obs = (data["obs"] - stats["obs_mean"]) / stats["obs_std"]
    action = (data["action"] - stats["action_mean"]) / stats["action_std"]
    if target == "residual":
        target_values = data["next_obs"] - data["obs"]
        target_mean, target_std = target_values.mean(0), np.maximum(target_values.std(0), 1e-6)
    else:
        target_values, target_mean, target_std = data["next_obs"], stats["next_mean"], stats["next_std"]
    if target_stats is not None:
        target_mean, target_std = target_stats["target_mean"], target_stats["target_std"]
    return {"obs": obs, "action": action, "next_obs": (target_values - target_mean) / target_std, "episode": data.get("episode")}, {"target_mean": target_mean, "target_std": target_std}
