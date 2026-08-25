from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _array(value):
    return np.asarray(value, dtype=np.float32).reshape(-1)

def load_single_arm(path, max_rows=None, max_episodes=None, stride=1):
    """Load adjacent state/action transitions from a LeRobot checkout."""
    root = Path(path); files = sorted((root / "data").rglob("*.parquet"))
    if not files: raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    table = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    required = {"observation.state", "action"}; missing = required - set(table.columns)
    if missing: raise ValueError(f"Missing columns {sorted(missing)}; found {list(table.columns)}")
    if "episode_index" in table:
        order = ["episode_index"] + (["frame_index"] if "frame_index" in table else [])
        table = table.sort_values(order)
        if max_episodes is not None: table = table[table.episode_index < max_episodes]
    obs = np.stack([_array(x) for x in table["observation.state"]])[::stride]
    actions = np.stack([_array(x) for x in table["action"]])[::stride]
    episodes = table.episode_index.to_numpy()[::stride] if "episode_index" in table else None
    valid = episodes[:-1] == episodes[1:] if episodes is not None else np.ones(len(obs)-1, dtype=bool)
    obs, actions, next_obs = obs[:-1][valid], actions[:-1][valid], obs[1:][valid]
    transition_episodes = episodes[:-1][valid] if episodes is not None else np.arange(len(obs))
    if max_rows is not None: obs, actions, next_obs, transition_episodes = obs[:max_rows], actions[:max_rows], next_obs[:max_rows], transition_episodes[:max_rows]
    return {"obs": obs, "action": actions, "next_obs": next_obs, "episode": transition_episodes}

def split(data, validation_fraction=0.1):
    episode = data.get("episode")
    if episode is not None:
        unique = np.unique(episode); cut_episode = max(1, int(len(unique) * (1 - validation_fraction)))
        cut = np.searchsorted(episode, unique[cut_episode - 1], side="right")
    else:
        cut = max(1, int(len(data["obs"]) * (1 - validation_fraction)))
    return ({k: v[:cut] for k, v in data.items()}, {k: v[cut:] for k, v in data.items()})
