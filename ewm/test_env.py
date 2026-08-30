"""
test_env.py — Simple 2D navigation env for testing gym_to_jax general conversion.

State: self.x (float), self.y (float), self.vx (float), self.vy (float)
Obs: [x, y, vx, vy]  (4-dim)
Action: [ax, ay]  (2-dim acceleration, clipped to [-1, 1])
Task: reach origin from random start, success when within 0.1 of origin
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NavEnv(gym.Env):
    """2D navigation: drive agent to origin. Pure NumPy, separate state fields."""
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(4,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.x = self.y = self.vx = self.vy = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x  = self.np_random.uniform(-1.0, 1.0)
        self.y  = self.np_random.uniform(-1.0, 1.0)
        self.vx = 0.0
        self.vy = 0.0
        self.steps = 0
        obs = np.array([self.x, self.y, self.vx, self.vy], np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.vx = float(np.clip(self.vx + 0.1 * action[0], -1.0, 1.0))
        self.vy = float(np.clip(self.vy + 0.1 * action[1], -1.0, 1.0))
        self.x  = float(np.clip(self.x  + 0.1 * self.vx,  -2.0, 2.0))
        self.y  = float(np.clip(self.y  + 0.1 * self.vy,  -2.0, 2.0))
        self.steps += 1
        dist    = float(np.sqrt(self.x**2 + self.y**2))
        success = bool(dist < 0.1)
        reward  = float(-dist)
        done    = success or self.steps >= self.max_steps
        obs     = np.array([self.x, self.y, self.vx, self.vy], np.float32)
        return obs, reward, success, self.steps >= self.max_steps, {"success": success}
