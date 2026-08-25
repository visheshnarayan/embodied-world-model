from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .real_data import load_single_arm
from .tabular_world_model import TabularWorldModel

class LearnedDynamicsEnv(gym.Env):
    """Goal-conditioned simulator driven by a learned SingleArm model."""
    def __init__(self, model_path, dataset_path, seed=0, max_steps=100):
        self.model,self.metadata=TabularWorldModel.load(model_path); self.data=load_single_arm(dataset_path); self.rng=np.random.default_rng(seed); self.max_steps=max_steps
        self.obs_dim=self.metadata["obs_dim"]; self.action_dim=self.metadata["action_dim"]; actions=self.data["action"]
        self.action_space=spaces.Box(actions.min(0),actions.max(0),dtype=np.float32); self.observation_space=spaces.Box(-np.inf,np.inf,(self.obs_dim,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        if seed is not None: self.rng=np.random.default_rng(seed)
        episode=self.rng.choice(np.unique(self.data["episode"])); mask=self.data["episode"]==episode; self.state=self.data["obs"][mask][0].copy(); self.goal=self.data["next_obs"][mask][-1].copy(); self.steps=0; self.previous_distance=self._distance(self.state)
        return self.state.copy(),{"goal_distance":self.previous_distance,"episode":int(episode)}
    def _distance(self,state): return float(np.linalg.norm(state-self.goal)/np.sqrt(self.obs_dim))
    def step(self,action):
        action=np.clip(np.asarray(action,np.float32),self.action_space.low,self.action_space.high); self.state=self.model.predict(self.state,action); distance=self._distance(self.state); reward=self.previous_distance-distance; self.previous_distance=distance; self.steps+=1; success=distance<.05
        return self.state.copy(),float(reward),success,self.steps>=self.max_steps,{"goal_distance":distance,"success":success}
