from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PushCubeEnv(gym.Env):
    """Deterministic 2-D point end-effector pushing a cube to a target."""
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, max_steps: int = 100, render_mode: str | None = None):
        self.max_steps, self.render_mode = max_steps, render_mode
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.np_random = np.random.default_rng(0); self.state = np.zeros(8, np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); rng=self.np_random
        # Start close enough for contact to be learnable from random exploration,
        # while keeping target position randomized for held-out evaluation.
        hand=np.array([rng.uniform(-.72,-.62), rng.uniform(-.08,.08)])
        cube=hand+np.array([rng.uniform(.12,.18), rng.uniform(-.04,.04)])
        target=np.array([rng.uniform(.42,.62), rng.uniform(-.18,.18)])
        self.state=np.array([*hand,0,0,*cube,*target],np.float32); self.steps=0
        return self.state.copy(), {"success":False}
    def step(self, action):
        action=np.clip(np.asarray(action,np.float32),-1,1); hand,cube,target=self.state[:2],self.state[4:6],self.state[6:8]
        old=np.linalg.norm(cube-target); hand[:]=np.clip(hand+.045*action,-1,1)
        contact = bool(np.linalg.norm(hand-cube)<.16)
        if contact: cube[:]=np.clip(cube+.045*action,-.8,.8)
        new=np.linalg.norm(cube-target); success=bool(new<.10)
        reward=float(old-new)-.002*np.linalg.norm(action)+(1.0 if success else 0.0); self.steps+=1
        return self.state.copy(),reward,success,self.steps>=self.max_steps,{"success":success,"distance":float(new),"contact":contact}
    def render(self):
        image=np.ones((64,64,3),np.uint8)*245
        def px(p): return np.clip(((p+1)*.5*63).astype(int),0,63)
        for pos,color,r in ((self.state[6:8],(70,180,90),4),(self.state[4:6],(210,80,60),4),(self.state[:2],(60,90,210),3)):
            x,y=px(pos); image[max(0,y-r):y+r+1,max(0,x-r):x+r+1]=color
        return image
