from __future__ import annotations
import numpy as np
from .env import PushCubeEnv
def collect_random(env: PushCubeEnv, episodes: int, seed: int=0):
    rng=np.random.default_rng(seed); rows=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=seed+ep); done=False
        while not done:
            action=rng.uniform(-1,1,2).astype(np.float32); nxt,reward,term,trunc,_=env.step(action)
            rows.append((obs,action,reward,nxt,term or trunc)); obs,done=nxt,term or trunc
    return {k:np.asarray([r[i] for r in rows],np.float32) for i,k in enumerate(("obs","action","reward","next_obs","done"))}
def batch(data,size,rng):
    idx=rng.integers(0,len(data["obs"]),size=size)
    return tuple(data[k][idx] for k in ("obs","action","reward","next_obs"))

