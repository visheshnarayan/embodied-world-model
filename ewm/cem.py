from __future__ import annotations
import numpy as np

def cem_action(env, rng, horizon=8, candidates=64, iterations=3, elite_fraction=.1):
    """Cross-Entropy Method planning entirely inside the learned dynamics env."""
    low, high = env.action_space.low, env.action_space.high; dim=low.shape[0]; mean=np.zeros((horizon,dim),np.float32); std=np.broadcast_to((high-low)/2,(horizon,dim)).copy()
    for _ in range(iterations):
        sequences=np.clip(rng.normal(mean,std,(candidates,horizon,dim)),low,high).astype(np.float32); states=np.repeat(env.state[None],candidates,axis=0); scores=np.zeros(candidates,np.float32)
        for step in range(horizon):
            states=env.model.predict_batch(states,sequences[:,step]); scores-=env._distance_batch(states)
        elite=sequences[np.argsort(scores)[-max(2,int(candidates*elite_fraction)):]]; mean=elite.mean(0); std=np.maximum(elite.std(0),.05*(high-low))
    return np.clip(mean[0],low,high).astype(np.float32)
