from __future__ import annotations
import numpy as np

def cem_action(env, rng, horizon=8, candidates=64, iterations=3, elite_fraction=.1):
    """Cross-Entropy Method planning entirely inside the learned dynamics env."""
    low, high = env.action_space.low, env.action_space.high; dim=low.shape[0]; mean=np.zeros((horizon,dim),np.float32); std=np.broadcast_to((high-low)/2,(horizon,dim)).copy()
    for _ in range(iterations):
        sequences=np.clip(rng.normal(mean,std,(candidates,horizon,dim)),low,high).astype(np.float32); scores=[]
        for sequence in sequences:
            state=env.state.copy(); score=0.
            for action in sequence:
                state=env.model.predict(state,action); score-=env._distance(state)
            scores.append(score)
        elite=sequences[np.argsort(scores)[-max(2,int(candidates*elite_fraction)):]]; mean=elite.mean(0); std=np.maximum(elite.std(0),.05*(high-low))
    return np.clip(mean[0],low,high).astype(np.float32)
