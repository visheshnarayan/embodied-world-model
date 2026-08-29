from __future__ import annotations
import numpy as np
from .env import PushCubeEnv
from .world_model import WorldModel
def mpc_action(env:PushCubeEnv,model:WorldModel|None,rng,candidates=128,horizon=8):
    best=np.zeros(2,np.float32); best_score=-np.inf
    if model is not None:
        mean=np.zeros((horizon,2),np.float32)
        std=np.full((horizon,2),.8,np.float32)
        for _ in range(4):
            sequences=np.clip(rng.normal(mean,std,(candidates,horizon,2)),-1,1).astype(np.float32)
            sim_states=np.repeat(env.state[None],candidates,axis=0)
            scores=np.zeros(candidates,np.float32)
            active=np.ones(candidates,bool)
            for step in range(horizon):
                old_dist=np.linalg.norm(sim_states[:,4:6]-sim_states[:,6:8],axis=1)
                next_states,_=model.predict_batch(sim_states,sequences[:,step])
                new_dist=np.linalg.norm(next_states[:,4:6]-next_states[:,6:8],axis=1)
                scores += (old_dist-new_dist-.002*np.linalg.norm(sequences[:,step],axis=1))*active
                scores += (new_dist<.10).astype(np.float32)*active
                sim_states=next_states
                active &= new_dist >= .10
            elite=np.argsort(scores)[-max(8,candidates//10):]
            mean=sequences[elite].mean(axis=0)
            std=np.maximum(sequences[elite].std(axis=0),.05)
        return mean[0]
    for _ in range(candidates):
        seq=rng.uniform(-1,1,(horizon,2)).astype(np.float32); sim_state=env.state.copy(); score=0.0
        for action in seq:
            if model is None:
                sim=PushCubeEnv(env.max_steps); sim.state=sim_state.copy(); sim.steps=env.steps
                next_state,reward,term,_,_=sim.step(action)
            else:
                next_state,reward=model.predict(sim_state,action); term=bool(np.linalg.norm(next_state[4:6]-next_state[6:8])<.10)
            score+=reward; sim_state=next_state
            if term: break
        if score>best_score: best_score,best=score,seq[0]
    return best
