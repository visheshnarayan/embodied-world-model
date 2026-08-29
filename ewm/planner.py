from __future__ import annotations
import numpy as np
from .env import PushCubeEnv
from .world_model import WorldModel
def mpc_action(env:PushCubeEnv,model:WorldModel|None,rng,candidates=128,horizon=8):
    best=np.zeros(2,np.float32); best_score=-np.inf
    if model is not None:
        sequences=rng.uniform(-1,1,(candidates,horizon,2)).astype(np.float32)
        sim_states=np.repeat(env.state[None],candidates,axis=0)
        scores=np.zeros(candidates,np.float32)
        active=np.ones(candidates,bool)
        for step in range(horizon):
            next_states,rewards=model.predict_batch(sim_states,sequences[:,step])
            scores += rewards.astype(np.float32)*active
            sim_states=next_states
            active &= np.linalg.norm(sim_states[:,4:6]-sim_states[:,6:8],axis=1) >= .10
        index=int(np.argmax(scores))
        return sequences[index,0]
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
