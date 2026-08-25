"""Save a qualitative trajectory figure and optional frame strip."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ewm.env import PushCubeEnv
from ewm.planner import mpc_action
from ewm.world_model import WorldModel

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="artifacts/world_model.pkl"); p.add_argument("--seed",type=int,default=0); p.add_argument("--out",default="artifacts/figures/rollout.png"); p.add_argument("--candidates",type=int,default=4); p.add_argument("--horizon",type=int,default=4); a=p.parse_args()
    env=PushCubeEnv(); obs,_=env.reset(seed=a.seed); model=WorldModel(a.seed); model.load(a.model); rng=np.random.default_rng(a.seed); hand=[env.state[:2].copy()]; cube=[env.state[4:6].copy()]
    done=False
    while not done:
        _,_,term,trunc,_=env.step(mpc_action(env,model,rng,a.candidates,a.horizon)); hand.append(env.state[:2].copy()); cube.append(env.state[4:6].copy()); done=term or trunc
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(5,5)); hand=np.asarray(hand); cube=np.asarray(cube); target=env.state[6:8]
    ax.plot(hand[:,0],hand[:,1],label="end-effector",color="#276FBF"); ax.plot(cube[:,0],cube[:,1],label="cube",color="#D1495B"); ax.scatter(*target,s=150,marker="*",label="target",color="#2A9D8F"); ax.set(xlim=(-1,1),ylim=(-1,1),aspect="equal",title="World-model MPC rollout"); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=220)
if __name__=="__main__": main()
