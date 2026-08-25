import argparse,json
from pathlib import Path
import numpy as np
from ewm.env import PushCubeEnv
from ewm.planner import mpc_action
from ewm.world_model import WorldModel
def main():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=30); p.add_argument('--seed',type=int,default=0); p.add_argument('--model',default='artifacts/world_model.pkl'); a=p.parse_args(); rng=np.random.default_rng(a.seed); wins=0; model=WorldModel(a.seed); model.load(a.model)
    for ep in range(a.episodes):
        env=PushCubeEnv(); env.reset(seed=a.seed+ep); done=False
        while not done: _,_,term,trunc,info=env.step(mpc_action(env,model,rng)); done=term or trunc
        wins+=int(info['success'])
    result={'episodes':a.episodes,'success_rate':wins/a.episodes,'controller':'mpc','seed':a.seed}; Path('artifacts').mkdir(exist_ok=True); json.dump(result,open('artifacts/mpc_metrics.json','w'),indent=2); print(result)
if __name__=='__main__': main()
