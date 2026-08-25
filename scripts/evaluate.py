import argparse,numpy as np
from pathlib import Path
from ewm.env import PushCubeEnv
from ewm.planner import mpc_action
from ewm.world_model import WorldModel
p=argparse.ArgumentParser(); p.add_argument('--controller',choices=['mpc','random'],default='mpc'); p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed',type=int,default=0); p.add_argument('--model',default='artifacts/world_model.pkl'); a=p.parse_args(); rng=np.random.default_rng(a.seed); wins=0
model=WorldModel(a.seed)
if a.controller=='mpc' and Path(a.model).exists(): model.load(a.model)
for ep in range(a.episodes):
    env=PushCubeEnv(); env.reset(seed=a.seed+ep); done=False
    while not done:
        action=mpc_action(env,model,rng) if a.controller=='mpc' else rng.uniform(-1,1,2); _,_,term,trunc,info=env.step(action); done=term or trunc
    wins+=info['success']
print({'controller':a.controller,'success_rate':wins/a.episodes})
