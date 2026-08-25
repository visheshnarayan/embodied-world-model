import argparse
import numpy as np
from ewm.model_env import LearnedDynamicsEnv
from ewm.cem import cem_action

parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("dataset"); parser.add_argument("--episodes",type=int,default=20); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--controller",choices=["random","cem"],default="cem"); parser.add_argument("--planning-candidates",type=int,default=8); parser.add_argument("--planning-horizon",type=int,default=4); parser.add_argument("--planning-iterations",type=int,default=2); parser.add_argument("--max-steps",type=int,default=30); args=parser.parse_args(); env=LearnedDynamicsEnv(args.model,args.dataset,args.seed,max_steps=args.max_steps); returns=[]; successes=0; rng=np.random.default_rng(args.seed)
for episode in range(args.episodes):
    env.reset(seed=args.seed+episode); done=False; total=0.
    while not done:
        action=env.action_space.sample() if args.controller=="random" else cem_action(env,rng,args.planning_horizon,args.planning_candidates,args.planning_iterations); _,reward,term,trunc,info=env.step(action); total+=reward; done=term or trunc
    returns.append(total); successes+=int(info["success"])
print({"controller":args.controller,"episodes":args.episodes,"mean_return":float(np.mean(returns)),"success_rate":successes/args.episodes})
