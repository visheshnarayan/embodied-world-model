import argparse
import numpy as np
from rl2xla.model_env import LearnedDynamicsEnv
from rl2xla.policy import BehaviorCloningPolicy

parser=argparse.ArgumentParser(); parser.add_argument("policy"); parser.add_argument("model"); parser.add_argument("dataset"); parser.add_argument("--episodes",type=int,default=20); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--max-steps",type=int,default=30); args=parser.parse_args(); policy,_=BehaviorCloningPolicy.load(args.policy); env=LearnedDynamicsEnv(args.model,args.dataset,args.seed,max_steps=args.max_steps,task="stacking"); returns=[]; successes=0
for episode in range(args.episodes):
    env.reset(seed=args.seed+episode); done=False; total=0.
    while not done:
        action=policy.predict(env.state,env.goal); action=np.clip(action,env.action_space.low,env.action_space.high); _,reward,term,trunc,info=env.step(action); total+=reward; done=term or trunc
    returns.append(total); successes+=int(info["success"])
print({"controller":"behavior_cloning","episodes":args.episodes,"mean_return":float(np.mean(returns)),"success_rate":successes/args.episodes})
