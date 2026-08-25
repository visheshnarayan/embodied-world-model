import argparse,json
from pathlib import Path
import numpy as np
from ewm.real_data import load_single_arm,split
from ewm.policy import BehaviorCloningPolicy

parser=argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--episodes",type=int,default=100); parser.add_argument("--steps",type=int,default=2000); parser.add_argument("--batch-size",type=int,default=128); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--output",default="artifacts/single_arm_bc.pkl"); args=parser.parse_args()
data=load_single_arm(args.dataset,max_episodes=args.episodes); train,val=split(data); goals=np.zeros_like(train["obs"])
for episode in np.unique(train["episode"]):
    mask=train["episode"]==episode; goals[mask]=train["next_obs"][mask][-1]
policy=BehaviorCloningPolicy(train["obs"].shape[1],train["action"].shape[1],args.seed); rng=np.random.default_rng(args.seed); loss=0.
for step in range(args.steps):
    idx=rng.integers(0,len(train["obs"]),args.batch_size); loss=policy.train_step(train["obs"][idx],goals[idx],train["action"][idx])
    if step%max(1,args.steps//10)==0: print(f"step={step} train_action_mse={loss:.6f}")
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); policy.save(out,{"state_dim":int(train["obs"].shape[1]),"action_dim":int(train["action"].shape[1]),"train_rows":len(train["obs"]),"final_action_mse":loss}); json.dump({"train_rows":len(train["obs"]),"final_action_mse":loss},open(out.with_suffix(".json"),"w"),indent=2)
