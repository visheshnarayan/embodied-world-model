"""Run the state-only data scaling experiment used by the paper."""
import argparse, csv
from pathlib import Path
import numpy as np
from rl2xla.real_data import load_single_arm, split
from rl2xla.preprocess import fit_standardizer, transform
from rl2xla.tabular_world_model import TabularWorldModel

parser=argparse.ArgumentParser(); parser.add_argument("path"); parser.add_argument("--episodes",type=int,nargs="+",default=[10,25,50,100]); parser.add_argument("--steps",type=int,default=500); parser.add_argument("--batch-size",type=int,default=128); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--output",default="artifacts/single_arm_scaling.csv"); args=parser.parse_args(); rows=[]
for episode_count in args.episodes:
    raw=load_single_arm(args.path,max_episodes=episode_count); train_raw,val_raw=split(raw)
    for variant,normalize,target in (("absolute_raw",False,"absolute"),("absolute_normalized",True,"absolute"),("residual_normalized",True,"residual")):
        stats=fit_standardizer(train_raw) if normalize else {k:np.zeros(v.shape[1],np.float32) for k,v in train_raw.items() if k in ("obs","action","next_obs")}
        if normalize: train,target_stats=transform(train_raw,stats,target); val,_=transform(val_raw,stats,target,target_stats)
        else: train,val=train_raw,val_raw
        model=TabularWorldModel(train["obs"].shape[1],train["action"].shape[1],args.seed); rng=np.random.default_rng(args.seed); loss=0.0
        for _ in range(args.steps):
            idx=rng.integers(0,len(train["obs"]),args.batch_size); loss=model.train_step(train["obs"][idx],train["action"][idx],train["next_obs"][idx])
        pred=np.stack([model.predict(o,a) for o,a in zip(val["obs"],val["action"])]); val_mse=float(np.mean((pred-val["next_obs"])**2)); row={"episodes":episode_count,"variant":variant,"train_transitions":len(train["obs"]),"val_transitions":len(val["obs"]),"train_mse":loss,"val_mse":val_mse}; rows.append(row); print(row)
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,"w",newline="") as f: writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
