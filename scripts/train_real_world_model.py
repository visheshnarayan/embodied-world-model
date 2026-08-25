import argparse, json
from pathlib import Path
import numpy as np
from ewm.real_data import load_single_arm, split
from ewm.preprocess import fit_standardizer, transform
from ewm.tabular_world_model import TabularWorldModel

parser=argparse.ArgumentParser(); parser.add_argument("path"); parser.add_argument("--steps",type=int,default=2000); parser.add_argument("--max-rows",type=int,default=200000); parser.add_argument("--max-episodes",type=int); parser.add_argument("--batch-size",type=int,default=256); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--target",choices=["absolute","residual"],default="absolute"); parser.add_argument("--normalize",action="store_true"); parser.add_argument("--output",default="artifacts/single_arm_world_model.pkl"); args=parser.parse_args()
raw=load_single_arm(args.path,max_rows=args.max_rows,max_episodes=args.max_episodes); train_raw,val_raw=split(raw); stats=fit_standardizer(train_raw) if args.normalize else {k: np.zeros(v.shape[1],np.float32) for k,v in train_raw.items() if k in ("obs","action","next_obs")}
if args.normalize: train,train_target_stats=transform(train_raw,stats,args.target); val,_=transform(val_raw,stats,args.target,train_target_stats)
else: train,train_target_stats=train_raw,{"target_mean":np.zeros(train_raw["next_obs"].shape[1]),"target_std":np.ones(train_raw["next_obs"].shape[1])}; val=val_raw
rng=np.random.default_rng(args.seed); model=TabularWorldModel(train["obs"].shape[1],train["action"].shape[1],args.seed); losses=[]
for step in range(args.steps):
    idx=rng.integers(0,len(train["obs"]),args.batch_size); losses.append(model.train_step(train["obs"][idx],train["action"][idx],train["next_obs"][idx]))
    if step%max(1,args.steps//10)==0: print(f"step={step} train_mse={losses[-1]:.6f}")
limit=min(5000,len(val["obs"])); pred=np.stack([model.predict(o,a) for o,a in zip(val["obs"][:limit],val["action"][:limit])]); val_mse=float(np.mean((pred-val["next_obs"][:limit])**2)); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); metadata={"obs_dim":int(train["obs"].shape[1]),"action_dim":int(train["action"].shape[1]),"train_rows":len(train["obs"]),"val_rows":len(val["obs"]),"val_mse":val_mse,"target":args.target,"normalized":args.normalize}; model.save(out,metadata); json.dump(metadata,open(out.with_suffix(".json"),"w"),indent=2); print(metadata)
