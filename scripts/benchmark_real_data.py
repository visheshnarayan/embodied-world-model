"""Small preprocessing benchmark for the SingleArm state/action track."""
import argparse, csv
from pathlib import Path
import numpy as np
from rl2xla.real_data import load_single_arm, split
from rl2xla.preprocess import fit_standardizer, transform
from rl2xla.tabular_world_model import TabularWorldModel

parser=argparse.ArgumentParser(); parser.add_argument("path"); parser.add_argument("--episodes",type=int,default=10); parser.add_argument("--steps",type=int,default=300); parser.add_argument("--batch-size",type=int,default=64); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--output",default="artifacts/single_arm_benchmark.csv"); args=parser.parse_args()
raw=load_single_arm(args.path,max_episodes=args.episodes); train_raw,val_raw=split(raw); rows=[]
for name,normalize,target in (("absolute_raw",False,"absolute"),("absolute_normalized",True,"absolute"),("residual_normalized",True,"residual")):
    stats=fit_standardizer(train_raw) if normalize else {k:np.zeros(v.shape[1],np.float32) for k,v in train_raw.items() if k in ("obs","action","next_obs")}
    if normalize: train,target_stats=transform(train_raw,stats,target); val,_=transform(val_raw,stats,target,target_stats)
    else: train,val=train_raw,val_raw
    model=TabularWorldModel(train["obs"].shape[1],train["action"].shape[1],args.seed); rng=np.random.default_rng(args.seed); losses=[]
    for _ in range(args.steps):
        idx=rng.integers(0,len(train["obs"]),args.batch_size); losses.append(model.train_step(train["obs"][idx],train["action"][idx],train["next_obs"][idx]))
    pred=np.stack([model.predict(o,a) for o,a in zip(val["obs"],val["action"])]); mse=float(np.mean((pred-val["next_obs"])**2)); rows.append({"variant":name,"episodes":args.episodes,"train_transitions":len(train["obs"]),"val_transitions":len(val["obs"]),"train_mse":losses[-1],"val_mse":mse}); print(rows[-1])
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,"w",newline="") as f: writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
