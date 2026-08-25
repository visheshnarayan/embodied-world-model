import argparse, csv
from pathlib import Path
import numpy as np
from ewm.real_data import load_single_arm
from ewm.tabular_world_model import TabularWorldModel

parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("dataset"); parser.add_argument("--episodes",type=int,default=20); parser.add_argument("--horizons",type=int,nargs="+",default=[1,5,10]); parser.add_argument("--output",default="artifacts/rollout_error.csv"); args=parser.parse_args()
model,_=TabularWorldModel.load(args.model); data=load_single_arm(args.dataset); rows=[]
for horizon in args.horizons:
    errors=[]
    for episode in np.unique(data["episode"])[:args.episodes]:
        mask=data["episode"]==episode; obs=data["obs"][mask]; actions=data["action"][mask]; targets=data["next_obs"][mask]; limit=min(horizon,len(actions)); predicted=obs[0].copy()
        for step in range(limit): predicted=model.predict(predicted,actions[step])
        errors.append(float(np.mean((predicted-targets[limit-1])**2)))
    row={"horizon":horizon,"episodes":len(errors),"mse":float(np.mean(errors)),"mse_std":float(np.std(errors))}; rows.append(row); print(row)
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,"w",newline="") as f: writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
