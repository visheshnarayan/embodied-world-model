"""Create paper-ready aggregate plots from benchmark.csv."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv",default="artifacts/benchmark.csv"); p.add_argument("--out",default="artifacts/figures"); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.csv)
    summary=df.groupby("method").agg(success_rate=("success_rate","mean"), success_std=("success_rate","std"), return_mean=("return_mean","mean"), distance=("final_distance_mean","mean"))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    order=list(summary.index); colors=["#276FBF" if x=="world_model_mpc" else "#999999" for x in order]
    axes[0].bar(order,summary.success_rate,color=colors); axes[0].set_ylabel("Success rate"); axes[0].set_ylim(0,1); axes[0].tick_params(axis="x",rotation=25)
    axes[1].bar(order,summary.return_mean,color=colors); axes[1].set_ylabel("Episode return"); axes[1].tick_params(axis="x",rotation=25)
    axes[2].bar(order,summary.distance,color=colors); axes[2].set_ylabel("Final cube-target distance"); axes[2].tick_params(axis="x",rotation=25)
    fig.tight_layout(); fig.savefig(out/"benchmark_summary.png",dpi=220,bbox_inches="tight"); summary.to_csv(out/"benchmark_summary.csv")
    curve=df[df.method=="world_model_mpc"].groupby("train_updates").success_rate.agg(["mean","std"]).reset_index()
    if len(curve)>1:
        fig,ax=plt.subplots(figsize=(5,3.5)); x=curve.train_updates.to_numpy(); y=curve["mean"].to_numpy(); s=curve["std"].fillna(0).to_numpy()
        ax.plot(x,y,marker="o",color="#276FBF",label="world-model MPC"); ax.fill_between(x,y-s,y+s,color="#276FBF",alpha=.18); ax.set(xlabel="World-model updates",ylabel="Success rate",ylim=(0,1)); fig.tight_layout(); fig.savefig(out/"sample_efficiency.png",dpi=220,bbox_inches="tight")
    print(summary)
if __name__=="__main__": main()
