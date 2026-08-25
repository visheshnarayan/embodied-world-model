from pathlib import Path
import argparse,json,numpy as np
from ewm.env import PushCubeEnv
from ewm.replay import collect_random,batch
from ewm.world_model import WorldModel
def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=1000); p.add_argument('--seed',type=int,default=0); p.add_argument('--episodes',type=int,default=100); a=p.parse_args(); out=Path('artifacts'); out.mkdir(exist_ok=True)
    data=collect_random(PushCubeEnv(),a.episodes,a.seed); model=WorldModel(a.seed); rng=np.random.default_rng(a.seed); losses=[]
    for step in range(a.steps):
        losses.append(model.train_step(*batch(data,128,rng)))
        if step%max(1,a.steps//10)==0: print(f'step={step} loss={losses[-1]:.5f}')
    model.save(out/'world_model.pkl'); json.dump({'seed':a.seed,'final_loss':losses[-1]},open(out/'world_model_metrics.json','w'),indent=2)
if __name__=='__main__': main()

