from __future__ import annotations
import numpy as np

def stacking_metrics(state):
    state=np.asarray(state); box0=state[0:3]; box1=state[14:17]; xy=float(np.linalg.norm(box0[:2]-box1[:2])); gap=float(box0[2]-box1[2]); distance=float(np.sqrt((xy/0.08)**2+((gap-0.055)/0.04)**2)); success=bool(xy<0.06 and 0.03<gap<0.09)
    return {"xy_distance":xy,"vertical_gap":gap,"task_distance":distance,"success":success}
