import numpy as np
from ewm.task import stacking_metrics

def test_stacking_success_geometry():
    state=np.zeros(53,np.float32); state[0:3]=[.4,.2,.055]; state[14:17]=[.4,.2,0.]
    assert stacking_metrics(state)["success"]
    state[0]=.7
    assert not stacking_metrics(state)["success"]
