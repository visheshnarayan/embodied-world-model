import numpy as np
from ewm.env import PushCubeEnv
def test_deterministic_rollout():
    e1,e2=PushCubeEnv(),PushCubeEnv(); o1,_=e1.reset(seed=7); o2,_=e2.reset(seed=7); assert np.allclose(o1,o2)
    for _ in range(5):
        x1=e1.step(np.array([.2,-.1],np.float32)); x2=e2.step(np.array([.2,-.1],np.float32)); assert np.allclose(x1[0],x2[0])
