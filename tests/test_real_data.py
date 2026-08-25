import numpy as np
import pandas as pd
from ewm.real_data import load_single_arm

def test_lerobot_transition_loader(tmp_path):
    root = tmp_path / "dataset" / "data"; root.mkdir(parents=True)
    frame = pd.DataFrame({
        "episode_index": [0, 0, 0, 1, 1],
        "frame_index": [0, 1, 2, 0, 1],
        "observation.state": [np.zeros(3), np.ones(3), np.ones(3)*2, np.ones(3)*4, np.ones(3)*5],
        "action": [np.zeros(2), np.ones(2), np.ones(2)*2, np.ones(2)*4, np.ones(2)*5],
    })
    frame.to_parquet(root / "chunk-000.parquet")
    data = load_single_arm(tmp_path / "dataset")
    assert data["obs"].shape == (3, 3)
    assert np.allclose(data["next_obs"][0], 1)
    assert np.allclose(data["next_obs"][-1], 5)
