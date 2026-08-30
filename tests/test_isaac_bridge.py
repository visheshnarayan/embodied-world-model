import numpy as np
import pytest

from rl2xla.isaac_bridge import (
    ACTION_DIM,
    SINGLE_ARM_ACTION_NAMES,
    SINGLE_ARM_STATE_NAMES,
    STATE_DIM,
    dataset_action_to_isaac,
    stacking_success_from_state,
    state_to_stacking_features,
    validate_action,
    validate_state,
)


def test_contract_dimensions_and_features():
    assert STATE_DIM == 53
    assert ACTION_DIM == 7
    assert len(SINGLE_ARM_STATE_NAMES) == STATE_DIM
    assert len(SINGLE_ARM_ACTION_NAMES) == ACTION_DIM
    state = np.arange(STATE_DIM, dtype=np.float32)
    features = state_to_stacking_features(state)
    np.testing.assert_array_equal(features["box0_pos"], [0, 1, 2])
    np.testing.assert_array_equal(features["box1_pos"], [14, 15, 16])
    np.testing.assert_array_equal(dataset_action_to_isaac(np.ones(ACTION_DIM)), np.ones(ACTION_DIM))


def test_contract_supports_batches_and_success_rule():
    states = np.zeros((2, STATE_DIM), dtype=np.float32)
    states[0, 0:3] = [0.0, 0.0, 0.10]
    states[0, 14:17] = [0.01, 0.01, 0.05]
    states[1, 0:3] = [0.4, 0.4, 0.10]
    assert stacking_success_from_state(states).tolist() == [True, False]
    assert validate_state(states).dtype == np.float32


def test_contract_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="state"):
        validate_state(np.zeros(52))
    with pytest.raises(ValueError, match="action"):
        validate_action(np.zeros(8))
