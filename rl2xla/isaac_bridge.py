"""Dataset <-> Isaac Lab contract for the SingleArm Panda benchmark.

This module intentionally has no Isaac dependency. It is the shared, tested
contract that both the offline benchmark and a future Isaac Lab environment
must use.
"""

from __future__ import annotations

import numpy as np

from .task import stacking_metrics

SINGLE_ARM_STATE_NAMES = (
    "box0_pos_x", "box0_pos_y", "box0_pos_z",
    "box0_pos_rel_eef_x", "box0_pos_rel_eef_y", "box0_pos_rel_eef_z",
    "box0_quat_x", "box0_quat_y", "box0_quat_z", "box0_quat_w",
    "box0_quat_rel_eef_x", "box0_quat_rel_eef_y", "box0_quat_rel_eef_z", "box0_quat_rel_eef_w",
    "box1_pos_x", "box1_pos_y", "box1_pos_z",
    "box1_pos_rel_eef_x", "box1_pos_rel_eef_y", "box1_pos_rel_eef_z",
    "box1_quat_x", "box1_quat_y", "box1_quat_z", "box1_quat_w",
    "box1_quat_rel_eef_x", "box1_quat_rel_eef_y", "box1_quat_rel_eef_z", "box1_quat_rel_eef_w",
    "eef_pos_x", "eef_pos_y", "eef_pos_z",
    "eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w",
    "panda_finger_joint1_pos", "panda_finger_joint2_pos",
    "panda_finger_joint1_vel", "panda_finger_joint2_vel",
    "panda_joint1_pos", "panda_joint2_pos", "panda_joint3_pos", "panda_joint4_pos",
    "panda_joint5_pos", "panda_joint6_pos", "panda_joint7_pos",
    "panda_joint1_vel", "panda_joint2_vel", "panda_joint3_vel", "panda_joint4_vel",
    "panda_joint5_vel", "panda_joint6_vel", "panda_joint7_vel",
)
SINGLE_ARM_ACTION_NAMES = (
    "ee_pos_delta_x", "ee_pos_delta_y", "ee_pos_delta_z",
    "ee_rot_vec_delta_x", "ee_rot_vec_delta_y", "ee_rot_vec_delta_z", "gripper",
)
STATE_DIM = len(SINGLE_ARM_STATE_NAMES)
ACTION_DIM = len(SINGLE_ARM_ACTION_NAMES)


def _validate(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != dim:
        raise ValueError(f"{name} must have shape (..., {dim}), got {array.shape}")
    return array


def validate_state(state: np.ndarray) -> np.ndarray:
    """Validate and return a float32 dataset/Isaac state vector or batch."""
    return _validate(state, STATE_DIM, "state")


def validate_action(action: np.ndarray) -> np.ndarray:
    """Validate and return a float32 dataset/Isaac action vector or batch."""
    return _validate(action, ACTION_DIM, "action")


def state_to_stacking_features(state: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the task-relevant positions from a state vector or batch."""
    state = validate_state(state)
    return {
        "box0_pos": state[..., 0:3],
        "box1_pos": state[..., 14:17],
        "eef_pos": state[..., 28:31],
        "gripper": state[..., 35:37],
    }


def dataset_action_to_isaac(action: np.ndarray) -> np.ndarray:
    """Return the action in the Isaac Lab task's 7-D action convention.

    The dataset already uses end-effector position/rotation deltas plus a
    scalar gripper command, so no reorder or unit conversion is required.
    """
    return validate_action(action).copy()


def isaac_observation_to_dataset(observation: np.ndarray) -> np.ndarray:
    """Validate an Isaac observation before feeding it to the offline model."""
    return validate_state(observation).copy()


def stacking_success_from_state(state: np.ndarray) -> np.ndarray | bool:
    """Apply the benchmark's stacking success rule to one state or a batch."""
    state = validate_state(state)
    if state.ndim == 1:
        return stacking_metrics(state)["success"]
    return np.asarray([stacking_metrics(row)["success"] for row in state], dtype=bool)
