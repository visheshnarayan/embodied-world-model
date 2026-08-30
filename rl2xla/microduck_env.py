"""
MicroduckWalkEnv — Simplified 14-DOF bipedal locomotion env.

Inspired by microduck_rl (pollen-robotics/microduck_rl).
Implemented in pure NumPy so it is compatible with gym_to_jax.

Observation (61-dim, matches the microduck real-deployment contract):
    base_ang_vel    3   gyro reading (body frame)
    gravity_body    3   unit gravity vector projected into body frame
    joint_pos_rel  14   joint positions minus default (standing) pose
    joint_vel      14   joint velocities
    last_action    14   previous action sent
    cmd_vel         3   [vx, vy, ω_z] velocity command
    cmd_head        4   head joint targets
    cmd_body        6   body CoM pose target [x,y,z,roll,pitch,yaw]

Action (14-dim):
    Joint position targets in [-1, 1] rad, added to DEFAULT_POSE.

Physics (simplified, pure NumPy):
    • Joints: PD control + first-order inertia (no rigid body solver)
    • Base linear: gravity + contact spring-damper when z < GROUND_Z
    • Base angular: coupling from net leg joint torques → tilt dynamics
    • Quaternion: 1st-order Euler integration + normalisation
    Dynamics run at 50 Hz (DT = 0.02 s).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Robot constants ────────────────────────────────────────────────────────────

NUM_JOINTS  = 14
OBS_DIM     = 61
DT          = np.float32(0.02)        # 50 Hz control

DEFAULT_POSE  = np.zeros(NUM_JOINTS, np.float32)
JOINT_LOWER   = np.full(NUM_JOINTS, -1.5, np.float32)
JOINT_UPPER   = np.full(NUM_JOINTS,  1.5, np.float32)
VEL_LIMIT     = np.float32(10.0)       # rad / s

# Joint PD gains and inertia
KP            = np.float32(20.0)
KD            = np.float32(0.5)
JOINT_INERTIA = np.float32(0.002)     # kg · m²

# Base body physics
MASS          = np.float32(0.8)       # kg
BASE_INERTIA  = np.float32(0.01)      # kg · m²  (isotropic)
GROUND_Z      = np.float32(0.15)      # nominal standing height, m
GROUND_K      = np.float32(5000.0)    # contact spring, N / m
GROUND_D      = np.float32(100.0)     # contact damper, N · s / m

# Friction mask: damp xy velocity each step, keep z unchanged
FRICTION_MASK = np.array([
    1.0 - 0.8 * float(DT),
    1.0 - 0.8 * float(DT),
    1.0,
], np.float32)

# Torque coupling coefficient from leg joints to base tilt
JOINT_TORQUE_COUPLING = np.float32(0.001)
ANG_DAMP              = np.float32(0.5)   # angular velocity damping per second


class MicroduckWalkEnv(gym.Env):
    """
    Simplified 14-DOF bipedal locomotion environment.

    Compatible with gym_to_jax for joint dynamics and reward computation.
    Quaternion integration is inlined (no helper method calls) for full
    AST-transformer compatibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self.max_steps = max_steps

        self.action_space      = spaces.Box(-1.0, 1.0, (NUM_JOINTS,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), np.float32
        )

        # State fields — all initialised in reset().
        # base_* : base-body kinematics
        self.base_pos     = np.zeros(3, np.float32)            # world position
        self.base_quat    = np.array([1., 0., 0., 0.], np.float32)  # [w,x,y,z]
        self.base_lin_vel = np.zeros(3, np.float32)            # world frame
        self.base_ang_vel = np.zeros(3, np.float32)            # body frame
        # joint state
        self.joint_pos    = np.zeros(NUM_JOINTS, np.float32)   # absolute positions
        self.joint_vel    = np.zeros(NUM_JOINTS, np.float32)
        self.last_action  = np.zeros(NUM_JOINTS, np.float32)
        # command
        self.cmd_vel      = np.zeros(3, np.float32)            # [vx, vy, ω]
        self.cmd_head     = np.zeros(4, np.float32)
        self.cmd_body     = np.zeros(6, np.float32)
        # step counter
        self.steps        = 0

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        # Projected gravity: rotate world [0,0,-1] into body frame via conjugate quat
        qw = self.base_quat[0]
        qx = self.base_quat[1]
        qy = self.base_quat[2]
        qz = self.base_quat[3]
        # Conjugate: (qw, -qx, -qy, -qz) rotates world→body
        cw, cx, cy, cz = qw, -qx, -qy, -qz
        # Rotate [0, 0, -1] (only z component contributes)
        gx = 2.0 * (cx * cz + cw * cy) * (-1.0)
        gy = 2.0 * (cy * cz - cw * cx) * (-1.0)
        gz = (1.0 - 2.0 * (cx * cx + cy * cy)) * (-1.0)
        gravity_body = np.array([gx, gy, gz], np.float32)

        return np.concatenate([
            self.base_ang_vel,                       # 3
            gravity_body,                            # 3
            self.joint_pos - DEFAULT_POSE,           # 14
            self.joint_vel,                          # 14
            self.last_action,                        # 14
            self.cmd_vel,                            # 3
            self.cmd_head,                           # 4
            self.cmd_body,                           # 6
        ]).astype(np.float32)                        # total = 61

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Base position: nominal height + small xy perturbation
        self.base_pos = np.array([
            self.np_random.uniform(-0.05, 0.05),
            self.np_random.uniform(-0.05, 0.05),
            float(GROUND_Z),
        ], np.float32)

        # Slight random initial tilt (roll / pitch only)
        roll  = self.np_random.uniform(-0.05, 0.05)
        pitch = self.np_random.uniform(-0.05, 0.05)
        cr = np.cos(roll  / 2.0)
        sr = np.sin(roll  / 2.0)
        cp = np.cos(pitch / 2.0)
        sp = np.sin(pitch / 2.0)
        self.base_quat = np.array([
            float(cr * cp),
            float(sr * cp),
            float(cr * sp),
            float(-sr * sp),
        ], np.float32)
        self.base_quat = self.base_quat / np.linalg.norm(self.base_quat)

        self.base_lin_vel = np.zeros(3, np.float32)
        self.base_ang_vel = np.zeros(3, np.float32)

        # Joints near default pose with small noise
        self.joint_pos   = self.np_random.uniform(-0.05, 0.05, NUM_JOINTS).astype(np.float32)
        self.joint_vel   = np.zeros(NUM_JOINTS, np.float32)
        self.last_action = np.zeros(NUM_JOINTS, np.float32)

        # Random velocity command
        self.cmd_vel  = np.array([
            self.np_random.uniform(-0.4,  0.4),
            self.np_random.uniform(-0.3,  0.3),
            self.np_random.uniform(-1.0,  1.0),
        ], np.float32)
        self.cmd_head = self.np_random.uniform(-0.2, 0.2, 4).astype(np.float32)
        self.cmd_body = self.np_random.uniform(-0.1, 0.1, 6).astype(np.float32)

        self.steps = 0
        return self._get_obs(), {}

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action):
        action = np.clip(action.astype(np.float32), -1.0, 1.0)

        # ── Joint PD dynamics ──────────────────────────────────────────────────
        target    = DEFAULT_POSE + action
        torque    = KP * (target - self.joint_pos) - KD * self.joint_vel
        jnt_acc   = torque / JOINT_INERTIA
        self.joint_vel = np.clip(
            self.joint_vel + jnt_acc * DT, -VEL_LIMIT, VEL_LIMIT
        )
        self.joint_pos = np.clip(
            self.joint_pos + self.joint_vel * DT, JOINT_LOWER, JOINT_UPPER
        )

        # ── Base linear dynamics ───────────────────────────────────────────────
        z = self.base_pos[2]
        in_contact  = z < GROUND_Z
        contact_fz  = float(in_contact) * (
            GROUND_K * (GROUND_Z - z) - GROUND_D * self.base_lin_vel[2]
        )
        lin_acc = np.array([0.0, 0.0, -9.81 + contact_fz / MASS], np.float32)
        self.base_lin_vel = (self.base_lin_vel + lin_acc * DT) * FRICTION_MASK
        self.base_pos     = self.base_pos + self.base_lin_vel * DT
        # Floor clamp (no digging)
        base_pos_z = self.base_pos[2]
        self.base_pos = np.array([
            self.base_pos[0],
            self.base_pos[1],
            max(0.0, base_pos_z),
        ], np.float32)

        # ── Base angular dynamics ──────────────────────────────────────────────
        # Simplified: net roll/pitch torque from left vs right leg joints
        leg_l = np.sum(torque[0:5])
        leg_r = np.sum(torque[5:10])
        ang_torque = np.array([
            (leg_l - leg_r) * JOINT_TORQUE_COUPLING,
            (leg_l + leg_r) * JOINT_TORQUE_COUPLING,
            0.0,
        ], np.float32)
        ang_acc           = ang_torque / BASE_INERTIA
        self.base_ang_vel = self.base_ang_vel + ang_acc * DT
        self.base_ang_vel = self.base_ang_vel * (1.0 - ANG_DAMP * DT)

        # ── Quaternion integration (inlined for gym_to_jax compatibility) ──────
        qw  = self.base_quat[0]
        qx  = self.base_quat[1]
        qy  = self.base_quat[2]
        qz  = self.base_quat[3]
        wx  = self.base_ang_vel[0]
        wy  = self.base_ang_vel[1]
        wz  = self.base_ang_vel[2]
        qdw = 0.5 * (-qx * wx - qy * wy - qz * wz)
        qdx = 0.5 * ( qw * wx + qy * wz - qz * wy)
        qdy = 0.5 * ( qw * wy - qx * wz + qz * wx)
        qdz = 0.5 * ( qw * wz + qx * wy - qy * wx)
        nqw = qw + qdw * DT
        nqx = qx + qdx * DT
        nqy = qy + qdy * DT
        nqz = qz + qdz * DT
        qnorm = np.sqrt(nqw * nqw + nqx * nqx + nqy * nqy + nqz * nqz)
        self.base_quat = np.array([
            nqw / qnorm, nqx / qnorm, nqy / qnorm, nqz / qnorm,
        ], np.float32)

        # ── Reward ─────────────────────────────────────────────────────────────
        vel_err = np.linalg.norm(np.array([
            self.base_lin_vel[0] - self.cmd_vel[0],
            self.base_lin_vel[1] - self.cmd_vel[1],
            self.base_ang_vel[2] - self.cmd_vel[2],
        ], np.float32))
        vel_reward = np.exp(-vel_err)

        # Upright: gravity projected into body frame (z-component → 1 when upright)
        cw2 = self.base_quat[0]
        cx2 = -self.base_quat[1]
        cy2 = -self.base_quat[2]
        cz2 = -self.base_quat[3]
        gz2 = (1.0 - 2.0 * (cx2 * cx2 + cy2 * cy2)) * (-1.0)
        upright_reward = gz2

        smooth_penalty = -0.01 * np.linalg.norm(action - self.last_action)
        alive_bonus    = np.float32(0.1)
        reward = float(vel_reward + upright_reward + smooth_penalty + alive_bonus)

        # ── Termination ────────────────────────────────────────────────────────
        fallen     = bool(self.base_pos[2] < 0.05)
        self.steps = self.steps + 1
        terminated = fallen
        truncated  = bool(self.steps >= self.max_steps)

        self.last_action = action.copy()
        return self._get_obs(), reward, terminated, truncated, {}
