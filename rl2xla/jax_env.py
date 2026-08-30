"""Pure-functional JAX environment equivalent to ewm.env.PushCubeEnv.

All functions are stateless and JIT/vmap/lax.scan-compatible — no NumPy,
no Python branching, no mutable state. EnvState is a NamedTuple (pytree).

API:
    state = reset(key)               # single env
    obs   = get_obs(state)           # 8-dim float32
    state, reward, done, success = step(state, action)

Batch via vmap:
    batch_reset = jax.vmap(reset)
    batch_step  = jax.vmap(step)
"""
from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Constants (must match ewm/env.py) ────────────────────────────────────────
MAX_STEPS: int = 100
CONTACT_RADIUS: float = 0.16
SUCCESS_DIST: float = 0.10
STEP_SIZE: float = 0.045
ACTION_COST: float = 0.002


class EnvState(NamedTuple):
    """Full environment state — a JAX-native pytree."""
    hand: jax.Array    # (2,) float32  end-effector position
    cube: jax.Array    # (2,) float32  cube position
    target: jax.Array  # (2,) float32  target position
    steps: jax.Array   # ()   int32    elapsed steps this episode


def reset(key: jax.Array) -> EnvState:
    """Stateless reset. Pure function — safe to vmap and jit."""
    k1, k2, k3 = jax.random.split(key, 3)
    hand = jax.random.uniform(
        k1, (2,),
        minval=jnp.array([-0.72, -0.08]),
        maxval=jnp.array([-0.62,  0.08]),
    )
    offset = jax.random.uniform(
        k2, (2,),
        minval=jnp.array([0.12, -0.04]),
        maxval=jnp.array([0.18,  0.04]),
    )
    target = jax.random.uniform(
        k3, (2,),
        minval=jnp.array([0.42, -0.18]),
        maxval=jnp.array([0.62,  0.18]),
    )
    return EnvState(
        hand=hand,
        cube=hand + offset,
        target=target,
        steps=jnp.int32(0),
    )


def get_obs(state: EnvState) -> jax.Array:
    """8-dim observation matching PushCubeEnv.observation_space.

    Layout: [hand(2), zeros(2), cube(2), target(2)]
    The two zero dimensions preserve compatibility with the NumPy env.
    """
    return jnp.concatenate([
        state.hand,
        jnp.zeros(2, jnp.float32),
        state.cube,
        state.target,
    ])


def step(
    state: EnvState, action: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
    """Stateless step. Pure function — safe to vmap and jit.

    Returns:
        next_state : EnvState
        reward     : () float32
        done       : () bool
        success    : () bool
    """
    action = jnp.clip(action.astype(jnp.float32), -1.0, 1.0)

    old_dist = jnp.linalg.norm(state.cube - state.target)

    new_hand = jnp.clip(state.hand + STEP_SIZE * action, -1.0, 1.0)

    # Contact: use new_hand position to decide whether cube moves
    contact = jnp.linalg.norm(new_hand - state.cube) < CONTACT_RADIUS
    new_cube = jnp.where(
        contact,
        jnp.clip(state.cube + STEP_SIZE * action, -0.8, 0.8),
        state.cube,
    )

    new_dist = jnp.linalg.norm(new_cube - state.target)
    success = new_dist < SUCCESS_DIST

    reward = (
        (old_dist - new_dist)
        - ACTION_COST * jnp.linalg.norm(action)
        + jnp.where(success, 1.0, 0.0)
    ).astype(jnp.float32)

    new_steps = state.steps + 1
    done = success | (new_steps >= MAX_STEPS)

    next_state = EnvState(
        hand=new_hand,
        cube=new_cube,
        target=state.target,
        steps=new_steps,
    )
    return next_state, reward, done, success


def reset_with_obs(key: jax.Array) -> tuple[EnvState, jax.Array]:
    """Wrapper satisfying the compile_ppo interface: key → (state, obs)."""
    state = reset(key)
    return state, get_obs(state)


def step_with_obs(
    state: EnvState, action: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
    """Wrapper satisfying the compile_ppo interface: (state, action) → (state, obs, reward, done)."""
    next_state, reward, done, _ = step(state, action)
    return next_state, get_obs(next_state), reward, done
