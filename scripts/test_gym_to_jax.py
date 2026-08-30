"""
Test that gym_to_jax works on both PushCubeEnv (packed state fallback)
and NavEnv (general separate-field AST path).
"""
import jax
import jax.numpy as jnp
import numpy as np

from ewm.env import PushCubeEnv
from ewm.test_env import NavEnv
from ewm.gym_to_jax import gym_to_jax, ConversionError


def test_env(env_class, name):
    print(f"\n{'='*50}")
    print(f"Testing gym_to_jax({name})")
    print(f"{'='*50}")
    try:
        reset_fn, step_fn = gym_to_jax(env_class)
        print(f"  Conversion: OK")
    except ConversionError as e:
        print(f"  ConversionError: {e}")
        return False

    # Test reset
    key = jax.random.PRNGKey(0)
    state, obs = reset_fn(key)
    print(f"  reset_fn -> state={type(state).__name__}, obs.shape={obs.shape}")

    # Test step
    action = jnp.zeros(env_class().action_space.shape, jnp.float32)
    new_state, new_obs, reward, done = step_fn(state, action)
    print(f"  step_fn  -> obs.shape={new_obs.shape}, reward={reward:.4f}, done={done}")

    # Test vmap
    keys = jax.random.split(key, 4)
    states, obss = jax.vmap(reset_fn)(keys)
    print(f"  vmap(reset_fn, 4 envs) -> obs.shape={obss.shape}")

    actions = jnp.zeros((4,) + env_class().action_space.shape, jnp.float32)
    new_states, new_obss, rewards, dones = jax.vmap(step_fn)(states, actions)
    print(f"  vmap(step_fn, 4 envs) -> obs.shape={new_obss.shape}")

    # Test jit
    jit_reset = jax.jit(reset_fn)
    jit_step  = jax.jit(step_fn)
    state2, obs2 = jit_reset(key)
    s3, o3, r3, d3 = jit_step(state2, action)
    print(f"  jit(reset_fn) + jit(step_fn): OK")

    print(f"  PASS")
    return True


if __name__ == "__main__":
    ok1 = test_env(PushCubeEnv, "PushCubeEnv")
    ok2 = test_env(NavEnv,      "NavEnv")
    print(f"\nResults: PushCubeEnv={'PASS' if ok1 else 'FAIL'}  NavEnv={'PASS' if ok2 else 'FAIL'}")
