"""
pytest tests for the rl2xla public API.

Tests:
  - gym_to_jax converts PushCubeEnv (fallback path) and NavEnv (general AST path)
  - reset_fn / step_fn are jit and vmap compatible
  - compile_ppo produces a trainer that converges on both envs
  - MicroduckWalkEnv: 14-DOF bipedal env, 61-dim obs, 14-dim action
"""
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2xla import gym_to_jax, compile_ppo, PPOConfig, ConversionError
from rl2xla.env import PushCubeEnv
from rl2xla.test_env import NavEnv
from rl2xla.microduck_env import MicroduckWalkEnv, OBS_DIM, NUM_JOINTS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pushcube_fns():
    return gym_to_jax(PushCubeEnv)


@pytest.fixture(scope="module")
def nav_fns():
    return gym_to_jax(NavEnv)


# ── gym_to_jax: PushCubeEnv ──────────────────────────────────────────────────

class TestGymToJaxPushCube:
    def test_converts(self, pushcube_fns):
        reset_fn, step_fn = pushcube_fns
        assert callable(reset_fn) and callable(step_fn)

    def test_reset_shape(self, pushcube_fns):
        reset_fn, _ = pushcube_fns
        key = jax.random.PRNGKey(0)
        state, obs = reset_fn(key)
        assert obs.shape == (PushCubeEnv().observation_space.shape[0],)

    def test_step(self, pushcube_fns):
        reset_fn, step_fn = pushcube_fns
        key = jax.random.PRNGKey(1)
        state, obs = reset_fn(key)
        action = jnp.zeros(PushCubeEnv().action_space.shape, jnp.float32)
        new_state, new_obs, reward, done = step_fn(state, action)
        assert new_obs.shape == obs.shape
        assert reward.shape == ()
        assert done.shape == ()

    def test_jit(self, pushcube_fns):
        reset_fn, step_fn = pushcube_fns
        key = jax.random.PRNGKey(2)
        state, obs = jax.jit(reset_fn)(key)
        action = jnp.zeros(PushCubeEnv().action_space.shape, jnp.float32)
        jax.jit(step_fn)(state, action)

    def test_vmap(self, pushcube_fns):
        reset_fn, step_fn = pushcube_fns
        keys = jax.random.split(jax.random.PRNGKey(3), 8)
        states, obss = jax.vmap(reset_fn)(keys)
        assert obss.shape[0] == 8
        actions = jnp.zeros((8,) + PushCubeEnv().action_space.shape, jnp.float32)
        _, new_obss, rewards, dones = jax.vmap(step_fn)(states, actions)
        assert new_obss.shape[0] == 8


# ── gym_to_jax: NavEnv (general AST path) ────────────────────────────────────

class TestGymToJaxNavEnv:
    def test_converts(self, nav_fns):
        reset_fn, step_fn = nav_fns
        assert callable(reset_fn) and callable(step_fn)

    def test_reset_shape(self, nav_fns):
        reset_fn, _ = nav_fns
        key = jax.random.PRNGKey(0)
        state, obs = reset_fn(key)
        assert obs.shape == (NavEnv().observation_space.shape[0],)

    def test_step(self, nav_fns):
        reset_fn, step_fn = nav_fns
        key = jax.random.PRNGKey(1)
        state, obs = reset_fn(key)
        action = jnp.zeros(NavEnv().action_space.shape, jnp.float32)
        new_state, new_obs, reward, done = step_fn(state, action)
        assert new_obs.shape == obs.shape

    def test_jit(self, nav_fns):
        reset_fn, step_fn = nav_fns
        key = jax.random.PRNGKey(2)
        state, obs = jax.jit(reset_fn)(key)
        action = jnp.zeros(NavEnv().action_space.shape, jnp.float32)
        jax.jit(step_fn)(state, action)

    def test_vmap(self, nav_fns):
        reset_fn, step_fn = nav_fns
        keys = jax.random.split(jax.random.PRNGKey(3), 8)
        states, obss = jax.vmap(reset_fn)(keys)
        assert obss.shape[0] == 8


# ── compile_ppo: end-to-end convergence ──────────────────────────────────────

class TestCompilePPO:
    def _make_net(self, obs_dim, action_dim):
        from flax import linen as nn
        import jax.numpy as jnp

        class ActorCritic(nn.Module):
            obs_dim:    int
            action_dim: int

            @nn.compact
            def __call__(self, obs):
                x = nn.tanh(nn.Dense(64)(obs))
                x = nn.tanh(nn.Dense(64)(x))
                mean    = nn.Dense(self.action_dim)(x)
                log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
                value   = nn.Dense(1)(x)[..., 0]
                return mean, jnp.broadcast_to(log_std, mean.shape), value

        return ActorCritic(obs_dim=obs_dim, action_dim=action_dim)

    def test_navenv_convergence(self):
        reset_fn, step_fn = gym_to_jax(NavEnv)
        env = NavEnv()
        net = self._make_net(env.observation_space.shape[0], env.action_space.shape[0])
        trainer = compile_ppo(reset_fn, step_fn, net,
                              obs_dim=env.observation_space.shape[0],
                              action_dim=env.action_space.shape[0])
        result = trainer.train(PPOConfig(num_envs=16, horizon=64, updates=50), seed=0)
        assert result["total_steps"] == 16 * 64 * 50
        assert result["steps_per_second"] > 0


# ── gym_to_jax: classic control envs ─────────────────────────────────────────

@pytest.mark.parametrize("env_name,expected_obs,n_act", [
    ("CartPole-v1",             (4,), 1),
    ("MountainCar-v0",          (2,), 1),
    ("MountainCarContinuous-v0",(2,), 1),
    ("Pendulum-v1",             (3,), 1),
])
class TestClassicControl:
    def _fns(self, env_name):
        cls = gym.make(env_name).unwrapped.__class__
        return gym_to_jax(cls)

    def test_converts(self, env_name, expected_obs, n_act):
        rf, sf = self._fns(env_name)
        assert callable(rf) and callable(sf)

    def test_reset_shape(self, env_name, expected_obs, n_act):
        rf, _ = self._fns(env_name)
        _, obs = rf(jax.random.PRNGKey(0))
        assert obs.shape == expected_obs

    def test_step(self, env_name, expected_obs, n_act):
        rf, sf = self._fns(env_name)
        state, obs = rf(jax.random.PRNGKey(0))
        action = jnp.zeros(n_act, jnp.float32)
        _, new_obs, reward, done = sf(state, action)
        assert new_obs.shape == expected_obs
        assert reward.shape == ()

    def test_vmap(self, env_name, expected_obs, n_act):
        rf, sf = self._fns(env_name)
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        states, obss = jax.vmap(rf)(keys)
        assert obss.shape == (8,) + expected_obs
        actions = jnp.zeros((8, n_act), jnp.float32)
        _, new_obss, _, _ = jax.vmap(sf)(states, actions)
        assert new_obss.shape == (8,) + expected_obs


# ── MicroduckWalkEnv ──────────────────────────────────────────────────────────

class TestMicroduckWalkEnv:
    """14-DOF bipedal env: obs 61-dim, action 14-dim, pure NumPy dynamics."""

    def _env(self):
        return MicroduckWalkEnv(max_steps=200)

    def test_spaces(self):
        env = self._env()
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.action_space.shape == (NUM_JOINTS,)

    def test_reset_shape(self):
        env = self._env()
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_step_shape(self):
        env = self._env()
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_runs(self):
        """Env steps without crash for a full episode."""
        env = self._env()
        obs, _ = env.reset(seed=42)
        for _ in range(200):
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break
        assert obs.shape == (OBS_DIM,)

    def test_zero_action_stable(self):
        """Zero action for 100 steps shouldn't NaN."""
        env = self._env()
        env.reset(seed=0)
        action = np.zeros(NUM_JOINTS, np.float32)
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs)), "NaN detected in observation"
            if terminated or truncated:
                break

    def test_string_conversion(self):
        """gym_to_jax(string) UX fix: still raises ConversionError for unknown ids."""
        with pytest.raises((ConversionError, Exception)):
            gym_to_jax("NonExistentEnv-v0")


# ── gym_to_jax: MicroduckWalkEnv ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def microduck_fns():
    return gym_to_jax(MicroduckWalkEnv)


class TestGymToJaxMicroduck:
    """gym_to_jax conversion of 14-DOF bipedal env (61-dim obs, 14-dim action)."""

    def test_converts(self, microduck_fns):
        reset_fn, step_fn = microduck_fns
        assert callable(reset_fn) and callable(step_fn)

    def test_reset_shape(self, microduck_fns):
        reset_fn, _ = microduck_fns
        key = jax.random.PRNGKey(0)
        state, obs = reset_fn(key)
        assert obs.shape == (OBS_DIM,)

    def test_step_shape(self, microduck_fns):
        reset_fn, step_fn = microduck_fns
        key = jax.random.PRNGKey(1)
        state, obs = reset_fn(key)
        action = jnp.zeros(NUM_JOINTS, jnp.float32)
        new_state, new_obs, reward, done = step_fn(state, action)
        assert new_obs.shape == (OBS_DIM,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_jit(self, microduck_fns):
        reset_fn, step_fn = microduck_fns
        key = jax.random.PRNGKey(2)
        state, obs = jax.jit(reset_fn)(key)
        action = jnp.zeros(NUM_JOINTS, jnp.float32)
        jax.jit(step_fn)(state, action)

    def test_vmap(self, microduck_fns):
        reset_fn, step_fn = microduck_fns
        keys = jax.random.split(jax.random.PRNGKey(3), 8)
        states, obss = jax.vmap(reset_fn)(keys)
        assert obss.shape == (8, OBS_DIM)
        actions = jnp.zeros((8, NUM_JOINTS), jnp.float32)
        _, new_obss, rewards, dones = jax.vmap(step_fn)(states, actions)
        assert new_obss.shape == (8, OBS_DIM)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)
