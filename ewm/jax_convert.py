"""
jax_convert.py — One-call PPO compilation from plain reset/step functions.

Interface contract
------------------
reset_fn : PRNGKey → (state: pytree, obs: float32[obs_dim])
step_fn  : (state: pytree, action: float32[action_dim])
           → (state: pytree, obs: float32[obs_dim], reward: float32, done: bool)

Both functions must be pure JAX (no Python side-effects, no dynamic shapes).
The framework handles vmap, lax.scan, auto-reset, GAE, and PPO updates
automatically.  The caller never writes a for-loop.

Example
-------
from ewm import jax_env
from ewm.jax_convert import compile_ppo, PPOConfig
from flax import linen as nn

class ActorCritic(nn.Module):
    ...

trainer = compile_ppo(
    reset_fn   = jax_env.reset_with_obs,
    step_fn    = jax_env.step_with_obs,
    net        = ActorCritic(),
    obs_dim    = 8,
    action_dim = 2,
)
result = trainer.train(PPOConfig(num_envs=256, updates=300), seed=1)
print(result["wall_time_s"], result["steps_per_second"])
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


# ── Public config ─────────────────────────────────────────────────────────────

class PPOConfig(NamedTuple):
    num_envs:       int   = 16
    horizon:        int   = 128
    updates:        int   = 300
    epochs:         int   = 4
    minibatch_size: int   = 256
    gamma:          float = 0.99
    lam:            float = 0.95
    clip_ratio:     float = 0.2
    lr:             float = 3e-4
    value_coef:     float = 0.5
    entropy_coef:   float = 0.01


# ── Public entry point ────────────────────────────────────────────────────────

def compile_ppo(
    reset_fn:   Callable,
    step_fn:    Callable,
    net:        nn.Module,
    obs_dim:    int | None = None,
    action_dim: int | None = None,
) -> "PPOTrainer":
    """
    Compile a fully JAX-accelerated PPO trainer from two pure functions.

    Parameters
    ----------
    reset_fn   : key → (state, obs)
    step_fn    : (state, action) → (state, obs, reward, done)
    net        : Flax module  obs → (mean, log_std, value)
    obs_dim    : dimensionality of obs  (inferred from reset_fn if omitted)
    action_dim : dimensionality of action  (inferred from net if omitted)
    """
    if obs_dim is None:
        _, obs = reset_fn(jax.random.PRNGKey(0))
        obs_dim = int(jnp.asarray(obs).shape[0])

    if action_dim is None:
        dummy_obs    = jnp.zeros((1, obs_dim), jnp.float32)
        dummy_params = net.init(jax.random.PRNGKey(0), dummy_obs)
        mean, _, _   = net.apply(dummy_params, dummy_obs)
        action_dim   = int(jnp.asarray(mean).shape[-1])

    return PPOTrainer(reset_fn, step_fn, net, obs_dim, action_dim)


# ── Internal trajectory buffer ────────────────────────────────────────────────

class _Trajectory(NamedTuple):
    obs:       jax.Array   # (H, N, obs_dim)
    actions:   jax.Array   # (H, N, action_dim)  — raw pre-tanh
    rewards:   jax.Array   # (H, N)
    values:    jax.Array   # (H, N)
    log_probs: jax.Array   # (H, N)
    dones:     jax.Array   # (H, N)  float32


# ── Trainer ───────────────────────────────────────────────────────────────────

class PPOTrainer:
    """Compiled PPO trainer. Construct via compile_ppo()."""

    def __init__(
        self,
        reset_fn:   Callable,
        step_fn:    Callable,
        net:        nn.Module,
        obs_dim:    int,
        action_dim: int,
    ):
        self.reset_fn   = reset_fn
        self.step_fn    = step_fn
        self.net        = net
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Compiled kernels — built lazily on first train() call
        self._collect:    Callable | None = None
        self._gae:        Callable | None = None
        self._run_epoch:  Callable | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, config: PPOConfig = PPOConfig(), seed: int = 0) -> dict:
        """
        Run PPO training.  Returns a dict with params, wall_time_s,
        steps_per_second, and total_steps.
        """
        cfg = config
        key = jax.random.PRNGKey(seed)

        # Initialise network params and optimiser
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((1, self.obs_dim), jnp.float32)
        params    = self.net.init(init_key, dummy_obs)
        optimizer = optax.adam(cfg.lr)
        opt_state = optimizer.init(params)

        # Build compiled kernels (once per unique config)
        if self._collect is None:
            self._build_kernels(cfg, optimizer)

        # Reset all environments
        key, reset_key = jax.random.split(key)
        env_keys        = jax.random.split(reset_key, cfg.num_envs)
        batch_reset     = jax.vmap(self.reset_fn)
        states, obs     = batch_reset(env_keys)

        start = time.perf_counter()

        for update_idx in range(cfg.updates):

            # ── 1. Collect rollout (fully compiled scan) ──────────────────────
            key, rollout_key = jax.random.split(key)
            (states, obs, params_out), traj = self._collect(
                states, obs, params, rollout_key
            )
            params = params_out   # params pass-through (unchanged during rollout)

            # ── 2. Bootstrap value for GAE ────────────────────────────────────
            _, _, last_value = self.net.apply(params, obs)   # (N,)

            # ── 3. GAE (compiled reverse scan) ────────────────────────────────
            advantages, returns = self._gae(
                traj.rewards, traj.values, traj.dones, last_value
            )

            # ── 4. Flatten (H, N, …) → (H*N, …) ─────────────────────────────
            flat = lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
            train_obs       = flat(traj.obs)
            train_actions   = flat(traj.actions)
            train_logps     = flat(traj.log_probs)
            train_returns   = flat(returns)
            train_adv       = flat(advantages)

            # ── 5. PPO update epochs (each epoch is a compiled minibatch scan) ─
            key, epoch_key = jax.random.split(key)
            for _ in range(cfg.epochs):
                params, opt_state, epoch_key, _ = self._run_epoch(
                    params, opt_state, epoch_key,
                    train_obs, train_actions, train_logps,
                    train_returns, train_adv,
                )

            if (update_idx + 1) % max(1, cfg.updates // 10) == 0:
                print(f"  update {update_idx + 1}/{cfg.updates}")

        jax.block_until_ready(params)
        elapsed     = time.perf_counter() - start
        total_steps = cfg.updates * cfg.num_envs * cfg.horizon

        return {
            "params":            params,
            "wall_time_s":       round(elapsed, 2),
            "steps_per_second":  round(total_steps / elapsed),
            "total_steps":       total_steps,
        }

    # ── kernel builders ───────────────────────────────────────────────────────

    def _build_kernels(self, cfg: PPOConfig, optimizer: optax.GradientTransformation):
        """Compile and cache all three XLA kernels."""

        reset_fn    = self.reset_fn
        step_fn     = self.step_fn
        net         = self.net
        N           = cfg.num_envs

        batch_reset = jax.vmap(reset_fn)
        batch_step  = jax.vmap(step_fn)

        # ── helper: broadcast done mask ───────────────────────────────────────
        def _select(done: jax.Array, fresh: jax.Array, current: jax.Array) -> jax.Array:
            mask = done
            while mask.ndim < fresh.ndim:
                mask = mask[..., None]
            return jnp.where(mask, fresh, current)

        # ── kernel 1: rollout collection ─────────────────────────────────────
        # params is captured from the outer @jax.jit argument — valid JAX pattern.

        @jax.jit
        def collect(states, obs, params, key):
            def step_body(carry, _):
                states, obs, key = carry

                # Forward pass
                mean, log_std, value = net.apply(params, obs)

                # Sample action
                key, act_key = jax.random.split(key)
                noise      = jax.random.normal(act_key, mean.shape)
                raw_action = mean + jnp.exp(log_std) * noise
                action     = jnp.tanh(raw_action)

                # Log probability (on raw / pre-tanh action)
                scale    = jnp.exp(log_std)
                log_prob = jnp.sum(
                    -0.5 * ((raw_action - mean) / scale) ** 2
                    - log_std
                    - 0.5 * jnp.log(2.0 * jnp.pi),
                    axis=-1,
                )

                # Environment step (vmapped over N envs)
                new_states, new_obs, reward, done = batch_step(states, action)

                # Auto-reset completed episodes
                key, rst_key = jax.random.split(key)
                fresh_states, fresh_obs = batch_reset(jax.random.split(rst_key, N))
                next_states = jax.tree_util.tree_map(
                    lambda f, c: _select(done, f, c), fresh_states, new_states
                )
                next_obs = _select(done, fresh_obs, new_obs)

                transition = _Trajectory(
                    obs       = obs,
                    actions   = raw_action,
                    rewards   = reward.astype(jnp.float32),
                    values    = value,
                    log_probs = log_prob,
                    dones     = done.astype(jnp.float32),
                )
                return (next_states, next_obs, key), transition

            (states, obs, key), traj = jax.lax.scan(
                step_body,
                (states, obs, key),
                None,
                length=cfg.horizon,
            )
            # Return params unchanged so caller can receive updated params
            # consistently (params is a pass-through here).
            return (states, obs, params), traj

        # ── kernel 2: GAE (reverse scan) ─────────────────────────────────────

        @jax.jit
        def gae(rewards, values, dones, last_value):
            """rewards/values/dones: (H, N);  last_value: (N,)."""
            def gae_step(carry, t):
                next_gae, next_val = carry
                r, v, d = t
                delta = r + cfg.gamma * next_val * (1.0 - d) - v
                g     = delta + cfg.gamma * cfg.lam * (1.0 - d) * next_gae
                return (g, v), g

            _, adv = jax.lax.scan(
                gae_step,
                (jnp.zeros(N, jnp.float32), last_value),
                (rewards, values, dones),
                reverse=True,
            )
            return adv, adv + values   # advantages, returns

        # ── kernel 3: PPO epoch (minibatch scan) ─────────────────────────────

        def _loss_fn(params, obs, actions, old_logp, returns, advantages):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            mean, log_std, values = net.apply(params, obs)
            scale    = jnp.exp(log_std)
            log_prob = jnp.sum(
                -0.5 * ((actions - mean) / scale) ** 2
                - log_std
                - 0.5 * jnp.log(2.0 * jnp.pi),
                axis=-1,
            )
            ratio   = jnp.exp(log_prob - old_logp)
            clipped = jnp.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
            policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
            value_loss  =  0.5 * jnp.mean((returns - values) ** 2)
            entropy     = jnp.mean(
                jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)
            )
            return (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )

        def _update_mb(carry, batch):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(_loss_fn)(params, *batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        @jax.jit
        def run_epoch(params, opt_state, key, obs, actions, logps, returns, advantages):
            n      = obs.shape[0]
            n_mb   = n // cfg.minibatch_size
            key, pk = jax.random.split(key)
            perm   = jax.random.permutation(pk, n)

            def _reshape(x):
                x = x[perm][: n_mb * cfg.minibatch_size]
                return x.reshape(n_mb, cfg.minibatch_size, *x.shape[1:])

            mbs = (
                _reshape(obs), _reshape(actions), _reshape(logps),
                _reshape(returns), _reshape(advantages),
            )
            (params, opt_state), losses = jax.lax.scan(
                _update_mb, (params, opt_state), mbs
            )
            return params, opt_state, key, jnp.mean(losses)

        # Cache compiled kernels
        self._collect   = collect
        self._gae       = gae
        self._run_epoch = run_epoch
