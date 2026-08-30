"""
rl2xla — Automatic compilation of Gymnasium RL environments to JAX/XLA kernels.

Public API
----------
    gym_to_jax(env_class)
        Convert any NumPy Gymnasium env to pure-JAX (reset_fn, step_fn).

    compile_ppo(reset_fn, step_fn, net, obs_dim, action_dim)
        Compile a fully JAX-accelerated PPO trainer from two pure functions.
        Returns a PPOTrainer with three compiled @jax.jit kernels.

    PPOConfig
        NamedTuple of PPO hyperparameters (num_envs, horizon, updates, …).

Example
-------
    from rl2xla import gym_to_jax, compile_ppo, PPOConfig

    reset_fn, step_fn = gym_to_jax(MyEnv)
    trainer = compile_ppo(reset_fn, step_fn, net, obs_dim=4, action_dim=2)
    result  = trainer.train(PPOConfig(num_envs=256, updates=300), seed=0)
"""

from rl2xla.gym_to_jax import gym_to_jax, ConversionError
from rl2xla.jax_convert import compile_ppo, PPOConfig, PPOTrainer

__version__ = "0.1.0"
__all__ = ["gym_to_jax", "ConversionError", "compile_ppo", "PPOConfig", "PPOTrainer"]
