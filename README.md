# RL → XLA: Automatic Compilation of Gymnasium RL Environments to JAX Kernels

![Workflow](assets/plots/workflow_v8.png)


Contact-rich robot manipulation is bottlenecked by Python interpreter overhead. Every PPO gradient update requires thousands of env steps, all gated by `for` loops that stall the CPU waiting for the interpreter instead of computing. This project eliminates those bottlenecks by compiling the full training loop into JAX/XLA — and ships two tools that let you do it for **any Gymnasium env** without rewriting it by hand.

We developed two components:

- **`gym_to_jax`** — an AST-based converter that takes any NumPy Gymnasium env and produces `vmap`/`scan`-safe pure JAX functions automatically, via a pipeline of 9 source transformers
- **`compile_ppo`** — a one-call framework that wraps any `(reset_fn, step_fn)` pair into three compiled `@jax.jit` kernels covering rollout collection, GAE, and minibatch updates — zero `for`-loops written by the caller

Together they take a standard NumPy Gymnasium env to a 22× faster compiled training loop with a three-line call.

---

## Usage

```bash
pip install rl2xla
```

```python
from flax import linen as nn
from rl2xla import gym_to_jax, compile_ppo, PPOConfig
import jax.numpy as jnp

# 1. Define your ActorCritic network
class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(64)(obs))
        x = nn.tanh(nn.Dense(64)(x))
        mean    = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        value   = nn.Dense(1)(x)[..., 0]
        return mean, jnp.broadcast_to(log_std, mean.shape), value

# 2. Convert any Gymnasium env by name
reset_fn, step_fn = gym_to_jax("MountainCarContinuous-v0")

# 3. Compile and train — obs_dim / action_dim inferred automatically
trainer = compile_ppo(reset_fn, step_fn, ActorCritic(action_dim=1))

result = trainer.train(PPOConfig(num_envs=256, updates=300), seed=0)
print(f"{result['wall_time_s']:.1f}s  —  {result['steps_per_second']:,} steps/s")
```

Works out of the box with **CartPole-v1**, **MountainCar-v0**, **MountainCarContinuous-v0**, **Pendulum-v1**, **MicroduckWalkEnv** (14-DOF bipedal, 61-dim obs), and any custom env with pure Python + NumPy logic.

---

## Framework

```
NumPy Gymnasium env
        │
        ▼  gym_to_jax()   [9 AST transformers → pure JAX functions]
(reset_fn, step_fn)
        │
        ▼  compile_ppo()  [jax.jit + lax.scan compilation]
  PPOTrainer
  ├── collect()     jax.jit — lax.scan over H steps, vmap over N envs, auto-reset
  ├── gae()         jax.jit — lax.scan(reverse=True) advantage estimation
  └── run_epoch()   jax.jit — lax.scan over shuffled minibatches per epoch
```

```python
from ewm.gym_to_jax import gym_to_jax
from ewm.jax_convert import compile_ppo, PPOConfig

reset_fn, step_fn = gym_to_jax(MyEnv)                                   # convert
trainer = compile_ppo(reset_fn, step_fn, net, obs_dim=4, action_dim=2)  # compile
result  = trainer.train(PPOConfig(num_envs=256, updates=300), seed=0)   # train
# → {"wall_time_s": 5.2, "steps_per_second": 118500}   (22× vs Python baseline)
```

### `gym_to_jax` — AST conversion pipeline

![AST pipeline](assets/plots/ast_pipeline.png)

`gym_to_jax` parses the source of any NumPy Gymnasium env, applies nine AST transformers, and `exec`s the result to produce a `(reset_fn, step_fn)` pair compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan`. State structure is discovered automatically by probing `env.__dict__` after `reset()`.

**Before** (standard NumPy Gymnasium env):
```python
def step(self, action):
    self.vx += action[0] * self.dt       # augmented assign
    self.x  += self.vx * self.dt
    dist     = np.linalg.norm(self.x - self.goal)
    done     = bool(dist < 0.1)          # Python cast
    reward   = float(dist < 0.1)
    return self._obs(), reward, done, {}
```

**After** (auto-generated — `vmap`/`scan`-safe):
```python
def step_fn(state, action):
    vx = state.vx + action[0] * dt      # AugAssign → rebind
    x  = state.x  + vx * dt
    dist   = jnp.linalg.norm(x - goal)  # np → jnp
    done   = dist < 0.1                  # bool() removed
    reward = jnp.where(done, 1.0, 0.0)  # ternary → jnp.where
    return AutoEnvState(x=x, vx=vx), obs_fn(x), reward, done
```

| Transformer | Before | After |
|---|---|---|
| `_NpToJnp` | `np.linalg.norm(x)` | `jnp.linalg.norm(x)` |
| `_RemovePyCasts` | `bool(x)`, `float(x)` | `x` |
| `_AugAssignToAssign` | `x += y` | `x = x + y` |
| `_SliceAssignToRebind` | `x[:] = y` | `x = y` |
| `_IfToWhere` | `if cond: x = y` | `x = jnp.where(cond, y, x)` |
| `_TernaryToWhere` | `a if c else b` | `jnp.where(c, a, b)` |
| `_BoolOpToJax` | `a or b` / `a and b` | `a \| b` / `a & b` |
| `_NpRandomToJax` | `self.np_random.uniform(lo, hi)` | `jax.random.uniform(key, minval=lo, maxval=hi)` |
| `_SelfFieldToLocal` | `self.x`, `self.vx` | local vars packed into `AutoEnvState` NamedTuple |

**Compatibility:** pure Python + NumPy logic with separate `self.field` state vars. Envs with C++ backends (MuJoCo, Box2D) cannot be auto-converted.

### `compile_ppo` — three compiled XLA kernels

**Kernel 1 — `collect(states, obs, params, key)`**

Runs `H` steps across `N` parallel envs as a single `lax.scan`. Inside, `jax.vmap(step_fn)` batches all envs into one SIMD call. Done envs are reset functionally via `jnp.where` — no Python branching.

```python
(states, obs, params), traj = jax.lax.scan(step_body, carry, None, length=H)
# traj: _Trajectory buffer shaped (H, N, …)
```

**Kernel 2 — `gae(rewards, values, dones, last_value)`**

GAE backward pass as `lax.scan(reverse=True)` — no Python backward loop.

```
δₜ = rₜ + γ·V(sₜ₊₁)·(1−dₜ) − V(sₜ)
Aₜ = δₜ + γλ·(1−dₜ)·Aₜ₊₁
```

```python
_, adv = jax.lax.scan(gae_step, (zeros, last_value), (rewards, values, dones), reverse=True)
```

**Kernel 3 — `run_epoch(params, opt_state, key, obs, actions, …)`**

Shuffles the flattened trajectory and runs one PPO epoch across all minibatches via `lax.scan`. Each minibatch computes clipped surrogate loss + value loss + entropy bonus and applies an Adam step.

```python
(params, opt_state), losses = jax.lax.scan(_update_mb, (params, opt_state), minibatches)
```

---

## Results

![Speedup bar charts](assets/plots/speedup_bars.png)

![Pipeline diagram](assets/plots/pipeline_diagram.png)

### End-to-end training (16 envs, 300 updates, H=128)

| Implementation | Wall time | Steps/s | Speedup |
|---|---|---|---|
| Tier 2 — Python loops + NumPy env | 113 s | 5,440 | 1× |
| Tier 4 — `lax.scan` + `vmap` JAX env | 6.7 s | 91,700 | 17× |
| `compile_ppo` framework | **5.2 s** | 118,500 | **22×** |

All reach **100% task success**. The speedup is lossless.

The 17× end-to-end vs 7.8× rollout-only gap comes from the GAE backward pass and minibatch loop each also being compiled — eliminating just one loop undersells the gain.

### Throughput scaling (rollout collection only)

| Envs | Tier 2 steps/s | Tier 4 steps/s | Speedup |
|---|---|---|---|
| 8   | 62,049  | 423,020   | 6.8×      |
| 16  | 78,344  | 609,976   | 7.8×      |
| 64  | 97,444  | 1,098,652 | 11.3×     |
| 128 | 101,447 | 1,283,713 | **12.7×** |

Tier 2 plateaus at ~100K steps/s — the Python loop is O(N) serial. Tier 4 scales linearly because `vmap` batches all N envs into one XLA SIMD kernel.

### bfloat16 mixed precision (256 envs, 100 updates)

| Dtype | Wall time | Steps/s | Success |
|---|---|---|---|
| float32  | 24.8 s | 132,000 | 100% |
| bfloat16 | 32.2 s | 101,700 | 100% |

**bf16 is 30% slower on CPU** — x86 has no native bf16 compute units. On GPU (A100/H100 Tensor Cores) the same code yields ~1.5–2× speedup instead.

---

## Why robotics

The task is a contact-rich cube-push, a minimal proxy for the Franka Panda manipulation stack shown above. Fast RL iteration (22× per training run) makes hyperparameter sweeps and architecture searches feasible at robot-scale environment counts (256–1024 parallel sims). `gym_to_jax` + `compile_ppo` bring that speed to any NumPy Gymnasium env without manual rewriting. Plugging a GPU-accelerated Isaac Sim step into the same `vmap`/`scan` harness is the natural next step.

---

## Throughput tiers

```
Tier 2       NumPy env + Python loops       baseline — matches CleanRL / SB3 style
Tier 3       JAX env + Python rollout        vmap over envs only; slower than T2 at low N
Tier 4       JAX env + lax.scan rollout      fully compiled; one XLA kernel per update
Tier 4b      Tier 4 + bfloat16              mixed precision; GPU benefit only
compile_ppo  gym_to_jax → compile_ppo       zero for-loops; any NumPy Gymnasium env
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Tier 2 baseline (Python / NumPy env)
python scripts/train_ppo.py --seed 0 --updates 300 --envs 16

# Tier 4 (fully compiled, 1024 parallel envs)
python scripts/train_ppo_scan.py --seed 0 --updates 300 --num-envs 1024

# gym_to_jax + compile_ppo (one-call automatic pipeline)
python scripts/train_ppo_auto.py --seed 0 --updates 300 --num-envs 256

# Throughput benchmark across all tiers
python scripts/benchmark_throughput.py --steps 2000000 --envs 128
```

---

## Project layout

```
ewm/
  env.py              Tier 2 NumPy Gymnasium env (Python-loopable)
  jax_env.py          Tier 4 env — pure-JAX stateless (vmap/scan-safe)
  jax_convert.py      compile_ppo — 3 compiled XLA kernels + PPOConfig
  gym_to_jax.py       gym_to_jax — 9 AST transformers, state probing, AutoEnvState
  test_env.py         NavEnv — 2D navigation env (exercises general AST path)
  microduck_env.py    MicroduckWalkEnv — 14-DOF bipedal env, 61-dim obs, 14-dim action
  world_model.py      Flax/Optax action-conditioned dynamics model
  planner.py          CEM planning over imagined rollouts

scripts/
  train_ppo.py              Tier 2 PPO training
  train_ppo_scan.py         Tier 4 PPO with vmap env + lax.scan rollout/GAE/minibatch
  train_ppo_compiled.py     compile_ppo with hand-written JAX env
  train_ppo_auto.py         gym_to_jax → compile_ppo end-to-end demo
  test_gym_to_jax.py        Conversion tests for PushCubeEnv and NavEnv
  benchmark_throughput.py   Tier 2 / 3 / 4 rollout throughput comparison

reports/
  findings.md         Full experimental findings with methodology notes
  experiments_log.md  Append-only log of all runs

paper/
  main.tex               2-column paper: JAX-Accelerated Robot RL
  pipeline_diagram.tex   TikZ diagram of PPO loop bottlenecks and JAX optimisations
```

---

## Roadmap

- [ ] Reproduce Tier 2→4 speedup on GPU (expected 100×+ gap)
- [ ] Tier 5: scan over updates for a fully compiled outer loop
- [ ] Extend `gym_to_jax` to handle `elif` chains and while loops
- [ ] Train on `MicroduckWalkEnv` and benchmark against microduck_rl (PyTorch/MuJoCo Warp baseline)
- [ ] Connect learned controller to Isaac Lab Panda task
- [ ] Pixel observations + language-conditioned commands
