# Contact in Latent Space

### Accelerated RL training through JAX kernel compilation for contact-rich robot manipulation

![Isaac Lab Panda rollout](assets/isaaclab/panda_stack_rollout_sequence.png)

*Cube-stacking experiments with a Franka Panda in Isaac Lab, using the simulation side of our JAX world-model pipeline.*

![Isaac Lab Panda scene](assets/isaaclab/panda_stack_scene_overview.png)

Contact in Latent Space studies how much RL training can be accelerated by progressively moving computation into JAX's XLA compiler — eliminating Python interpreter overhead at every stage of the training loop. The task is contact-rich robot manipulation; the contribution is a systematic ablation of each bottleneck, measured end-to-end.

---

## What we fix

Standard RL training has three Python-level bottlenecks. Each one stalls the CPU waiting for the interpreter instead of computing:

| # | Bottleneck | Traditional code | Our fix |
|---|---|---|---|
| 1 | Parallel env steps | `for env in envs: env.step(a)` | `jax.vmap(step)` — one SIMD kernel |
| 2 | Rollout collection | `for t in range(H): ...` | `jax.lax.scan` — single XLA graph |
| 3 | GAE + minibatch updates | Python backward loop + minibatch for-loop | `lax.scan(reverse=True)` + scanned epochs |

We also rewrite the environment itself (`ewm/jax_env.py`) as a pure stateless JAX function so that `vmap` and `scan` can compile through it. The NumPy-based `PushCubeEnv` cannot be traced by XLA.

---

## Key results (CPU, JAX 0.11.1)

### End-to-end training speed — same config (16 envs, 300 updates, horizon 128)

| Implementation | Wall time | Steps/s | Speedup |
|---|---|---|---|
| **Tier 2** — Python loops + NumPy env | 113 s | 5,440 | 1× |
| **Tier 4** — `lax.scan` + `vmap` JAX env | **6.7 s** | 91,700 | **17×** |

Both reach **100% success** — the speedup is lossless.
The 17× vs 7.8× rollout-only gap shows that fixing the rollout loop alone undersells the gain; GAE and minibatch loops each add further speedup when compiled.

### Throughput scaling (rollout collection only)

| Envs | Tier 2 steps/s | Tier 4 steps/s | Speedup |
|---|---|---|---|
| 8 | 62,049 | 423,020 | 6.8× |
| 16 | 78,344 | 609,976 | 7.8× |
| 64 | 97,444 | 1,098,652 | 11.3× |
| 128 | 101,447 | 1,283,713 | **12.7×** |

Tier 2 plateaus at ~100K steps/s (Python loop is O(N) serial).
Tier 4 keeps scaling because `vmap` batches all N envs into one XLA kernel.

### bfloat16 (mixed precision, 256 envs, 100 updates)

| Dtype | Wall time | Steps/s | Success |
|---|---|---|---|
| float32 | 24.8 s | 132,000 | 100% |
| bfloat16 | 32.2 s | 101,700 | 100% |

**bf16 is 30% slower on CPU** — x86 lacks native bf16 compute units so XLA promotes to fp32 for matmuls. No quality loss. On GPU (A100/H100 Tensor Cores) the same change gives ~1.5–2× speedup.

---

## Throughput tiers

```
Tier 2   NumPy env   + Python loops       (baseline — matches standard CleanRL / SB3 style)
Tier 3   JAX env     + Python rollout     (vmap over envs only — slower than T2 at low N)
Tier 4   JAX env     + lax.scan rollout   (fully compiled — one XLA kernel per update)
Tier 4b  Tier 4      + bfloat16 compute   (mixed precision — GPU benefit only)
```

```bash
# Measure throughput across tiers
python scripts/benchmark_throughput.py --steps 2000000 --envs 128

# Train Tier 4 (1024 parallel envs, fully scanned)
python scripts/train_ppo_scan.py --num-envs 1024 --updates 300

# Mixed-precision variant
python scripts/train_ppo_scan.py --num-envs 1024 --updates 300 --dtype bfloat16
```

---

## Quickstart (world-model track)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

python scripts/train_world_model.py --steps 1000 --seed 0
python scripts/train_mpc.py --model artifacts/world_model.pkl --episodes 30 --seed 0
python scripts/evaluate.py --controller mpc --episodes 20 --seed 0
```

## PPO baseline

```bash
# Tier 2 (Python/NumPy env, original)
python scripts/train_ppo.py --seed 0 --updates 300 --envs 16

# Tier 4 (fully compiled, 1024 envs)
python scripts/train_ppo_scan.py --seed 0 --updates 300 --num-envs 1024
```

## Real-data track

Uses NVIDIA's [PhysicalAI-Robotics-Manipulation-SingleArm dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm) — Isaac Sim Franka Panda with state, action, RGB, and depth observations.

```bash
python scripts/download_single_arm.py --output data/single_arm
python scripts/train_real_world_model.py data/single_arm --max-rows 200000 --steps 2000
python scripts/run_data_scaling.py data/single_arm/panda-stack-wide --episodes 10 25 50 100
```

## Isaac Lab

Integration contract defined in [`ewm/isaac_bridge.py`](ewm/isaac_bridge.py) and [`configs/single_arm_isaac_contract.json`](configs/single_arm_isaac_contract.json).

```bash
python scripts/preflight_isaac.py
```

## Project layout

```
ewm/
  env.py              Tier 2 baseline — NumPy Gymnasium env (Python-loopable)
  jax_env.py          Tier 4 env — pure-JAX stateless functions (vmap/scan-safe)
  world_model.py      Flax/Optax action-conditioned dynamics model
  planner.py          CEM planning over imagined rollouts
  real_data.py        SingleArm dataset loader

scripts/
  train_ppo.py        Tier 2 PPO — JAX network, Python env loop, Python GAE
  train_ppo_scan.py   Tier 4 PPO — vmap env, lax.scan rollout + GAE + minibatch
  benchmark_throughput.py   Tier 2 / 3 / 4 rollout throughput comparison

reports/
  findings.md         Full experimental findings with methodology notes
  experiments_log.md  Append-only log of all runs
```

## Local viewer

```bash
pip install -e '.[ui]'
streamlit run scripts/dashboard.py
```

## Roadmap

- [ ] Reproduce Tier 2→4 speedup on GPU (expected 100×+ gap)
- [ ] Quantization track: int8 critic value estimates, gradient compression ablations
- [ ] Tier 5: fully compiled outer training loop (scan over updates, not just rollout)
- [ ] Connect learned controller to Isaac Lab Panda task
- [ ] Pixel observations + language-conditioned commands

## Reproducibility

All experiments accept explicit `--seed` flags and write metrics under `artifacts/` (gitignored). Run with `--seeds 0 1 2` for multi-seed estimates. The core toy environment and all JAX tiers are CPU-runnable; Isaac Lab requires a Linux NVIDIA workstation or cloud GPU.
