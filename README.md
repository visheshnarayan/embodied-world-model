# Contact in Latent Space

**Accelerating robot RL through JAX kernel compilation**

Contact-rich robot manipulation is computationally expensive — thousands of environment interactions per gradient update, all bottlenecked by Python interpreter overhead. This project shows how progressively moving the PPO training loop into JAX's XLA compiler eliminates those bottlenecks, yielding a **17× end-to-end speedup with zero loss in policy quality**.

![Pipeline diagram](assets/plots/pipeline_diagram.png)

---

## What we fix

Standard RL training has three Python-level bottlenecks. Each one stalls the CPU waiting for the interpreter:

| # | Bottleneck | Python code | JAX fix |
|---|---|---|---|
| 1 | Parallel env steps | `for env in envs: env.step(a)` | `jax.vmap(step)` — one SIMD kernel |
| 2 | Rollout collection | `for t in range(H): …` | `jax.lax.scan` — single XLA graph |
| 3 | GAE + minibatch updates | Python backward loop + minibatch for-loop | `lax.scan(reverse=True)` + scanned epochs |

Fixing all three requires rewriting the environment itself (`ewm/jax_env.py`) as a pure stateless JAX function so XLA can trace through it. The NumPy-based env cannot be compiled.

---

## Results

![Speedup bar charts](assets/plots/speedup_bars.png)

### End-to-end training — same config (16 envs · 300 updates · H=128)

| Implementation | Wall time | Steps/s | Speedup |
|---|---|---|---|
| Tier 2 — Python loops + NumPy env | 113 s | 5,440 | 1× |
| Tier 4 — `lax.scan` + `vmap` JAX env | **6.7 s** | 91,700 | **17×** |

Both reach **100% task success** — the speedup is lossless.

The 17× end-to-end vs 7.8× rollout-only gap shows that the GAE backward pass and minibatch loop each contribute meaningfully: eliminating just the rollout loop undersells the gain.

### Throughput scaling (rollout collection only)

| Envs | Tier 2 steps/s | Tier 4 steps/s | Speedup |
|---|---|---|---|
| 8  | 62,049  | 423,020   | 6.8×  |
| 16 | 78,344  | 609,976   | 7.8×  |
| 64 | 97,444  | 1,098,652 | 11.3× |
| 128 | 101,447 | 1,283,713 | **12.7×** |

Tier 2 plateaus at ~100K steps/s — the Python loop is O(N) serial. Tier 4 keeps scaling because `vmap` batches all N envs into one XLA kernel.

### bfloat16 mixed precision (256 envs · 100 updates)

| Dtype | Wall time | Steps/s | Success |
|---|---|---|---|
| float32  | 24.8 s | 132,000 | 100% |
| bfloat16 | 32.2 s | 101,700 | 100% |

**bf16 is 30% slower on CPU** — x86 lacks native bf16 compute units, so XLA promotes to fp32 for matmuls. On GPU (A100/H100 Tensor Cores) the same change gives ~1.5–2× speedup.

---

## Robotics context

The task is a 2D contact-rich cube-push — a proxy for the kind of contact-rich manipulation studied in Isaac Lab with the Franka Panda. Fast RL iteration (17× per training run) makes hyperparameter sweeps and architecture searches feasible at robot-scale environment counts (256–1024 parallel sims).

![Isaac Lab Panda scene](assets/isaaclab/panda_stack_scene_overview.png)

The JAX environment (`ewm/jax_env.py`) models the same contact dynamics and success threshold as the NumPy baseline, keeping the comparison apples-to-apples. Plugging a GPU-accelerated Isaac Sim step into the same `vmap`/`scan` harness is the natural next step.

---

## Throughput tiers

```
Tier 2   NumPy env   + Python loops       baseline — matches CleanRL / SB3 style
Tier 3   JAX env     + Python rollout     vmap over envs only — slower than T2 at low N
Tier 4   JAX env     + lax.scan rollout   fully compiled — one XLA kernel per update
Tier 4b  Tier 4      + bfloat16           mixed precision — GPU benefit only
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

# Measure throughput across all tiers
python scripts/benchmark_throughput.py --steps 2000000 --envs 128
```

---

## Project layout

```
ewm/
  env.py              Tier 2 — NumPy Gymnasium env (Python-loopable)
  jax_env.py          Tier 4 env — pure-JAX stateless (vmap/scan-safe)
  world_model.py      Flax/Optax action-conditioned dynamics model
  planner.py          CEM planning over imagined rollouts

scripts/
  train_ppo.py              Tier 2 PPO training
  train_ppo_scan.py         Tier 4 PPO — vmap env, lax.scan rollout + GAE + minibatch
  benchmark_throughput.py   Tier 2 / 3 / 4 rollout throughput comparison

reports/
  findings.md         Full experimental findings with methodology notes
  experiments_log.md  Append-only log of all runs

paper/
  pipeline_diagram.tex   TikZ diagram of the PPO loop and JAX optimisations
```

---

## Roadmap

- [ ] Reproduce Tier 2→4 speedup on GPU (expected 100×+ gap)
- [ ] Tier 5: scan over updates, not just rollout — fully compiled outer loop
- [ ] Connect learned controller to Isaac Lab Panda task
- [ ] Quantization: int8 critic value estimates, gradient compression ablations
- [ ] Pixel observations + language-conditioned commands
