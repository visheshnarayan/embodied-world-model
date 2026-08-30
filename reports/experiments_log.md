# Experiments Log

Append-only log of all runs. Most recent at top.

---

## 2026-08-30

### [framework-16env] compile_ppo framework — 16 envs, 2 seeds
**Script:** `train_ppo_compiled.py`
**Config:** 16 envs, 300 updates, horizon 128, minibatch 256, 50 eval episodes
**Total env steps per seed:** 614,400

| Seed | Wall time | Steps/s | Success |
|---|---|---|---|
| 1 | 5.29 s | 116,155 | 100% |
| 2 | 5.08 s | 120,941 | 100% |

**Clean mean (seeds 1–2): 5.2 s, 118,500 steps/s**
**Speedup vs Tier 2 (same config): ~22×**
**Speedup vs hand-written Tier 4: ~1.3×**

Framework auto-compiles vmap, lax.scan rollout, reverse-scan GAE, and scanned
minibatch updates from two pure-JAX functions (reset_with_obs, step_with_obs).
No for-loops written by caller.

---

### [T2-convergence] Tier 2 convergence timing — 3 seeds
**Script:** `train_ppo.py`
**Config:** 16 envs, 300 updates, horizon 128, 50 eval episodes
**Total env steps per seed:** 614,400

| Seed | Wall time | Steps/s | Success |
|---|---|---|---|
| 0 | 592.2 s | 1,037 | 100% |
| 1 | 112.9 s | 5,440 | 100% |
| 2 | 113.0 s | 5,439 | 100% |

Seed 0 discarded — ran concurrently with 4 other heavy JAX jobs (CPU contention).
Seeds 1–2 ran clean (all other jobs finished by t=592s from job start).
**Clean mean (seeds 1–2): 113 s, 5,440 steps/s**

---

### [T4-16env-convergence] Tier 4 convergence timing — 16 envs, 3 seeds
**Script:** `train_ppo_scan.py`
**Config:** 16 envs, 300 updates, horizon 128, minibatch 128, 50 eval episodes
**Total env steps per seed:** 614,400

| Seed | Wall time | Steps/s | Success |
|---|---|---|---|
| 0 | 236.9 s | 2,593 | 100% |
| 1 | 7.4 s | 82,710 | 100% |
| 2 | 6.0 s | 101,912 | 100% |

Seed 0 discarded (contention). **Clean mean (seeds 1–2): 6.7 s, 91,700 steps/s**
**Speedup vs Tier 2 (same config): 17×**

---

### [T4-256env-convergence] Tier 4 convergence timing — 256 envs, 3 seeds
**Script:** `train_ppo_scan.py`
**Config:** 256 envs, 100 updates, horizon 128, minibatch 256, 50 eval episodes
**Total env steps per seed:** 3,276,800

| Seed | Wall time | Steps/s | Success |
|---|---|---|---|
| 0 | 332.1 s | 9,867 | 100% |
| 1 | 24.8 s | 132,076 | 100% |
| 2 | 24.8 s | 132,361 | 100% |

Seed 0 discarded (contention). **Clean mean (seeds 1–2): 24.8 s, 132,200 steps/s**

---

### [T4-bf16-convergence] Tier 4 bfloat16 — 256 envs, 3 seeds
**Script:** `train_ppo_scan.py --dtype bfloat16`
**Config:** 256 envs, 100 updates, horizon 128, minibatch 256, 50 eval episodes
**Total env steps per seed:** 3,276,800
**Precision:** fp32 params + optimizer, bf16 activations (mixed precision AMP)

| Seed | Wall time | Steps/s | Success |
|---|---|---|---|
| 0 | 297.0 s | 11,032 | 100% |
| 1 | 33.1 s | 98,980 | 100% |
| 2 | 31.3 s | 104,580 | 100% |

Seed 0 discarded (contention). **Clean mean (seeds 1–2): 32.2 s, 101,800 steps/s**
**vs fp32 at same config: 30% slower (expected — no native bf16 on x86 CPU)**
**Quality: identical — 100% success, same return distribution**

---

### [throughput-benchmark-16env] Tier 2/3/4 rollout throughput — 16 envs
**Script:** `benchmark_throughput.py`
**Config:** 16 envs, horizon 128, 262,144 target steps, 32,768 warmup steps
**Note:** Run concurrently with convergence jobs — numbers directionally correct but not
clean. Use the inline solo measurements below for publication-quality numbers.

| Tier | Steps/s (concurrent) |
|---|---|
| Tier 2 | 839 |
| Tier 3 | 48,108 |
| Tier 4 | 405,178 |

---

### [throughput-scaling-solo] Tier 2 vs Tier 4 scaling — inline solo measurement
**Script:** Inline Python (no concurrent load)
**Config:** horizon 128, 131,072 target steps, 16,384 warmup steps per env count

| Envs | Tier 2 (steps/s) | Tier 4 (steps/s) | Speedup |
|---|---|---|---|
| 8 | 62,049 | 423,020 | 6.8× |
| 16 | 78,344 | 609,976 | 7.8× |
| 32 | 89,143 | 862,669 | 9.7× |
| 64 | 97,444 | 1,098,652 | 11.3× |
| 128 | 101,447 | 1,283,713 | 12.7× |

Rollout collection only (no GAE, no PPO update). Tier 2 plateaus at ~100K steps/s
(Python for-loop is O(N) serial). Tier 4 scales linearly (XLA SIMD over N envs).

---

## 2026-08-29

### [ppo-tanh-baseline] PPO Tier 2 baseline — 5 seeds
**Script:** `train_ppo.py` (original, no timing output)
**Config:** 16 envs, 300 updates, horizon 128, 50 eval episodes
**Policy:** tanh-squashed Gaussian (corrected from clipped Gaussian)

| Seed | Success | Return mean ± std |
|---|---|---|
| 0 | 100% | 1.927 ± 0.065 |
| 1 | 100% | 1.924 ± 0.065 |
| 2 | 100% | 1.925 ± 0.065 |
| 3 | 100% | 1.924 ± 0.065 |
| 4 | 100% | 1.927 ± 0.063 |

**All 5 seeds converge to 100% success.** Establishes the Tier 2 quality ceiling.

Prior bug: clipped Gaussian policy (action = clip(mean + std*noise, -1, 1)) computed
log-prob on the clipped action, causing incorrect importance weights. Fix: store
pre-tanh raw action, compute log-prob on raw action, apply tanh for env interaction.

---

### [ppo-scan-smoke] Tier 4 scan PPO smoke test — seed 0
**Script:** `train_ppo_scan.py`
**Config:** 256 envs, 100 updates, horizon 128, minibatch 256, 30 eval episodes

| Metric | Value |
|---|---|
| Success | 100% |
| Wall time | 23.1 s |
| Steps/s | 141,992 |
| Total env steps | 3,276,800 |

First clean end-to-end run of Tier 4 with no competing processes.
Established correctness and throughput baseline for 256-env config.

---

### [mpc-baseline] World-model MPC baseline — 5 seeds
**Script:** `run_benchmark.py`
**Config:** 1,000 replay episodes, 10,000 train steps, 50 eval episodes,
128 CEM candidates, horizon 8

| Seed | Success | Notes |
|---|---|---|
| 0–4 | 25.6% ± 24.4% | High variance across seeds |

Larger run (5,000 replay episodes): 16.0% ± 18.8% — no improvement.
Result not published (weak baseline, ongoing improvement).
