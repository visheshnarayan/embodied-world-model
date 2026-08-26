# Learning to Push with a Tiny Action-Conditioned World Model

## Abstract

We study whether a compact learned dynamics model can improve sample efficiency on a deterministic contact-rich manipulation task. An 8-dimensional state, 2-dimensional action, and small MLP latent transition model are paired with random-shooting model-predictive control. We compare against random exploration and report success rate, return, final distance, contact rate, episode length, and one-step prediction MSE across three seeds.

## Figures to generate

```bash
python scripts/run_benchmark.py --seeds 0 1 2 --train-steps 0 250 1000 5000
python scripts/plot_results.py
python scripts/plot_rollout.py
```

* `artifacts/figures/benchmark_summary.png`: aggregate task metrics.
* `artifacts/figures/sample_efficiency.png`: success versus world-model updates.
* `artifacts/figures/rollout.png`: qualitative hand/cube/target trajectory.

## Results table

Fill this table from `benchmark_summary.csv`; report mean ± standard deviation over seeds.

| Method | Success ↑ | Return ↑ | Final distance ↓ | Contact rate |
|---|---:|---:|---:|---:|
| Random | — | — | — | — |
| World-model MPC | — | — | — | — |

## Preliminary run

The first executable run used 2 seeds, 5 evaluation episodes per seed, 8 shooting candidates, horizon 6, and world-model budgets of 0 and 50 updates. The aggregate CSV currently reports:

| Method | Success | Return | Final distance |
|---|---:|---:|---:|
| Random | 0.00 | -0.002 | 0.871 |
| World-model MPC | 0.00 | 0.216 | 0.646 |

This is a debugging baseline, not a positive learning result. The model-based controller improves dense return and final distance but has not yet crossed the sparse success threshold. The next scientifically important run is to add contact-balanced data or an expert warm-start, train for 1k–10k updates, and add a real PPO/SAC baseline before making a sample-efficiency claim.

## Real-data follow-up

The next experiment replaces the toy transition source with NVIDIA's SingleArm LeRobot state/action trajectories. We will first exclude videos and train a state-space action-conditioned predictor, reporting validation MSE and multi-step rollout error versus the number of real/simulated demonstration transitions. Vision and closed-loop control are follow-up stages.

The first bounded state-only run used 10 `panda-stack-wide` episodes, 611 training transitions, 76 validation transitions, and 300 updates:

| Variant | Validation MSE |
|---|---:|
| Absolute raw | 0.0496 |
| Absolute normalized | 0.6770 |
| Residual normalized | 0.9617 |

The absolute raw model is currently best, but these are target-space MSEs and the dataset split is intentionally tiny. The result motivates reporting physical-unit errors, per-dimension errors, and larger episode-level splits before drawing conclusions about preprocessing.

The first data-scaling sweep used 10/25/50/100 episodes and 500 updates. Absolute raw validation MSE was 0.0407/0.0265/0.0222/0.0220, respectively. The plot is saved as `artifacts/figures/single_arm_scaling.png`.

The learned-simulator policy smoke test used the 100-episode raw model, five episodes, and ten model steps. Mean dense goal-progress return was -0.0914 for random actions, -0.0561 for goal-conditioned behavior cloning, and -0.0277 for CEM. Sparse success was 0.0 for all methods, so this is evidence of a planning signal rather than a completed control result.

The task-aware Panda stacking smoke test used the same model with three episodes and ten model steps. Returns were -0.5276 random, -0.5322 behavior cloning, and 0.0292 CEM. Success means box 0 is within 6 cm horizontally of box 1 and 3–9 cm above it; all methods still had 0% sparse success.

## Claims boundary

This is a controlled educational benchmark, not evidence of general robot intelligence. The useful result is a transparent demonstration of the loop: collect transitions, learn action-conditioned predictions, plan in the learned model, and quantify both model quality and downstream behavior. A stronger preprint should add PPO/SAC, confidence intervals over at least five seeds, held-out target distributions, compute/runtime, and pixel or MuJoCo validation.
