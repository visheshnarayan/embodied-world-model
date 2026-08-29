# Contact in Latent Space

### Learning to predict, imagine, and control contact-rich robot manipulation

![Isaac Lab Panda rollout](assets/isaaclab/panda_stack_rollout_sequence.png)

*Cube-stacking experiments with a Franka Panda in Isaac Lab, using the simulation side of our JAX world-model pipeline.*

![Isaac Lab Panda scene](assets/isaaclab/panda_stack_scene_overview.png)

Contact in Latent Space is a compact research stack for model-based robot learning. It combines a deterministic Gymnasium manipulation task, an action-conditioned JAX world model, imagined CEM planning, and a real-data track built around NVIDIA's SingleArm Panda dataset.

The goal is simple: learn how actions change contact-rich manipulation, then use the learned dynamics to plan with fewer environment interactions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

python scripts/train_world_model.py --steps 1000 --seed 0
python scripts/train_mpc.py --model artifacts/world_model.pkl --episodes 30 --seed 0
python scripts/evaluate.py --controller mpc --episodes 20 --seed 0
```

## Benchmark

Run the reproducible toy-task comparison:

```bash
python scripts/run_benchmark.py \
  --seeds 0 1 2 3 4 \
  --data-episodes 1000 \
  --train-steps 10000 \
  --eval-episodes 50 \
  --candidates 128 \
  --horizon 8 \
  --output artifacts/benchmark.csv

python scripts/plot_results.py --csv artifacts/benchmark.csv
```

The benchmark compares random exploration, a scripted reference controller, and JAX world-model MPC. It records success rate, return, final distance, contact rate, episode length, and one-step prediction MSE. Generated CSVs and figures are written to `artifacts/`.

The current five-seed candidate result is:

| Method | Success rate |
| --- | ---: |
| Random | 0.0% |
| Scripted reference | 100.0% |
| World-model MPC | 25.6% ± 24.4% |

## Real-data track

The primary data track uses NVIDIA's [PhysicalAI-Robotics-Manipulation-SingleArm dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm), an Isaac Sim Franka Panda collection with state, action, RGB, and depth observations.

```bash
python scripts/download_single_arm.py --output data/single_arm
python scripts/inspect_single_arm.py data/single_arm
python scripts/train_real_world_model.py data/single_arm --max-rows 200000 --steps 2000
python scripts/evaluate_rollouts.py artifacts/single_arm_absolute.pkl data/single_arm/panda-stack-wide
python scripts/run_data_scaling.py data/single_arm/panda-stack-wide --episodes 10 25 50 100 --steps 500
```

The current real-data model predicts the next state from the current state and action. The main evaluation targets are data scaling, multi-step rollout error, and held-out task generalization.

## Isaac Lab

Isaac Lab is installed and running on an NVIDIA A10G cloud workstation. The integration contract for the Panda state/action layout is defined in [`ewm/isaac_bridge.py`](ewm/isaac_bridge.py) and [`configs/single_arm_isaac_contract.json`](configs/single_arm_isaac_contract.json).

```bash
python scripts/preflight_isaac.py
```

The next integration step is a custom Isaac Lab task that routes Panda actions through the existing world-model and benchmark interfaces.

## Project layout

- `ewm/env.py` — deterministic Gymnasium push task
- `ewm/world_model.py` — Flax/Optax action-conditioned dynamics model
- `ewm/planner.py` — batched JAX imagined rollouts and CEM planning
- `ewm/real_data.py` — SingleArm dataset loading and preprocessing
- `scripts/` — training, evaluation, benchmark, and plotting commands
- `reports/` — preprint notes and tracked benchmark tables
- `isaaclab_extension/` — Isaac Lab integration notes

## Local viewer

```bash
pip install -e '.[ui]'
streamlit run scripts/dashboard.py
```

The viewer shows dataset samples, model errors, policy comparisons, and experiment figures.

## Roadmap

- Improve multi-seed world-model MPC performance
- Add PPO as a model-free baseline
- Connect the learned controller to the Isaac Lab Panda task
- Add pixel observations, multi-task targets, and language-conditioned commands

## Reproducibility

All experiments accept explicit seeds and write metrics/checkpoints under `artifacts/`. The core toy environment is CPU-friendly; Isaac Lab requires a Linux NVIDIA workstation or cloud GPU.
