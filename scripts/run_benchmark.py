"""Run a small multi-seed benchmark and emit a long-format CSV."""
import argparse
import csv
from pathlib import Path

import numpy as np

from ewm.env import PushCubeEnv
from ewm.metrics import evaluate_controller, one_step_model_mse
from ewm.replay import batch, collect_random
from ewm.world_model import WorldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-steps", type=int, nargs="+", default=[1000], help="One or more update budgets.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", default="artifacts/benchmark.csv")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        data = collect_random(PushCubeEnv(), 100, seed)
        for train_steps in args.train_steps:
            model = WorldModel(seed)
            rng = np.random.default_rng(seed)
            for step in range(train_steps):
                model.train_step(*batch(data, 128, rng))
            learned = evaluate_controller(model, args.eval_episodes, seed, args.candidates, args.horizon)
            learned.update({"method": "world_model_mpc", "train_updates": train_steps,
                            "model_one_step_mse": one_step_model_mse(model, data)})
            rows.append(learned)
            print(seed, train_steps, learned["success_rate"])
        for name in ("random", "scripted"):
            baseline = evaluate_controller(name, args.eval_episodes, seed, args.candidates, args.horizon)
            baseline.update({"method": name, "train_updates": 0, "model_one_step_mse": np.nan})
            rows.append(baseline)
        print(seed, learned["success_rate"], rows[-2]["success_rate"], rows[-1]["success_rate"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
