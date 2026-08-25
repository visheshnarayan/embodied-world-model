import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(); parser.add_argument("path", nargs="?", default="data/PhysicalAI-Robotics-Manipulation-SingleArm"); args = parser.parse_args()
files = sorted((Path(args.path) / "data").rglob("*.parquet")); print(f"parquet_files={len(files)}")
if files:
    frame = pd.read_parquet(files[0]); print(frame.head()); print("columns:", list(frame.columns)); print("rows:", len(frame))

