import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download SingleArm state/action files first.")
parser.add_argument("--output", default="data/PhysicalAI-Robotics-Manipulation-SingleArm")
parser.add_argument("--include-videos", action="store_true")
args = parser.parse_args()
patterns = ["data/**/*.parquet", "meta/**", "README.md"]
if args.include_videos: patterns += ["**/*.mp4"]
print(snapshot_download("nvidia/PhysicalAI-Robotics-Manipulation-SingleArm", repo_type="dataset", local_dir=args.output, allow_patterns=patterns))

