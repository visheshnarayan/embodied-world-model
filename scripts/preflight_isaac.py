"""Check whether the local machine is ready to run the Isaac Lab handoff."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/single_arm/panda-stack-wide"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/single_arm_100ep.pkl"))
    args = parser.parse_args()
    checks = {
        "dataset": args.dataset.exists(),
        "model": args.model.exists(),
        "isaacsim": importlib.util.find_spec("isaacsim") is not None,
        "isaaclab": importlib.util.find_spec("isaaclab") is not None,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    for name, ok in checks.items():
        print(f"{name:12} {'OK' if ok else 'MISSING'}")
    if not checks["isaacsim"] or not checks["isaaclab"] or not checks["nvidia_smi"]:
        print("\nLocal data/model checks are available; Isaac Lab requires a Linux/Windows NVIDIA RTX machine.")
        return 0
    print("\nIsaac runtime appears available. Run the extension instructions in isaaclab_extension/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
