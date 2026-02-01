"""One-click end-to-end GNNFingers pipeline (no CLI args)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import subprocess
import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    gpu_id = 0 if torch.cuda.is_available() else -1
    train_cmd = [
        "python",
        "train_gnnfingers.py",
        "--task",
        "node_classification",
        "--dataset",
        "citeseer_full",
        "--target-model",
        "gat",
        "--output-dir",
        "./gnnfingers_out",
        "--gpu",
        str(gpu_id),
    ]
    verify_cmd = [
        "python",
        "verify_suspect.py",
        "--fingerprints",
        "./gnnfingers_out/fingerprints.pt",
        "--univerifier",
        "./gnnfingers_out/univerifier.pt",
        "--lambda-threshold",
        "0.5",
        "--suspect-model",
        "gin",
        "--num-classes",
        "6",
        "--gpu",
        str(gpu_id),
    ]

    print("[pipeline]", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=str(repo_root), check=True)
    print("[pipeline]", " ".join(verify_cmd))
    subprocess.run(verify_cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
