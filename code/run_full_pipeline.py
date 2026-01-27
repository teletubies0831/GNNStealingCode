"""One-click end-to-end GNNFingers pipeline (no CLI args)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import subprocess


def main() -> None:
    repo_root = Path(__file__).resolve().parent
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
        "-1",
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
        "-1",
    ]

    print("[pipeline]", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=str(repo_root), check=True)
    print("[pipeline]", " ".join(verify_cmd))
    subprocess.run(verify_cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
