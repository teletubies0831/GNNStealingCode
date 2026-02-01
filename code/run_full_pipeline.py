"""One-click end-to-end pipeline with target training, stealing, and verification."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import subprocess
import torch


def _run_step(label: str, command: list[str], repo_root: Path) -> None:
    print(f"[pipeline] {label}")
    print("[pipeline]", " ".join(command))
    subprocess.run(command, cwd=str(repo_root), check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    gpu_id = 0 if torch.cuda.is_available() else -1

    dataset = "citeseer_full"
    target_model = "gat"
    surrogate_model = "gin"
    suspect_model = "gin"
    num_classes = 6
    gnnfingers_out = "./gnnfingers_out"
    target_hidden = 256
    surrogate_hidden = 256
    structure_mode = "learned"
    classifier_epochs = 50

    if target_model == surrogate_model:
        raise ValueError("target_model and surrogate_model must be different for stealing.")

    train_target_cmd = [
        "python",
        "train_target_model.py",
        "--dataset",
        dataset,
        "--target-model",
        target_model,
        "--num-hidden",
        str(target_hidden),
        "--gpu",
        str(gpu_id),
    ]
    steal_cmd = [
        "python",
        "attack.py",
        "--dataset",
        dataset,
        "--target-model",
        target_model,
        "--target-model-dim",
        str(target_hidden),
        "--surrogate-model",
        surrogate_model,
        "--num-hidden",
        str(surrogate_hidden),
        "--save-surrogate",
        "--structure",
        structure_mode,
        "--classifier-epochs",
        str(classifier_epochs),
        "--gpu",
        str(gpu_id),
    ]
    train_gnnfingers_cmd = [
        "python",
        "train_gnnfingers.py",
        "--task",
        "node_classification",
        "--dataset",
        dataset,
        "--target-model",
        target_model,
        "--num-fingerprints",
        "32",
        "--output-dir",
        gnnfingers_out,
        "--gpu",
        str(gpu_id),
    ]
    verify_cmd = [
        "python",
        "verify_suspect.py",
        "--fingerprints",
        f"{gnnfingers_out}/fingerprints.pt",
        "--univerifier",
        f"{gnnfingers_out}/univerifier.pt",
        "--lambda-threshold",
        "0.5",
        "--suspect-model",
        suspect_model,
        "--num-classes",
        str(num_classes),
        "--gpu",
        str(gpu_id),
    ]

    _run_step("1) train target model", train_target_cmd, repo_root)
    _run_step("2) run GNN stealing (surrogate training)", steal_cmd, repo_root)
    _run_step("3) train GNNFingers + classifier", train_gnnfingers_cmd, repo_root)
    _run_step("4) verify suspect", verify_cmd, repo_root)


if __name__ == "__main__":
    main()
