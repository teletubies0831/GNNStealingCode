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
    gnn_layers = 3
    gnn_heads = 4
    surrogate_hidden_sizes = [32, 64]
    num_steal_runs = 32
    finetune_epochs = 20
    prune_ratios = [0.3]
    structure_mode = "learned"
    classifier_epochs = 50
    steal_log_every = 50

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
        "--num-layers",
        str(gnn_layers),
        "--head",
        str(gnn_heads),
        "--gpu",
        str(gpu_id),
    ]
    steal_cmds: list[tuple[str, list[str]]] = []
    for hidden_size in surrogate_hidden_sizes:
        for run_index in range(1, num_steal_runs + 1):
            run_tag = f"h{hidden_size}_r{run_index:02d}"
            steal_cmds.append(
                (
                    f"2) run GNN stealing (hidden={hidden_size}, run={run_index})",
                    [
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
                        str(hidden_size),
                        "--num-layers",
                        str(gnn_layers),
                        "--head",
                        str(gnn_heads),
                        "--save-surrogate",
                        "--surrogate-tag",
                        run_tag,
                        "--round_index",
                        str(run_index),
                        "--structure",
                        structure_mode,
                        "--classifier-epochs",
                        str(classifier_epochs),
                        "--log-every",
                        str(steal_log_every),
                        "--gpu",
                        str(gpu_id),
                    ],
                )
            )
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
    verify_base_cmd = [
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
        "--suspect-layers",
        str(gnn_layers),
        "--suspect-heads",
        str(gnn_heads),
        "--num-classes",
        str(num_classes),
        "--gpu",
        str(gpu_id),
    ]
    postprocess_cmd = [
        "python",
        "postprocess_suspect.py",
        "--dataset",
        dataset,
        "--suspect-model",
        suspect_model,
        "--suspect-layers",
        str(gnn_layers),
        "--suspect-heads",
        str(gnn_heads),
        "--suspect-dropout",
        "0.5",
        "--num-classes",
        str(num_classes),
        "--finetune-epochs",
        str(finetune_epochs),
        "--gpu",
        str(gpu_id),
    ]

    _run_step("1) train target model", train_target_cmd, repo_root)
    for label, cmd in steal_cmds:
        _run_step(label, cmd, repo_root)
    _run_step("3) train GNNFingers + classifier", train_gnnfingers_cmd, repo_root)
    for hidden_size in surrogate_hidden_sizes:
        for run_index in range(1, num_steal_runs + 1):
            run_tag = f"h{hidden_size}_r{run_index:02d}"
            suspect_ckpt = f"./surrogate_models/surrogate_{surrogate_model}_{dataset}_{run_tag}.pt"
            verify_for_surrogate = verify_base_cmd + ["--suspect-ckpt", suspect_ckpt, "--suspect-hidden", str(hidden_size)]
            _run_step(
                f"4) verify suspect (hidden={hidden_size}, run={run_index})",
                verify_for_surrogate,
                repo_root,
            )
            for ratio in prune_ratios:
                output_tag = f"{run_tag}_prune{int(ratio * 100)}"
                postprocess_for_prune = postprocess_cmd + [
                    "--suspect-ckpt",
                    suspect_ckpt,
                    "--suspect-hidden",
                    str(hidden_size),
                    "--prune-ratio",
                    str(ratio),
                    "--output-tag",
                    output_tag,
                ]
                _run_step(
                    f"5) prune suspect (hidden={hidden_size}, run={run_index}, ratio={ratio})",
                    postprocess_for_prune,
                    repo_root,
                )
                pruned_ckpt = f"./suspect_models/suspect_{suspect_model}_{dataset}_{output_tag}.pt"
                verify_pruned = verify_base_cmd + [
                    "--suspect-ckpt",
                    pruned_ckpt,
                    "--suspect-hidden",
                    str(hidden_size),
                ]
                _run_step(
                    f"6) verify pruned suspect (hidden={hidden_size}, run={run_index}, ratio={ratio})",
                    verify_pruned,
                    repo_root,
                )
            if finetune_epochs > 0:
                output_tag = f"{run_tag}_ft{finetune_epochs}"
                postprocess_for_ft = postprocess_cmd + [
                    "--suspect-ckpt",
                    suspect_ckpt,
                    "--suspect-hidden",
                    str(hidden_size),
                    "--prune-ratio",
                    "0.0",
                    "--output-tag",
                    output_tag,
                ]
                _run_step(
                    f"7) fine-tune suspect (hidden={hidden_size}, run={run_index}, epochs={finetune_epochs})",
                    postprocess_for_ft,
                    repo_root,
                )
                ft_ckpt = f"./suspect_models/suspect_{suspect_model}_{dataset}_{output_tag}.pt"
                verify_ft = verify_base_cmd + ["--suspect-ckpt", ft_ckpt, "--suspect-hidden", str(hidden_size)]
                _run_step(
                    f"8) verify fine-tuned suspect (hidden={hidden_size}, run={run_index}, epochs={finetune_epochs})",
                    verify_ft,
                    repo_root,
                )


if __name__ == "__main__":
    main()
