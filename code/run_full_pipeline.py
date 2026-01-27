"""One-click end-to-end GNNStealing + GNNFingers pipeline (no CLI args)."""

from __future__ import annotations

from pathlib import Path

from scripts.run_gnnfingers_pipeline import _build_args, _run_step, _write_csv


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    suspect_csv = repo_root / "suspect_list.csv"
    source_csv = repo_root / "source_list.csv"
    _write_csv(
        suspect_csv,
        [
            ["surrogate", "./surrogate_models/surrogate_gin_citeseer_full.pt", "gin", 256, 2, 4, 0.5, 1],
            ["neg_1", "./negative_models/neg_gat.pt", "gat", 256, 2, 4, 0.5, 0],
        ],
    )
    _write_csv(
        source_csv,
        [
            ["./target_model_gat_256/target_model_gat_citeseer_full", "gat", 256, 2, 4, 0.5],
        ],
    )

    steps = [
        {
            "name": "train_target",
            "script": "train_target_model.py",
            "args": {
                "dataset": "citeseer_full",
                "target_model": "gat",
                "num_hidden": 256,
                "num_layers": 2,
                "gpu": -1,
            },
        },
        {
            "name": "steal_gnn",
            "script": "attack.py",
            "args": {
                "dataset": "citeseer_full",
                "target_model_dim": 256,
                "num_hidden": 256,
                "target_model": "gat",
                "surrogate_model": "gin",
                "recovery_from": "prediction",
                "query_ratio": 1.0,
                "structure": "original",
                "save_surrogate": True,
                "gpu": -1,
            },
        },
        {
            "name": "build_fingerprints",
            "script": "scripts/build_fingerprints.py",
            "args": {
                "dataset": "citeseer_full",
                "source_model": "gat",
                "source_model_dir": "./target_model_gat_256",
                "output_dir": "./fingerprints",
                "mode": "embedding",
            },
        },
        {
            "name": "compute_source_scores",
            "script": "scripts/compute_source_scores.py",
            "args": {
                "dataset": "citeseer_full",
                "fingerprints": "./fingerprints",
                "source_list": "./source_list.csv",
                "mode": "embedding",
                "output": "./source_scores.txt",
            },
        },
        {
            "name": "verify_ownership",
            "script": "scripts/verify_ownership.py",
            "args": {
                "dataset": "citeseer_full",
                "fingerprints": "./fingerprints",
                "suspect_ckpt": "./surrogate_models/surrogate_gin_citeseer_full.pt",
                "suspect_model": "gin",
                "suspect_hidden": 256,
                "suspect_layers": 2,
                "suspect_heads": 4,
                "suspect_dropout": 0.5,
                "mode": "embedding",
                "source_aggregates": "./source_scores.txt",
            },
        },
        {
            "name": "eval_report",
            "script": "scripts/eval_gnnfingers_on_stealgnn.py",
            "args": {
                "dataset": "citeseer_full",
                "fingerprints": "./fingerprints",
                "suspect_list": "./suspect_list.csv",
                "mode": "embedding",
                "output_csv": "./gnnfingers_report.csv",
            },
        },
    ]

    for step in steps:
        cmd = ["python", step["script"]] + _build_args(step.get("args", {}))
        _run_step(cmd, cwd=repo_root, dry_run=False)


if __name__ == "__main__":
    main()
