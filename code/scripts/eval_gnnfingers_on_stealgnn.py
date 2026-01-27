"""Evaluate GNNFingers on surrogate and negative models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from gnnfingers.io import load_fingerprints
from gnnfingers.model_utils import load_model_from_checkpoint
from gnnfingers.verifier import OwnershipVerifier, VerificationConfig
from src.utils import load_npz_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate GNNFingers on StealGNN outputs")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--fingerprints", type=str, required=True)
    parser.add_argument("--suspect-list", type=str, required=True)
    parser.add_argument("--mode", type=str, default="embedding")
    parser.add_argument("--similarity", type=str, default="cosine")
    parser.add_argument("--output-csv", type=str, default="./gnnfingers_report.csv")
    parser.add_argument("--target-tnr", type=float, default=0.99)
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _load_suspects(path: str) -> List[Tuple[str, str, str, int, int, int, float, int]]:
    records: List[Tuple[str, str, str, int, int, int, float, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, ckpt, model, hidden, layers, heads, dropout, label = line.split(",")
        records.append(
            (
                name.strip(),
                ckpt.strip(),
                model.strip(),
                int(hidden),
                int(layers),
                int(heads),
                float(dropout),
                int(label),
            )
        )
    return records


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)
    data, num_classes = load_npz_graph(args.dataset)

    graphs, meta, source_signature = load_fingerprints(Path(args.fingerprints))
    verifier = OwnershipVerifier(VerificationConfig(mode=args.mode, similarity=args.similarity, target_tnr=args.target_tnr))

    suspects = _load_suspects(args.suspect_list)
    scores = []
    labels = []
    names = []
    for name, ckpt, model, hidden, layers, heads, dropout, label in suspects:
        suspect_model = load_model_from_checkpoint(
            Path(ckpt),
            model,
            data.num_features,
            num_classes,
            hidden,
            layers,
            heads,
            dropout,
        ).to(device)
        suspect_signature = verifier.encode_responses(suspect_model, graphs, meta, device)
        score = verifier.score_signatures(source_signature, suspect_signature)
        scores.append(score.aggregate)
        labels.append(label)
        names.append(name)

    negatives = [s for s, y in zip(scores, labels) if y == 0]
    threshold = verifier.threshold_from_negatives(negatives)
    verdicts = ["stolen" if s >= threshold else "not_stolen" for s in scores]

    lines = ["name,score,label,verdict"]
    for name, score, label, verdict in zip(names, scores, labels, verdicts):
        lines.append(f"{name},{score:.6f},{label},{verdict}")
    Path(args.output_csv).write_text("\n".join(lines), encoding="utf-8")

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        print(f"AUC={auc:.4f}")
        print(f"TPR@FPR~1%: {np.interp(0.01, fpr, tpr):.4f}")
    else:
        print("AUC not computed: labels are not diverse.")

    print(f"Saved report to {args.output_csv}")


if __name__ == "__main__":
    main()
