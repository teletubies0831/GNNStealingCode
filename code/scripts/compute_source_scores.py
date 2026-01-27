"""Compute source aggregate scores for one-class thresholding."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from gnnfingers.io import load_fingerprints
from gnnfingers.model_utils import load_model_from_checkpoint
from gnnfingers.verifier import OwnershipVerifier, VerificationConfig
from src.utils import load_npz_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute source aggregate scores")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--fingerprints", type=str, required=True)
    parser.add_argument("--source-list", type=str, required=True)
    parser.add_argument("--mode", type=str, default="embedding")
    parser.add_argument("--similarity", type=str, default="cosine")
    parser.add_argument("--output", type=str, default="./source_scores.txt")
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _load_sources(path: str) -> List[Tuple[str, str, int, int, int, float]]:
    records: List[Tuple[str, str, int, int, int, float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ckpt, model, hidden, layers, heads, dropout = line.split(",")
        records.append((ckpt.strip(), model.strip(), int(hidden), int(layers), int(heads), float(dropout)))
    return records


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)
    data, num_classes = load_npz_graph(args.dataset)

    graphs, meta, source_signature = load_fingerprints(Path(args.fingerprints))
    verifier = OwnershipVerifier(VerificationConfig(mode=args.mode, similarity=args.similarity))

    scores = []
    for ckpt, model, hidden, layers, heads, dropout in _load_sources(args.source_list):
        source_model = load_model_from_checkpoint(
            Path(ckpt),
            model,
            data.num_features,
            num_classes,
            hidden,
            layers,
            heads,
            dropout,
        ).to(device)
        source_signature_i = verifier.encode_responses(source_model, graphs, meta, device)
        score = verifier.score_signatures(source_signature, source_signature_i)
        scores.append(score.aggregate)

    Path(args.output).write_text("\n".join(f"{s:.6f}" for s in scores), encoding="utf-8")
    print(f"Wrote {len(scores)} source aggregate scores to {args.output}")


if __name__ == "__main__":
    main()
