"""Verify model ownership using GNNFingers."""

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
    parser = argparse.ArgumentParser("Verify GNNFingers ownership")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--fingerprints", type=str, required=True)
    parser.add_argument("--mode", type=str, default="embedding")
    parser.add_argument("--similarity", type=str, default="cosine")
    parser.add_argument("--suspect-ckpt", type=str, required=True)
    parser.add_argument("--suspect-model", type=str, default="sage")
    parser.add_argument("--suspect-hidden", type=int, default=256)
    parser.add_argument("--suspect-layers", type=int, default=2)
    parser.add_argument("--suspect-heads", type=int, default=4)
    parser.add_argument("--suspect-dropout", type=float, default=0.5)
    parser.add_argument("--negative-list", type=str, default="")
    parser.add_argument("--source-aggregates", type=str, default="")
    parser.add_argument("--target-tnr", type=float, default=0.99)
    parser.add_argument("--one-class-sigma", type=float, default=2.0)
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _load_negative_specs(path: str) -> List[Tuple[str, str, int, int, int, float]]:
    specs = []
    if not path:
        return specs
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Negative list not found: {file_path}")
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, ckpt, model, hidden, layers, heads, dropout = line.split(",")
        specs.append((ckpt.strip(), model.strip(), int(hidden), int(layers), int(heads), float(dropout)))
    return specs


def _load_source_aggregates(path: str) -> List[float]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source aggregates not found: {file_path}")
    scores = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            scores.append(float(line.strip()))
    return scores


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)
    data, num_classes = load_npz_graph(args.dataset)

    graphs, meta, source_signature = load_fingerprints(Path(args.fingerprints))

    verifier = OwnershipVerifier(
        VerificationConfig(
            mode=args.mode,
            similarity=args.similarity,
            target_tnr=args.target_tnr,
            one_class_sigma=args.one_class_sigma,
        )
    )

    suspect_model = load_model_from_checkpoint(
        Path(args.suspect_ckpt),
        args.suspect_model,
        data.num_features,
        num_classes,
        args.suspect_hidden,
        args.suspect_layers,
        args.suspect_heads,
        args.suspect_dropout,
    ).to(device)

    suspect_signature = verifier.encode_responses(suspect_model, graphs, meta, device)

    negative_specs = _load_negative_specs(args.negative_list)
    negative_scores: List[float] = []
    for ckpt, model, hidden, layers, heads, dropout in negative_specs:
        neg_model = load_model_from_checkpoint(
            Path(ckpt),
            model,
            data.num_features,
            num_classes,
            hidden,
            layers,
            heads,
            dropout,
        ).to(device)
        neg_signature = verifier.encode_responses(neg_model, graphs, meta, device)
        neg_score = verifier.score_signatures(source_signature, neg_signature)
        negative_scores.append(neg_score.aggregate)

    source_scores = _load_source_aggregates(args.source_aggregates)
    if not negative_scores and not source_scores:
        raise ValueError("Provide --negative-list or --source-aggregates for thresholding.")

    result = verifier.verify(
        source_signature,
        suspect_signature,
        negative_aggregates=negative_scores if negative_scores else None,
        source_aggregates=source_scores if source_scores else None,
    )

    print(f"verdict={result.verdict}")
    print(f"aggregate_score={result.aggregate_score:.4f}")
    print(f"threshold={result.threshold:.4f}")
    print(f"confidence={result.confidence:.4f}")


if __name__ == "__main__":
    main()
