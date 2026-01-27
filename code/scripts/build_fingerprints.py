"""Build fingerprint queries and source signatures for GNNFingers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from gnnfingers.fingerprint_builder import FingerprintConfig, build_fingerprint_queries
from gnnfingers.io import save_fingerprints
from gnnfingers.model_utils import load_model_from_dir
from gnnfingers.verifier import OwnershipVerifier, VerificationConfig
from src.utils import load_npz_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build GNNFingers fingerprints")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--source-model", type=str, default="gat")
    parser.add_argument("--source-model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./fingerprints")
    parser.add_argument("--strategy", type=str, default="motif")
    parser.add_argument("--num-anchors", type=int, default=20)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--motif-size", type=int, default=4)
    parser.add_argument("--trigger-idx", type=int, default=0)
    parser.add_argument("--trigger-value", type=float, default=5.0)
    parser.add_argument("--mode", type=str, default="embedding")
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)
    data, num_classes = load_npz_graph(args.dataset)

    config = FingerprintConfig(
        num_anchors=args.num_anchors,
        num_hops=args.num_hops,
        motif_size=args.motif_size,
        trigger_feature_idx=args.trigger_idx,
        trigger_value=args.trigger_value,
        strategy=args.strategy,
    )
    graphs, meta = build_fingerprint_queries(data, config)

    source_model, _ = load_model_from_dir(
        Path(args.source_model_dir),
        args.source_model,
        args.dataset,
        data.num_features,
        num_classes,
    )
    source_model = source_model.to(device)

    verifier = OwnershipVerifier(VerificationConfig(mode=args.mode))
    signature = verifier.encode_responses(source_model, graphs, meta, device)

    save_fingerprints(Path(args.output_dir), graphs, meta, signature)
    print(f"Saved {len(graphs)} fingerprint queries to {args.output_dir}")


if __name__ == "__main__":
    main()
