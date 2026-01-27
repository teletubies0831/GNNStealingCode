"""Input/output helpers for GNNFingers artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data


def save_fingerprints(
    output_dir: Path,
    graphs: List[Data],
    meta: List[Dict[str, object]],
    signature: torch.Tensor,
) -> None:
    """Save fingerprint graphs, metadata, and source signature."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_dir / "fingerprint_graphs.pt")
    torch.save(signature, output_dir / "fingerprint_signature.pt")
    with (output_dir / "fingerprint_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_fingerprints(
    fingerprint_dir: Path,
) -> Tuple[List[Data], List[Dict[str, object]], torch.Tensor]:
    """Load fingerprint graphs, metadata, and signature from disk."""
    graphs_path = fingerprint_dir / "fingerprint_graphs.pt"
    meta_path = fingerprint_dir / "fingerprint_meta.json"
    signature_path = fingerprint_dir / "fingerprint_signature.pt"
    if not graphs_path.exists():
        raise FileNotFoundError(f"Missing fingerprint graphs: {graphs_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing fingerprint metadata: {meta_path}")
    if not signature_path.exists():
        raise FileNotFoundError(f"Missing fingerprint signature: {signature_path}")

    graphs = torch.load(graphs_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    signature = torch.load(signature_path)
    return graphs, meta, signature
