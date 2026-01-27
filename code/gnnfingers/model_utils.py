"""Model loading utilities for GNNFingers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import torch

from src.gat import GAT
from src.gin import GIN
from src.sage import SAGE


def build_model(
    model_name: str,
    in_feats: int,
    hidden_dim: int,
    num_classes: int,
    num_layers: int,
    heads: int,
    dropout: float,
):
    if model_name == "gat":
        return GAT(in_feats, hidden_dim, num_classes, num_layers, heads, dropout)
    if model_name == "gin":
        return GIN(in_feats, hidden_dim, num_classes, num_layers, dropout)
    if model_name == "sage":
        return SAGE(in_feats, hidden_dim, num_classes, num_layers, dropout)
    raise ValueError("model_name must be gat, gin, or sage.")


def load_model_from_dir(
    model_dir: Path,
    model_name: str,
    dataset: str,
    in_feats: int,
    num_classes: int,
    fallback_hidden: int = 256,
    fallback_layers: int = 2,
    fallback_heads: int = 4,
    fallback_dropout: float = 0.5,
) -> Tuple[torch.nn.Module, dict]:
    """Load a model checkpoint, using saved model_args when available."""
    model_args_path = model_dir / "model_args"
    if model_args_path.exists():
        args = pickle.load(model_args_path.open("rb"))
        hidden = getattr(args, "num_hidden", fallback_hidden)
        layers = getattr(args, "num_layers", fallback_layers)
        heads = getattr(args, "head", fallback_heads)
        dropout = getattr(args, "dropout", fallback_dropout)
    else:
        hidden = fallback_hidden
        layers = fallback_layers
        heads = fallback_heads
        dropout = fallback_dropout

    model = build_model(model_name, in_feats, hidden, num_classes, layers, heads, dropout)
    ckpt_path = model_dir / f"target_model_{model_name}_{dataset}"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model, {
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "dropout": dropout,
    }


def load_model_from_checkpoint(
    ckpt_path: Path,
    model_name: str,
    in_feats: int,
    num_classes: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
) -> torch.nn.Module:
    model = build_model(model_name, in_feats, hidden_dim, num_classes, num_layers, heads, dropout)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model
