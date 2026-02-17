"""Utility functions for GNNFingers training and evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from torch_geometric.data import Data


def flatten_outputs(
    model_outputs: Dict[str, torch.Tensor],
    task_type: str,
    meta: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten outputs into a 1D vector for univerifier input."""
    logits = model_outputs["logits"]
    embeddings = model_outputs["embeddings"]

    if task_type == "node_classification":
        node_indices = meta["node_indices"]
        return logits[node_indices].reshape(-1)
    if task_type == "link_prediction":
        pair_indices = meta["pair_indices"]
        src = pair_indices[:, 0]
        dst = pair_indices[:, 1]
        scores = (embeddings[src] * embeddings[dst]).sum(dim=1)
        return scores.reshape(-1)
    if task_type == "graph_classification":
        graph_repr = logits.mean(dim=0)
        return graph_repr.reshape(-1)
    if task_type == "graph_matching":
        graph_repr = embeddings.mean(dim=0)
        return graph_repr.reshape(-1)
    raise ValueError("Unsupported task_type.")


def compute_thresholds(scores: List[float], num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    values = np.array(scores)
    lambdas = np.linspace(values.min(), values.max(), num_bins)
    return lambdas, values


def evaluate_metrics(
    scores_pos: List[float],
    scores_neg: List[float],
    lambdas: np.ndarray,
) -> Dict[str, float]:
    robustness = []
    uniqueness = []
    for lam in lambdas:
        tp = np.mean(np.array(scores_pos) >= lam)
        tn = np.mean(np.array(scores_neg) < lam)
        robustness.append(tp)
        uniqueness.append(tn)
    robustness = np.array(robustness)
    uniqueness = np.array(uniqueness)
    return {
        "robustness": float(robustness.max()),
        "uniqueness": float(uniqueness.max()),
    }


def select_best_threshold(scores_pos: List[float], scores_neg: List[float]) -> float:
    """Select a lambda threshold that maximizes balanced accuracy."""
    pos = np.array(scores_pos)
    neg = np.array(scores_neg)
    if pos.size == 0 or neg.size == 0:
        return 0.5

    candidates = np.unique(np.concatenate([pos, neg]))
    best_lambda = float(candidates[0])
    best_score = -1.0
    for lam in candidates:
        tpr = np.mean(pos >= lam)
        tnr = np.mean(neg < lam)
        balanced_acc = 0.5 * (tpr + tnr)
        if balanced_acc > best_score:
            best_score = balanced_acc
            best_lambda = float(lam)
    return best_lambda


def save_config(path: Path, config: Dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_edge_index_from_logits(adj_logits: torch.Tensor) -> torch.Tensor:
    adj = (torch.sigmoid(adj_logits) > 0.5).float()
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index
