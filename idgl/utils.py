"""Re-export IDGL helpers from core utils for shared usage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from src.utils import (
    GraphStats,
    adjacency_stats,
    build_knn_graph,
    graph_smoothness_loss,
    normalize_features,
    relative_change,
    similarity_matrix,
    sparse_degree_loss,
    sparse_from_similarity,
)

__all__ = [
    "GraphStats",
    "adjacency_stats",
    "build_knn_graph",
    "graph_smoothness_loss",
    "normalize_features",
    "relative_change",
    "similarity_matrix",
    "sparse_degree_loss",
    "sparse_from_similarity",
]
