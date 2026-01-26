"""Minimal demo for IDGL-style structure learning."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from src.idgl import IDGLConfig, learn_graph_structure


def _graph_stats(edge_index: torch.Tensor, num_nodes: int) -> dict:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = adj.maximum(adj.transpose())
    n_components, labels = sp.csgraph.connected_components(adj, directed=False)
    degrees = np.asarray(adj.sum(axis=1)).squeeze()
    return {
        "num_edges": int(adj.nnz),
        "num_components": int(n_components),
        "avg_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
    }


def main() -> None:
    torch.manual_seed(7)
    num_nodes = 500
    feat_dim = 128
    features = torch.randn(num_nodes, feat_dim)

    config = IDGLConfig(k=20, iters=3, inner_epochs=20, lambda_sparse=1e-3)
    edge_index, edge_weight = learn_graph_structure(
        features,
        y_q=None,
        init="knn",
        k=config.k,
        iters=config.iters,
        mode="inductive",
        config=config,
    )

    stats = _graph_stats(edge_index, num_nodes)
    print("IDGL demo graph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  edge_weight_mean: {edge_weight.mean().item():.4f}")
    print(
        "Example usage in attack.py:\n"
        "  python attack.py --structure learned --structure_learner idgl "
        "--idgl-k 20 --idgl-iters 5"
    )


if __name__ == "__main__":
    main()
