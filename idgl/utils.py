"""Utility helpers for iterative graph learning (IDGL-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GraphStats:
    """Summary statistics for a learned adjacency."""

    avg_degree: float
    nnz: int


def normalize_features(x: Tensor) -> Tensor:
    """Row-normalize features to unit norm for cosine similarity."""
    return F.normalize(x, p=2.0, dim=1)


def build_knn_graph(
    x: Tensor,
    k: int,
    self_loop: bool = True,
    symmetric: bool = True,
) -> Tensor:
    """Build a kNN adjacency (sparse edge index) using cosine similarity."""
    x_norm = normalize_features(x)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1.0)
    _, idx = torch.topk(sim, k=k, dim=1)
    row = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
    col = idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    if self_loop:
        self_loop_idx = torch.arange(x.size(0), device=x.device)
        self_edges = torch.stack([self_loop_idx, self_loop_idx], dim=0)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
    if symmetric:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def sparse_from_similarity(
    sim: Tensor,
    top_k: int,
    self_loop: bool = True,
    symmetric: bool = True,
) -> Tensor:
    """Sparsify a similarity matrix into an edge index via top-k."""
    sim = sim.clone()
    sim.fill_diagonal_(-1.0)
    _, idx = torch.topk(sim, k=top_k, dim=1)
    row = torch.arange(sim.size(0), device=sim.device).repeat_interleave(top_k)
    col = idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    if self_loop:
        diag = torch.arange(sim.size(0), device=sim.device)
        edge_index = torch.cat([edge_index, torch.stack([diag, diag], dim=0)], dim=1)
    if symmetric:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def adjacency_stats(edge_index: Tensor, num_nodes: int) -> GraphStats:
    """Compute average degree and nnz for logging."""
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
    return GraphStats(avg_degree=deg.mean().item(), nnz=edge_index.size(1))


def graph_smoothness_loss(h: Tensor, edge_index: Tensor) -> Tensor:
    """Smoothness penalty: sum_{i,j} A_ij ||h_i - h_j||^2."""
    src, dst = edge_index
    diff = h[src] - h[dst]
    return (diff.pow(2).sum(dim=1)).mean()


def sparse_degree_loss(edge_index: Tensor, num_nodes: int, target_degree: float) -> Tensor:
    """Penalty to avoid degenerate graphs by matching degree statistics."""
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
    return (deg.mean() - target_degree).abs()


def relative_change(a_prev: Tensor, a_next: Tensor) -> float:
    """Relative Frobenius change between two dense matrices."""
    delta = (a_next - a_prev).pow(2).sum().sqrt()
    denom = a_prev.pow(2).sum().sqrt().clamp_min(1e-8)
    return (delta / denom).item()


def similarity_matrix(h: Tensor) -> Tensor:
    """Cosine similarity matrix from node embeddings."""
    h_norm = normalize_features(h)
    return h_norm @ h_norm.t()
