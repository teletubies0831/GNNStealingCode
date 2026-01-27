"""IDGL-style graph structure learning utilities (core module)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import coalesce


@dataclass
class IDGLConfig:
    """Hyperparameters for IDGL structure learning."""

    k: int = 20
    iters: int = 5
    inner_epochs: int = 50
    lr: float = 0.01
    weight_decay: float = 5e-4
    hidden_dim: int = 64
    lambda_task: float = 1.0
    lambda_smooth: float = 1.0
    lambda_sparse: float = 1e-3
    tau: float = 0.5
    epsilon: float = 0.0
    tol: float = 1e-3
    self_loop: bool = True
    task_mode: str = "auto"


class GraphEncoder(nn.Module):
    """Lightweight GraphSAGE encoder for IDGL."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class IDGLModel(nn.Module):
    """IDGL encoder with optional classifier/decoder heads."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, num_classes: Optional[int]) -> None:
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes) if num_classes is not None else None
        self.decoder = nn.Linear(emb_dim, in_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)


def _build_knn_graph(
    features: torch.Tensor,
    k: int,
    tau: float,
    epsilon: float,
    self_loop: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a kNN graph from features using cosine similarity."""
    feats = features.detach().cpu().numpy()
    k_eff = min(k + 1, feats.shape[0])
    nn_model = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn_model.fit(feats)
    distances, indices = nn_model.kneighbors(feats)

    rows = []
    cols = []
    weights = []
    for i in range(indices.shape[0]):
        for j, dist in zip(indices[i], distances[i]):
            if i == j and not self_loop:
                continue
            if i == j and self_loop:
                sim = 1.0
            else:
                sim = max(0.0, 1.0 - dist)
            weight = float(np.exp(sim / tau))
            if weight <= epsilon:
                continue
            rows.append(i)
            cols.append(j)
            weights.append(weight)

    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=features.device)
    edge_weight = torch.tensor(weights, dtype=torch.float, device=features.device)
    edge_index, edge_weight = _sym_norm(edge_index, edge_weight, features.size(0))
    return edge_index, edge_weight


def _sym_norm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize and normalize edge weights with D^{-1/2} A D^{-1/2}."""
    if edge_index.numel() == 0:
        return edge_index, edge_weight
    rev_edge_index = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=num_nodes)

    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_weight.device)
    deg.index_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    norm_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm_weight


def _smoothness_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute Laplacian smoothness loss: sum A_ij ||z_i - z_j||^2."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    row, col = edge_index
    diff = embeddings[row] - embeddings[col]
    per_edge = edge_weight * (diff.pow(2).sum(dim=1))
    return per_edge.mean()


def _sparse_loss(edge_weight: torch.Tensor) -> torch.Tensor:
    """Encourage sparse adjacency weights via L1 penalty."""
    if edge_weight.numel() == 0:
        return torch.tensor(0.0, device=edge_weight.device)
    return edge_weight.abs().mean()


def _relative_change(
    prev_edge_index: torch.Tensor,
    prev_edge_weight: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> float:
    """Compute relative Frobenius change between two sparse adjacency matrices."""
    if prev_edge_index.numel() == 0:
        return float("inf")
    prev = torch.sparse_coo_tensor(prev_edge_index, prev_edge_weight, (num_nodes, num_nodes)).coalesce()
    curr = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()
    diff = (curr - prev).coalesce()
    diff_norm = torch.sqrt((diff.values().pow(2)).sum()).item()
    prev_norm = torch.sqrt((prev.values().pow(2)).sum()).item()
    if prev_norm == 0.0:
        return float("inf")
    return diff_norm / prev_norm


def learn_graph_structure(
    X_q: torch.Tensor,
    y_q: Optional[torch.Tensor] = None,
    init: str = "knn",
    k: int = 20,
    iters: int = 5,
    mode: str = "inductive",
    config: Optional[IDGLConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Learn graph structure for query features.

    Args:
        X_q: Node features with shape [num_nodes, num_features].
        y_q: Optional labels with shape [num_nodes].
        init: Graph initialization method ("knn").
        k: Number of neighbors for kNN.
        iters: Number of alternating optimization steps.
        mode: Placeholder for inductive/transductive mode.
        config: Optional IDGLConfig to override hyperparameters.

    Returns:
        edge_index: COO indices with shape [2, num_edges].
        edge_weight: Edge weights with shape [num_edges].
    """
    cfg = config or IDGLConfig()
    cfg.k = k if k is not None else cfg.k
    cfg.iters = iters if iters is not None else cfg.iters

    if init != "knn":
        raise ValueError("Only knn initialization is supported for IDGL.")

    device = X_q.device
    num_nodes, feat_dim = X_q.size()
    num_classes = int(y_q.max().item() + 1) if y_q is not None else None
    task_mode = cfg.task_mode

    edge_index, edge_weight = _build_knn_graph(X_q, cfg.k, cfg.tau, cfg.epsilon, cfg.self_loop)
    if cfg.iters <= 0:
        return edge_index, edge_weight

    model = IDGLModel(feat_dim, cfg.hidden_dim, cfg.hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.iters):
        for _ in range(cfg.inner_epochs):
            model.train()
            optimizer.zero_grad()
            embeddings = model(X_q, edge_index)

            task_loss = torch.tensor(0.0, device=device)
            if y_q is not None and task_mode in ("auto", "supervised"):
                logits = model.classifier(embeddings)
                task_loss = F.cross_entropy(logits, y_q)
            elif task_mode in ("auto", "selfsupervised"):
                recon = model.decoder(embeddings)
                task_loss = F.mse_loss(recon, X_q)

            smooth_loss = _smoothness_loss(embeddings, edge_index, edge_weight)
            sparse_loss = _sparse_loss(edge_weight)
            total_loss = (
                cfg.lambda_task * task_loss
                + cfg.lambda_smooth * smooth_loss
                + cfg.lambda_sparse * sparse_loss
            )
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(X_q, edge_index)

        prev_edge_index = edge_index
        prev_edge_weight = edge_weight
        edge_index, edge_weight = _build_knn_graph(
            embeddings,
            cfg.k,
            cfg.tau,
            cfg.epsilon,
            cfg.self_loop,
        )

        if cfg.tol > 0:
            rel_change = _relative_change(
                prev_edge_index,
                prev_edge_weight,
                edge_index,
                edge_weight,
                num_nodes,
            )
            if rel_change < cfg.tol:
                break

    return edge_index, edge_weight
