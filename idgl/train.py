"""Run IDGL-style iterative graph learning on citation datasets."""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from idgl.model import GNNEncoder, IDGLConfig
from idgl.utils import (
    GraphStats,
    adjacency_stats,
    build_knn_graph,
    graph_smoothness_loss,
    relative_change,
    similarity_matrix,
    sparse_degree_loss,
    sparse_from_similarity,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative Deep Graph Learning (PyG)")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--inner-epochs", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--lambda-smooth", type=float, default=1.0)
    parser.add_argument("--mu-degree", type=float, default=1.0)
    parser.add_argument("--use-sage", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _load_dataset(name: str) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dataset = Planetoid(root="data", name=name.capitalize(), transform=NormalizeFeatures())
    data = dataset[0]
    return data.x, data.y, data.edge_index, data.train_mask, data.val_mask, data.test_mask


def _accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == labels[mask]).float().mean()
    return correct.item()


def _train_epoch(
    model: GNNEncoder,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    edge_index: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    lambda_smooth: float,
    mu_degree: float,
    target_degree: float,
) -> Tuple[Tensor, Tensor]:
    model.train()
    optimizer.zero_grad()
    embeddings, logits = model(x, edge_index)
    task_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    smooth_loss = graph_smoothness_loss(embeddings, edge_index)
    degree_loss = sparse_degree_loss(edge_index, x.size(0), target_degree)
    loss = task_loss + lambda_smooth * smooth_loss + mu_degree * degree_loss
    loss.backward()
    optimizer.step()
    return embeddings.detach(), logits.detach()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    x, y, edge_index, train_mask, val_mask, test_mask = _load_dataset(args.dataset)
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if edge_index.numel() == 0:
        edge_index = build_knn_graph(x, k=args.knn).to(device)

    num_classes = int(y.max().item() + 1)
    config = IDGLConfig(hidden_dim=args.hidden_dim, num_classes=num_classes, dropout=args.dropout, use_sage=args.use_sage)
    model = GNNEncoder(x.size(1), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    prev_dense = torch.zeros((x.size(0), x.size(0)), device=device)
    for iteration in range(1, args.max_iter + 1):
        for _ in range(args.inner_epochs):
            embeddings, logits = _train_epoch(
                model,
                optimizer,
                x,
                edge_index,
                y,
                train_mask,
                args.lambda_smooth,
                args.mu_degree,
                target_degree=float(args.knn),
            )

        sim = similarity_matrix(embeddings)
        new_edge_index = sparse_from_similarity(sim, top_k=args.knn).to(device)
        stats = adjacency_stats(new_edge_index, x.size(0))
        dense_next = torch.zeros_like(prev_dense)
        dense_next[new_edge_index[0], new_edge_index[1]] = 1.0

        change = relative_change(prev_dense, dense_next) if iteration > 1 else float("inf")
        prev_dense = dense_next
        edge_index = new_edge_index

        train_acc = _accuracy(logits, y, train_mask)
        val_acc = _accuracy(logits, y, val_mask)
        test_acc = _accuracy(logits, y, test_mask)

        stop_flag = change < args.tol
        print(
            f"[Iter {iteration:02d}] train={train_acc:.4f} val={val_acc:.4f} "
            f"test={test_acc:.4f} avg_deg={stats.avg_degree:.2f} change={change:.6f} stop={stop_flag}"
        )

        if stop_flag:
            break


if __name__ == "__main__":
    main()
