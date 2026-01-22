"""GIN implementation using PyTorch Geometric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


@dataclass
class TrainConfig:
    num_epochs: int
    lr: float
    weight_decay: float
    dropout: float


class GIN(nn.Module):
    """Multi-layer GIN with shared API for target/surrogate training."""

    def __init__(self, in_feats: int, hidden_dim: int, num_classes: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        mlp = nn.Sequential(nn.Linear(in_feats, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(GINConv(mlp))
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(GINConv(mlp))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embeddings = x
        logits = self.classifier(x)
        return logits, embeddings


def train_gin(model: GIN, data, train_idx: torch.Tensor, config: TrainConfig, device: torch.device) -> None:
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for _ in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()


def evaluate_gin(
    model: GIN,
    data,
    eval_idx: torch.Tensor,
    device: torch.device,
    train_idx: torch.Tensor | None = None,
    log_metrics: bool = False,
):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
    pred = logits[eval_idx]
    labels = data.y[eval_idx]
    acc = (pred.argmax(dim=1) == labels).float().mean()
    if log_metrics:
        eval_loss = F.cross_entropy(pred, labels).item()
        msg = f"Eval acc={acc:.4f} loss={eval_loss:.4f}"
        if train_idx is not None:
            train_pred = logits[train_idx]
            train_labels = data.y[train_idx]
            train_acc = (train_pred.argmax(dim=1) == train_labels).float().mean().item()
            train_loss = F.cross_entropy(train_pred, train_labels).item()
            msg = f"Train acc={train_acc:.4f} loss={train_loss:.4f} | {msg}"
        print(msg)
    return acc, logits, embeddings
