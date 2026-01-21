"""IDGL-style encoder and graph learner modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


@dataclass
class IDGLConfig:
    """Configuration for the encoder and graph learner."""

    hidden_dim: int
    num_classes: int
    dropout: float
    use_sage: bool = False


class GNNEncoder(nn.Module):
    """Two-layer GCN/GraphSAGE encoder for IDGL."""

    def __init__(self, in_dim: int, config: IDGLConfig) -> None:
        super().__init__()
        if config.use_sage:
            self.conv1 = SAGEConv(in_dim, config.hidden_dim)
            self.conv2 = SAGEConv(config.hidden_dim, config.hidden_dim)
        else:
            self.conv1 = GCNConv(in_dim, config.hidden_dim)
            self.conv2 = GCNConv(config.hidden_dim, config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.dropout = config.dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        logits = self.classifier(x)
        return x, logits
