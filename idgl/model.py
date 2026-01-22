"""IDGL-style encoder and graph learner modules.

The encoder is a simple 2-layer GCN/GraphSAGE used to produce embeddings H^(t)
for graph learning. These embeddings are then used outside this module to
construct a refined graph A^(t+1) in the iterative loop.
"""

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
    """Two-layer GCN/GraphSAGE encoder for IDGL.

    This module outputs both node embeddings (for graph learning) and logits
    (for the node classification task loss).
    """

    def __init__(self, in_dim: int, config: IDGLConfig) -> None:
        super().__init__()
        # Choose between GCN and GraphSAGE based on config.
        if config.use_sage:
            self.conv1 = SAGEConv(in_dim, config.hidden_dim)
            self.conv2 = SAGEConv(config.hidden_dim, config.hidden_dim)
        else:
            self.conv1 = GCNConv(in_dim, config.hidden_dim)
            self.conv2 = GCNConv(config.hidden_dim, config.hidden_dim)
        # Final linear layer maps embeddings to class logits.
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        # Dropout applied between layers to avoid overfitting.
        self.dropout = config.dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        # First message passing layer.
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        # Second message passing layer to produce final embeddings.
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # Linear classifier for node labels.
        logits = self.classifier(x)
        return x, logits
