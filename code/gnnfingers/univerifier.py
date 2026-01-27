"""Univerifier MLP for GNNFingers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class UniverifierConfig:
    """Configuration for Univerifier MLP."""

    input_dim: int
    hidden_dims: List[int]
    dropout: float = 0.1


class UniverifierMLP(nn.Module):
    """Three-layer MLP for ownership verification."""

    def __init__(self, config: UniverifierConfig) -> None:
        super().__init__()
        dims = [config.input_dim] + config.hidden_dims
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)
