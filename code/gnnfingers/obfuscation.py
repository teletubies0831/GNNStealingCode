"""Generate positive and negative model ensembles for GNNFingers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.gat import GAT
from src.gin import GIN
from src.sage import SAGE


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble generation."""

    num_positive: int = 100
    num_negative: int = 100
    num_epochs: int = 10
    distill_epochs: int = 10
    architectures: Tuple[str, ...] = ("gat", "gin", "sage")


def _build_model(model_name: str, in_feats: int, hidden_dim: int, num_classes: int, num_layers: int, heads: int, dropout: float):
    if model_name == "gat":
        return GAT(in_feats, hidden_dim, num_classes, num_layers, heads, dropout)
    if model_name == "gin":
        return GIN(in_feats, hidden_dim, num_classes, num_layers, dropout)
    if model_name == "sage":
        return SAGE(in_feats, hidden_dim, num_classes, num_layers, dropout)
    raise ValueError("model_name must be gat, gin, or sage")


def _train_model(model, data, train_idx, num_epochs: int) -> None:
    model.to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()


def _fine_tune_last_layer(model, data, train_idx, num_epochs: int) -> None:
    last_layer_idx = len(model.layers) - 1
    for name, param in model.named_parameters():
        param.requires_grad = f"layers.{last_layer_idx}" in name
    _train_model(model, data, train_idx, num_epochs)


def _reset_parameters(model) -> None:
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _distill_student(teacher, student, data, train_idx, num_epochs: int) -> None:
    student.to(data.x.device)
    teacher.to(data.x.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    teacher.eval()
    with torch.no_grad():
        teacher_logits, _ = teacher(data.x, data.edge_index)
        soft_targets = teacher_logits.softmax(dim=1)
    for _ in range(num_epochs):
        student.train()
        optimizer.zero_grad()
        logits, _ = student(data.x, data.edge_index)
        loss = F.kl_div(F.log_softmax(logits[train_idx], dim=1), soft_targets[train_idx], reduction="batchmean")
        loss.backward()
        optimizer.step()


def prepare_model_ensemble(target_model, data, train_idx, cfg: EnsembleConfig, base_model_cfg: dict):
    """Return positive and negative model lists for GNNFingers."""
    pos_models = []
    neg_models = []

    for _ in range(cfg.num_positive // 4):
        model = copy.deepcopy(target_model)
        _fine_tune_last_layer(model, data, train_idx, cfg.num_epochs)
        pos_models.append(model)

        model = copy.deepcopy(target_model)
        _train_model(model, data, train_idx, cfg.num_epochs)
        pos_models.append(model)

        model = copy.deepcopy(target_model)
        _reset_parameters(model)
        _fine_tune_last_layer(model, data, train_idx, cfg.num_epochs)
        pos_models.append(model)

        model = copy.deepcopy(target_model)
        _reset_parameters(model)
        _train_model(model, data, train_idx, cfg.num_epochs)
        pos_models.append(model)

    for idx in range(cfg.num_negative):
        arch = cfg.architectures[idx % len(cfg.architectures)]
        model = _build_model(
            arch,
            data.num_features,
            base_model_cfg["hidden_dim"],
            base_model_cfg["num_classes"],
            base_model_cfg["num_layers"],
            base_model_cfg["heads"],
            base_model_cfg["dropout"],
        )
        _train_model(model, data, train_idx, cfg.num_epochs)
        neg_models.append(model)

    if cfg.num_positive > len(pos_models):
        pos_models = pos_models[: cfg.num_positive]

    if cfg.num_positive > 0:
        student_arch = cfg.architectures[-1]
        student = _build_model(
            student_arch,
            data.num_features,
            base_model_cfg["hidden_dim"],
            base_model_cfg["num_classes"],
            base_model_cfg["num_layers"],
            base_model_cfg["heads"],
            base_model_cfg["dropout"],
        )
        _distill_student(target_model, student, data, train_idx, cfg.distill_epochs)
        pos_models.append(student)

    return pos_models[: cfg.num_positive], neg_models[: cfg.num_negative]
