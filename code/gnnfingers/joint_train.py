"""Joint optimization for GNNFingers fingerprints and univerifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from gnnfingers.fingerprints import GraphFingerprintSet
from gnnfingers.univerifier import UniverifierConfig, UniverifierMLP
from gnnfingers.utils import flatten_outputs
from gnnfingers.wrappers import LocalPyGSuspectWrapper


@dataclass
class JointTrainConfig:
    """Configuration for joint training."""

    iterations: int = 1000
    lr: float = 0.001
    top_k: int = 20
    feature_lr: float = 0.1
    adj_lr: float = 1.0
    feature_clip_min: float = -1.0
    feature_clip_max: float = 1.0
    task_type: str = "node_classification"


def _model_outputs(wrapper: LocalPyGSuspectWrapper, graphs, meta_list) -> List[torch.Tensor]:
    outputs = []
    raw_outputs = wrapper.predict(graphs)
    for raw, meta in zip(raw_outputs, meta_list):
        outputs.append(flatten_outputs(raw, wrapper.task_type, meta))
    return outputs


def train_gnnfingers(
    target_model,
    pos_models,
    neg_models,
    fingerprint_set: GraphFingerprintSet,
    config: JointTrainConfig,
    device: torch.device,
) -> Dict:
    graphs = fingerprint_set.build_graphs()
    meta_list = fingerprint_set.get_query_meta()

    target_wrapper = LocalPyGSuspectWrapper(target_model, device, task_type=config.task_type)
    sample_output = _model_outputs(target_wrapper, graphs, meta_list)[0]
    input_dim = sample_output.numel()

    univerifier_cfg = UniverifierConfig(input_dim=input_dim, hidden_dims=[128, 64, 32])
    univerifier = UniverifierMLP(univerifier_cfg).to(device)
    optimizer = torch.optim.Adam(univerifier.parameters(), lr=config.lr)
    pos_wrappers = [LocalPyGSuspectWrapper(model, device, task_type=config.task_type) for model in pos_models]
    neg_wrappers = [LocalPyGSuspectWrapper(model, device, task_type=config.task_type) for model in neg_models]

    history = {"loss": [], "accuracy": [], "edge_updates": []}

    for step in range(config.iterations):
        graphs = fingerprint_set.build_graphs()
        meta_list = fingerprint_set.get_query_meta()

        optimizer.zero_grad()
        fingerprint_set.zero_grad()

        outputs = []
        labels = []

        for wrapper in [target_wrapper] + pos_wrappers:
            for out in _model_outputs(wrapper, graphs, meta_list):
                outputs.append(out)
                labels.append(torch.tensor([1.0, 0.0], device=device))

        for wrapper in neg_wrappers:
            for out in _model_outputs(wrapper, graphs, meta_list):
                outputs.append(out)
                labels.append(torch.tensor([0.0, 1.0], device=device))

        if not outputs:
            break
        batch = torch.stack(outputs).to(device)
        targets = torch.stack(labels).to(device)

        preds = univerifier(batch)
        loss = -(targets * torch.log(preds + 1e-8)).mean()
        acc = (preds.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()
        loss.backward()
        fingerprint_set.update_features(
            config.feature_lr,
            config.feature_clip_min,
            config.feature_clip_max,
        )
        fingerprint_set.update_adjacency(config.top_k, step_size=config.adj_lr)
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["accuracy"].append(float(acc))
        if step % 50 == 0:
            print(f"[GNNFingers] step={step} loss={loss.item():.4f} acc={acc:.4f}")

    return {
        "univerifier": univerifier,
        "history": history,
    }
