"""Fingerprint query construction for node-classification GNNs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint construction."""

    num_anchors: int = 20
    num_hops: int = 2
    motif_types: List[str] = field(default_factory=lambda: ["clique", "star", "path"])
    motif_size: int = 4
    trigger_value: float = 5.0
    trigger_feature_idx: int = 0
    anchor_strategy: str = "random"
    strategy: str = "motif"
    num_views: int = 3
    edge_drop_prob: float = 0.1
    feature_noise_std: float = 0.05


def _select_anchors(data: Data, config: FingerprintConfig, rng: np.random.RandomState) -> np.ndarray:
    if config.num_anchors <= 0:
        raise ValueError("num_anchors must be positive.")
    num_nodes = data.num_nodes
    if config.anchor_strategy == "random":
        return rng.choice(num_nodes, size=config.num_anchors, replace=False)
    if config.anchor_strategy == "high_degree":
        row = data.edge_index[0].cpu().numpy()
        degrees = np.bincount(row, minlength=num_nodes)
        return np.argsort(degrees)[-config.num_anchors :]
    raise ValueError("anchor_strategy must be random or high_degree.")


def _trigger_features(num_nodes: int, feat_dim: int, idx: int, value: float, device: torch.device) -> torch.Tensor:
    if idx < 0 or idx >= feat_dim:
        raise ValueError("trigger_feature_idx must be within feature dimension.")
    features = torch.zeros((num_nodes, feat_dim), device=device)
    features[:, idx] = value
    return features


def _build_motif_edges(motif_type: str, start_idx: int, size: int) -> List[Tuple[int, int]]:
    nodes = list(range(start_idx, start_idx + size))
    edges: List[Tuple[int, int]] = []
    if motif_type == "clique":
        for i in range(size):
            for j in range(i + 1, size):
                edges.append((nodes[i], nodes[j]))
    elif motif_type == "star":
        center = nodes[0]
        for node in nodes[1:]:
            edges.append((center, node))
    elif motif_type == "path":
        for i in range(size - 1):
            edges.append((nodes[i], nodes[i + 1]))
    else:
        raise ValueError("motif_type must be clique, star, or path.")
    return edges


def _inject_motif(
    subgraph: Data,
    anchor_index: int,
    motif_type: str,
    config: FingerprintConfig,
) -> Tuple[Data, Dict[str, object]]:
    device = subgraph.x.device
    feat_dim = subgraph.x.size(1)
    motif_start = subgraph.num_nodes
    motif_edges = _build_motif_edges(motif_type, motif_start, config.motif_size)
    anchor_edges = [(anchor_index, node) for node in range(motif_start, motif_start + config.motif_size)]
    all_edges = motif_edges + anchor_edges

    if all_edges:
        edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()
        edge_index = torch.cat([subgraph.edge_index, edge_index], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=motif_start + config.motif_size)
    else:
        edge_index = subgraph.edge_index

    trigger_x = _trigger_features(
        config.motif_size,
        feat_dim,
        config.trigger_feature_idx,
        config.trigger_value,
        device,
    )
    x = torch.cat([subgraph.x, trigger_x], dim=0)
    updated = Data(x=x, edge_index=edge_index)
    meta = {
        "motif": motif_type,
        "motif_nodes": list(range(motif_start, motif_start + config.motif_size)),
        "anchor_index": int(anchor_index),
        "trigger_feature_idx": config.trigger_feature_idx,
        "trigger_value": config.trigger_value,
    }
    return updated, meta


def _extract_anchor_subgraph(data: Data, anchor: int, num_hops: int) -> Tuple[Data, int]:
    subset, edge_index, mapping, _ = k_hop_subgraph(
        anchor,
        num_hops,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    sub_x = data.x[subset]
    sub_data = Data(x=sub_x, edge_index=edge_index)
    return sub_data, int(mapping)


def _perturb_view(data: Data, config: FingerprintConfig, rng: np.random.RandomState) -> Data:
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    keep_mask = rng.rand(num_edges) > config.edge_drop_prob
    edge_index = edge_index[:, keep_mask]
    if edge_index.numel() == 0:
        edge_index = data.edge_index
    noise = torch.randn_like(data.x) * config.feature_noise_std
    x = data.x + noise
    return Data(x=x, edge_index=edge_index)


def build_fingerprint_queries(
    data: Data,
    config: FingerprintConfig,
    seed: int = 7,
) -> Tuple[List[Data], List[Dict[str, object]]]:
    """Build fingerprint query graphs and metadata."""
    rng = np.random.RandomState(seed)
    anchors = _select_anchors(data, config, rng)
    graphs: List[Data] = []
    meta_list: List[Dict[str, object]] = []

    for anchor in anchors:
        subgraph, anchor_idx = _extract_anchor_subgraph(data, int(anchor), config.num_hops)
        if config.strategy == "motif":
            motif_type = rng.choice(config.motif_types)
            updated, meta = _inject_motif(subgraph, anchor_idx, motif_type, config)
            meta.update(
                {
                    "anchor_id": int(anchor),
                    "strategy": "motif",
                    "num_hops": config.num_hops,
                }
            )
            graphs.append(updated)
            meta_list.append(meta)
        elif config.strategy == "perturb":
            for view_id in range(config.num_views):
                view = _perturb_view(subgraph, config, rng)
                meta_list.append(
                    {
                        "anchor_id": int(anchor),
                        "anchor_index": int(anchor_idx),
                        "strategy": "perturb",
                        "num_hops": config.num_hops,
                        "view_id": int(view_id),
                        "edge_drop_prob": config.edge_drop_prob,
                        "feature_noise_std": config.feature_noise_std,
                    }
                )
                graphs.append(view)
        else:
            raise ValueError("strategy must be motif or perturb.")

    return graphs, meta_list
