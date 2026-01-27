"""Graph fingerprint data structures and sampling logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.data import Data


@dataclass
class FingerprintMeta:
    """Metadata for a single fingerprint graph."""

    node_indices: torch.Tensor
    pair_indices: torch.Tensor


class GraphFingerprintSet(nn.Module):
    """Maintain a set of graph fingerprints with learnable structure and features."""

    def __init__(
        self,
        num_fingerprints: int,
        num_nodes: int,
        feat_dim: int,
        epsilon: float,
        num_queries: int,
        task_type: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_fingerprints = num_fingerprints
        self.num_nodes = num_nodes
        self.feat_dim = max(feat_dim, 1)
        self.epsilon = epsilon
        self.num_queries = num_queries
        self.task_type = task_type
        self.device = device

        self._init_features()
        self._init_adjacency()
        self.meta = self._init_meta()

    def _init_features(self) -> None:
        features = torch.randn(self.num_fingerprints, self.num_nodes, self.feat_dim, device=self.device)
        self.features = nn.Parameter(features)

    def _init_adjacency(self) -> None:
        init_prob = torch.full((self.num_fingerprints, self.num_nodes, self.num_nodes), self.epsilon, device=self.device)
        logits = torch.log(init_prob / (1 - init_prob + 1e-6))
        self.adj_logits = nn.Parameter(logits)

    def _init_meta(self) -> List[FingerprintMeta]:
        meta: List[FingerprintMeta] = []
        for _ in range(self.num_fingerprints):
            node_indices = torch.randperm(self.num_nodes, device=self.device)[: self.num_queries]
            pair_indices = torch.randint(0, self.num_nodes, (self.num_queries, 2), device=self.device)
            meta.append(FingerprintMeta(node_indices=node_indices, pair_indices=pair_indices))
        return meta

    def build_graphs(self) -> List[Data]:
        graphs: List[Data] = []
        for idx in range(self.num_fingerprints):
            adj_logits = self.adj_logits[idx]
            adj_soft = torch.sigmoid(adj_logits)
            adj_bin = (adj_soft > 0.5).float()
            adj_ste = adj_soft + (adj_bin - adj_soft).detach()

            features = self.features[idx]
            features = adj_ste @ features

            edge_index = adj_bin.nonzero(as_tuple=False).t().contiguous()
            graphs.append(Data(x=features, edge_index=edge_index))
        return graphs

    def clip_features(self, min_val: float, max_val: float) -> None:
        with torch.no_grad():
            self.features.data.clamp_(min_val, max_val)

    def update_adjacency(self, top_k: int) -> None:
        if self.adj_logits.grad is None:
            return
        grad = self.adj_logits.grad.detach()
        for idx in range(self.num_fingerprints):
            g = grad[idx].abs().view(-1)
            if g.numel() == 0:
                continue
            topk = torch.topk(g, k=min(top_k, g.numel()))
            flat_indices = topk.indices
            for flat_idx in flat_indices:
                row = flat_idx // self.num_nodes
                col = flat_idx % self.num_nodes
                sign = grad[idx, row, col]
                adj_bin = (torch.sigmoid(self.adj_logits[idx, row, col]) > 0.5).float()
                if adj_bin == 1.0 and sign <= 0:
                    self.adj_logits.data[idx, row, col] -= 5.0
                elif adj_bin == 0.0 and sign >= 0:
                    self.adj_logits.data[idx, row, col] += 5.0

    def get_query_meta(self) -> List[Dict[str, torch.Tensor]]:
        return [
            {"node_indices": meta.node_indices, "pair_indices": meta.pair_indices}
            for meta in self.meta
        ]

    def save(self, path: str) -> None:
        payload = {
            "num_fingerprints": self.num_fingerprints,
            "num_nodes": self.num_nodes,
            "feat_dim": self.feat_dim,
            "epsilon": self.epsilon,
            "num_queries": self.num_queries,
            "task_type": self.task_type,
            "features": self.features.detach().cpu(),
            "adj_logits": self.adj_logits.detach().cpu(),
            "meta": self.get_query_meta(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "GraphFingerprintSet":
        payload = torch.load(path, map_location=device)
        fingerprint_set = GraphFingerprintSet(
            num_fingerprints=payload["num_fingerprints"],
            num_nodes=payload["num_nodes"],
            feat_dim=payload["feat_dim"],
            epsilon=payload["epsilon"],
            num_queries=payload["num_queries"],
            task_type=payload["task_type"],
            device=device,
        )
        fingerprint_set.features.data = payload["features"].to(device)
        fingerprint_set.adj_logits.data = payload["adj_logits"].to(device)
        fingerprint_set.meta = [
            FingerprintMeta(
                node_indices=meta["node_indices"].to(device),
                pair_indices=meta["pair_indices"].to(device),
            )
            for meta in payload["meta"]
        ]
        return fingerprint_set
