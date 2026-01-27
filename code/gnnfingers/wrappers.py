"""Wrappers for suspect model inference (local or HTTP API)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List
from urllib.request import Request, urlopen

import torch
from torch_geometric.data import Data


class SuspectModelWrapper(ABC):
    """Abstract wrapper for suspect models."""

    @abstractmethod
    def predict(self, graphs: List[Data]) -> List[torch.Tensor]:
        """Return model outputs for each graph."""


class LocalPyGSuspectWrapper(SuspectModelWrapper):
    """Wrapper around a local PyG model."""

    def __init__(self, model, device: torch.device, task_type: str = "node_classification") -> None:
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type

    def predict(self, graphs: List[Data]) -> List[torch.Tensor]:
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(self.device)
                logits, embeddings = self.model(graph.x, graph.edge_index)
                outputs.append({"logits": logits, "embeddings": embeddings})
        return outputs


class HttpSuspectWrapper(SuspectModelWrapper):
    """HTTP API wrapper for black-box suspect models."""

    def __init__(self, endpoint: str, timeout: int = 30) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def predict(self, graphs: List[Data]) -> List[torch.Tensor]:
        outputs = []
        for graph in graphs:
            payload = {
                "x": graph.x.cpu().tolist(),
                "edge_index": graph.edge_index.cpu().tolist(),
            }
            req = Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            outputs.append(
                {
                    "logits": torch.tensor(data["logits"]),
                    "embeddings": torch.tensor(data.get("embeddings", data["logits"])),
                }
            )
        return outputs
