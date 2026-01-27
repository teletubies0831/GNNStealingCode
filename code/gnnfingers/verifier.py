"""Ownership verification logic for GNNFingers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.data import Data

from gnnfingers.response_encoder import encode_response


@dataclass
class VerificationConfig:
    """Configuration for ownership verification."""

    mode: Literal["prediction", "embedding", "projection"] = "embedding"
    similarity: Literal["cosine", "l2"] = "cosine"
    temperature: float = 1.0
    target_tnr: float = 0.99
    one_class_sigma: float = 2.0


@dataclass
class OwnershipScore:
    """Similarity scores for a suspect model."""

    scores: torch.Tensor
    aggregate: float


@dataclass
class VerificationResult:
    """Final ownership verification output."""

    verdict: str
    confidence: float
    threshold: float
    aggregate_score: float


def _project_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings.detach().cpu().numpy())
    return torch.from_numpy(coords).float()


def _extract_response(
    model,
    data: Data,
    anchor_index: int,
    mode: Literal["prediction", "embedding", "projection"],
) -> torch.Tensor:
    logits, embeddings = model(data.x, data.edge_index)
    if mode == "prediction":
        return logits[anchor_index]
    if mode == "embedding":
        return embeddings[anchor_index]
    coords = _project_embeddings(embeddings)
    return coords[anchor_index]


def _compute_similarity(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cosine":
        return F.cosine_similarity(a, b, dim=-1)
    if mode == "l2":
        dist = torch.norm(a - b, p=2, dim=-1)
        return torch.exp(-dist)
    raise ValueError("similarity must be cosine or l2.")


def _align_signatures(
    source: torch.Tensor,
    suspect: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align signature dimensions by truncating to the minimum embedding size."""
    if source.shape[0] != suspect.shape[0]:
        raise ValueError("Signature query counts must match.")
    if source.shape[1] == suspect.shape[1]:
        return source, suspect
    target_dim = min(source.shape[1], suspect.shape[1])
    return source[:, :target_dim], suspect[:, :target_dim]


class OwnershipVerifier:
    """Ownership verification with adaptive or one-class thresholds."""

    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    def encode_responses(
        self,
        model,
        graphs: Iterable[Data],
        meta: List[dict],
        device: torch.device,
    ) -> torch.Tensor:
        responses: List[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for graph, info in zip(graphs, meta):
                graph = graph.to(device)
                anchor_index = int(info["anchor_index"])
                raw = _extract_response(model, graph, anchor_index, self.config.mode)
                encoded = encode_response(raw, self.config.mode, temperature=self.config.temperature)
                responses.append(encoded.cpu())
        return torch.stack(responses, dim=0)

    def score_signatures(self, source: torch.Tensor, suspect: torch.Tensor) -> OwnershipScore:
        source_aligned, suspect_aligned = _align_signatures(source, suspect)
        scores = _compute_similarity(source_aligned, suspect_aligned, self.config.similarity)
        aggregate = scores.mean().item()
        return OwnershipScore(scores=scores, aggregate=aggregate)

    def threshold_from_negatives(self, negative_scores: Iterable[float]) -> float:
        scores = np.array(list(negative_scores))
        if scores.size == 0:
            raise ValueError("negative_scores must be non-empty.")
        return float(np.quantile(scores, self.config.target_tnr))

    def threshold_one_class(self, source_scores: Iterable[float]) -> float:
        scores = np.array(list(source_scores))
        if scores.size == 0:
            raise ValueError("source_scores must be non-empty.")
        mean = float(scores.mean())
        std = float(scores.std())
        return mean - self.config.one_class_sigma * std

    def verify(
        self,
        source_signature: torch.Tensor,
        suspect_signature: torch.Tensor,
        negative_aggregates: Optional[List[float]] = None,
        source_aggregates: Optional[List[float]] = None,
    ) -> VerificationResult:
        score = self.score_signatures(source_signature, suspect_signature)
        if negative_aggregates:
            threshold = self.threshold_from_negatives(negative_aggregates)
        elif source_aggregates:
            threshold = self.threshold_one_class(source_aggregates)
        else:
            raise ValueError("Provide negative_aggregates or source_aggregates for thresholding.")

        verdict = "stolen" if score.aggregate >= threshold else "not_stolen"
        confidence = float(score.aggregate - threshold)
        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            threshold=float(threshold),
            aggregate_score=score.aggregate,
        )
