"""Response encoding utilities for GNNFingers."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def encode_response(
    response: torch.Tensor,
    mode: Literal["prediction", "embedding", "projection"],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Encode model responses to a unified vector space.

    Args:
        response: Tensor of shape [d] representing logits, embeddings, or projections.
        mode: Response mode.
        temperature: Softmax temperature for prediction mode.

    Returns:
        Encoded response vector of shape [d].
    """
    if response.dim() != 1:
        raise ValueError("response must be a 1D tensor.")
    if mode == "prediction":
        logits = response / max(temperature, 1e-6)
        return F.softmax(logits, dim=0)
    if mode == "embedding":
        return F.normalize(response, p=2, dim=0)
    if mode == "projection":
        return (response - response.mean()) / (response.std() + 1e-12)
    raise ValueError("mode must be prediction, embedding, or projection.")
