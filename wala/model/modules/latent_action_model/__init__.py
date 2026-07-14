"""Latent action model used by WALA."""

from .latent_action_model import (
    ContinuousTransitionBottleneckTokenizer,
    LatentActionToTransitionTokens,
)

__all__ = [
    "ContinuousTransitionBottleneckTokenizer",
    "LatentActionToTransitionTokens",
]
