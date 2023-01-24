r"""Embedding Models."""

__all__ = [
    # Types
    "Embedding",
    # Meta-Objects
    "EMBEDDINGS",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
]

from typing import Final, TypeAlias

from torch import nn

from linodenet.models.embeddings._embeddings import ConcatEmbedding, ConcatProjection

Embedding: TypeAlias = nn.Module
r"""Type hint for Embeddings."""


EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "ConcatEmbedding": ConcatEmbedding,
    "ConcatProjection": ConcatProjection,
}
r"""Dictionary of available embeddings."""
