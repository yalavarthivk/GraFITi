r"""Encoder Models."""

__all__ = [
    # Types
    "Encoder",
    # Meta-Objects
    "ENCODERS",
    # Classes
    "FTTransformer",
    "LinearContraction",
    "MLP",
    "ResNet",
    "ResNetBlock",
    "SpectralNorm",
    "Transformer",
    "iResNet",
    "iResNetBlock",
    # Functions
    "spectral_norm",
]

from typing import Final, TypeAlias

from torch import nn

from linodenet.models.encoders.ft_transformer import FTTransformer
from linodenet.models.encoders.iresnet import (
    LinearContraction,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from linodenet.models.encoders.mlp import MLP
from linodenet.models.encoders.resnet import ResNet, ResNetBlock
from linodenet.models.encoders.transformer import Transformer

Encoder: TypeAlias = nn.Module
r"""Type hint for Encoders."""

ENCODERS: Final[dict[str, type[Encoder]]] = {
    "FTTransformer": FTTransformer,
    "MLP": MLP,
    "ResNet": ResNet,
    "Transformer": Transformer,
    "iResNet": iResNet,
}
r"""Dictionary containing all available encoders."""
