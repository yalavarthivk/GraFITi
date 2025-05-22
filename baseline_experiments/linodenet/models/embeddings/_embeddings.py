r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
    "LinearEmbedding",
]

from typing import Any, Final

import torch
from torch import Tensor, jit, nn


class ConcatEmbedding(nn.Module):
    r"""Maps $x ⟼ [x,w]$."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

    # BUFFERS
    scale: Tensor
    r"""BUFFER: The scaling scalar."""

    # Parameters
    padding: Tensor
    r"""PARAM: The padding vector."""

    def __init__(self, input_size: int, output_size: int, **cfg: Any) -> None:
        super().__init__()
        assert (
            input_size <= output_size
        ), f"ConcatEmbedding requires {input_size=} ≤ {output_size=}!"
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = output_size - input_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = list(x.shape[:-1]) + [self.padding_size]
        return torch.cat([x, self.padding.expand(shape)], dim=-1)

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return y[..., : self.input_size]


class ConcatProjection(nn.Module):
    r"""Maps $z = [x,w] ⟼ x$."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    padding_size: Final[int]
    r"""CONST: The size of the padding."""

    def __init__(self, input_size: int, output_size: int, **cfg: Any) -> None:
        super().__init__()
        assert (
            input_size >= output_size
        ), f"ConcatEmbedding requires {input_size=} ≥ {output_size=}!"
        self.input_size = input_size
        self.output_size = output_size
        self.padding_size = input_size - output_size
        self.padding = nn.Parameter(torch.randn(self.padding_size))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        return x[..., : self.output_size]

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., d+e)``.
        """
        shape = list(y.shape[:-1]) + [self.padding_size]
        return torch.cat([y, self.padding.expand(shape)], dim=-1)

    # TODO: Add variant with filter in latent space
    # TODO: Add Controls


class LinearEmbedding(nn.Module):
    r"""Maps $x ⟼ Ax$ and $y→A⁺y$."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    # PARAMS
    weight: Tensor
    r"""PARAM: The weight matriz."""

    # BUFFERS
    pinv_weight: Tensor
    r"""BUFFER: The pseudo-inverse of the weight."""

    def __init__(self, input_size: int, output_size: int, **cfg: Any) -> None:
        super().__init__()
        assert (
            input_size <= output_size
        ), f"ConcatEmbedding requires {input_size=} ≤ {output_size=}!"
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.register_buffer("pinv_weight", torch.linalg.pinv(self.weight))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r"""Concatenate the input with the padding.

        .. Signature:: ``(..., d) -> (..., e)``.
        """
        return torch.einsum("...d, de-> ...e", x, self.weight)

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Remove the padded state.

        .. Signature: ``(..., d+e) -> (..., d)``.
        """
        self.pinv_weight = torch.linalg.pinv(self.weight)
        return torch.einsum("...d, de-> ...e", y, self.pinv_weight)
