r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in functional form.
  - See `~linodenet.regularizations.modular` for modular implementations.
"""


__all__ = [
    # Functions
    "banded",
    "diagonal",
    "identity",
    "logdetexp",
    "masked",
    "matrix_norm",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
]

from typing import Optional

import torch.linalg
from torch import BoolTensor, Tensor, jit

from linodenet.projections import functional as projections


@jit.script
def logdetexp(x: Tensor, p: float = 1.0) -> Tensor:
    r"""Bias $\det(e^A)$ towards 1.

    .. Signature:: ``(..., n, n) -> ...``

    By Jacobi's formula

    .. math:: \det(e^A) = e^{\tr(A)} ⟺ \log(\det(e^A)) = \tr(A)

    In particular, we can regularize the LinODE model by adding a regularization term of the form

    .. math:: |\tr(A)|^p
    """
    traces = torch.sum(torch.diagonal(x, dim1=-1, dim2=-2), dim=-1)
    return torch.abs(traces) ** p


@jit.script
def matrix_norm(
    r: Tensor, p: Optional[float] = None, size_normalize: bool = True
) -> Tensor:
    r"""Return the matrix regularization term.

    .. Signature:: ``(..., n, n) -> ...``
    """
    if p is None:
        s = torch.linalg.matrix_norm(r)
    else:
        s = torch.linalg.matrix_norm(r, ord=p)
    if size_normalize:
        s = s / r.shape[-1]
    return s


@jit.script
def identity(
    x: Tensor, p: Optional[float] = None, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being zero.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X∥_F^2
    """
    return matrix_norm(x, p=p, size_normalize=size_normalize)


@jit.script
def skew_symmetric(
    x: Tensor, p: Optional[float] = None, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being skew-symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = -X
    """
    r = x - projections.skew_symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def symmetric(
    x: Tensor, p: Optional[float] = None, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = +X
    """
    r = x - projections.symmetric(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def orthogonal(
    x: Tensor, p: Optional[float] = None, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being orthogonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = 𝕀
    """
    r = x - projections.orthogonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def normal(
    x: Tensor, p: Optional[float] = None, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being normal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = XX^⊤
    """
    r = x - projections.normal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def diagonal(
    x: Tensor, p: Optional[float] = None, size_normalize: bool = False
) -> Tensor:
    r"""Bias the matrix towards being diagonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X⊙𝕀 = X
    """
    r = x - projections.diagonal(x)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def banded(
    x: Tensor,
    u: int = 0,
    l: int = 0,
    p: Optional[float] = None,
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being banded.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X⊙B = X
    """
    r = x - projections.banded(x, u=u, l=l)
    return matrix_norm(r, p=p, size_normalize=size_normalize)


@jit.script
def masked(
    x: Tensor,
    m: BoolTensor,
    p: Optional[float] = None,
    size_normalize: bool = False,
) -> Tensor:
    r"""Bias the matrix towards being masked.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X⊙M = X
    """
    r = x - projections.masked(x, m=m)
    return matrix_norm(r, p=p, size_normalize=size_normalize)
