r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in modular  form.
  - See `~linodenet.regularizations.functional` for functional implementations.
"""

__all__ = [
    # Classes
    "Banded",
    "Diagonal",
    "Identity",
    "LogDetExp",
    "Masked",
    "MatrixNorm",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
]

from typing import Optional

from torch import BoolTensor, Tensor, nn

from linodenet.regularizations.functional import (
    banded,
    diagonal,
    identity,
    logdetexp,
    masked,
    matrix_norm,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)


class LogDetExp(nn.Module):
    r"""Bias $\det(e^A)$ towards 1.

    .. Signature:: ``(..., n, n) -> ...``

    By Jacobi's formula

    .. math:: \det(e^A) = e^{\tr(A)} ⟺ \log(\det(e^A)) = \tr(A)

    In particular, we can regularize the LinODE model by adding a regularization term of the form

    .. math:: |\tr(A)|^p
    """

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias $\det(e^A)$ towards 1."""
        return logdetexp(x, self.p)


class MatrixNorm(nn.Module):
    r"""Return the matrix regularization term.

    .. Signature:: ``(..., n, n) -> ...``
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards zero matrix."""
        return matrix_norm(x, self.p, self.size_normalize)


class Identity(nn.Module):
    r"""Identity regularization.

    .. Signature:: ``(..., n, n) -> ...``
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards zero matrix."""
        return identity(x, self.p, self.size_normalize)


class SkewSymmetric(nn.Module):
    r"""Bias the matrix towards being skew-symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = -X
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards skew-symmetric matrix."""
        return skew_symmetric(x, self.p, self.size_normalize)


class Symmetric(nn.Module):
    r"""Bias the matrix towards being symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = +X
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards symmetric matrix."""
        return symmetric(x, self.p, self.size_normalize)


class Orthogonal(nn.Module):
    r"""Bias the matrix towards being orthogonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = 𝕀
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards orthogonal matrix."""
        return orthogonal(x, self.p, self.size_normalize)


class Normal(nn.Module):
    r"""Bias the matrix towards being orthogonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = 𝕀
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return normal(x, self.p, self.size_normalize)


class Diagonal(nn.Module):
    r"""Bias the matrix towards being diagonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X⊙𝕀 = X
    """

    def __init__(self, p: Optional[float] = None, size_normalize: bool = True):
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards diagonal matrix."""
        return diagonal(x, self.p, self.size_normalize)


class Banded(nn.Module):
    r"""Bias the matrix towards being banded.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X⊙B = X
    """

    def __init__(
        self,
        u: int = 0,
        l: int = 0,
        p: Optional[float] = None,
        size_normalize: bool = True,
    ):
        super().__init__()
        self.u = u
        self.l = l
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards banded matrix."""
        return banded(
            x, u=self.u, l=self.l, p=self.p, size_normalize=self.size_normalize
        )


class Masked(nn.Module):
    r"""Bias the matrix towards being masked.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X⊙M = X
    """

    def __init__(
        self, m: BoolTensor, p: Optional[float] = None, size_normalize: bool = True
    ):
        super().__init__()
        self.m = m
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards masked matrix."""
        return masked(x, m=self.m, p=self.p, size_normalize=self.size_normalize)
