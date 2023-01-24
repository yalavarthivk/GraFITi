r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes
-----
Contains initializations in functional form.
  - See `~linodenet.initializations.modular` for modular implementations.
"""

__all__ = [
    # Functions
    "canonical_skew_symmetric",
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
]

from collections.abc import Sequence
from math import prod, sqrt
from typing import Optional, TypeAlias

import torch
from scipy import stats
from torch import Tensor

SizeLike: TypeAlias = int | tuple[int, ...]
r"""Type hint for shape-like inputs."""


def gaussian(n: SizeLike, sigma: float = 1.0) -> Tensor:
    r"""Sample a random gaussian matrix, i.e. $A_{ij}∼𝓝(0,1/n)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$ if $σ=1$.

    Parameters
    ----------
    n: int or tuple[int]
      If `tuple`, the last axis is interpreted as dimension and the others as batch
    sigma: float = 1.0

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    return torch.normal(mean=torch.zeros(shape), std=sigma / sqrt(dim))


def diagonally_dominant(n: SizeLike) -> Tensor:
    r"""Sample a random diagonally dominant matrix, i.e. $A = 𝕀_n + B$,with $B_{ij}∼𝓝(0,1/n²)$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Parameters
    ----------
    n: int or tuple[int]
        If `tuple`, the last axis is interpreted as dimension and the others as batch

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    return torch.eye(dim) + torch.normal(mean=torch.zeros(shape), std=1 / dim)


def symmetric(n: SizeLike) -> Tensor:
    r"""Sample a symmetric matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    A = torch.normal(mean=torch.zeros(shape), std=1 / sqrt(dim))
    return (A + A.swapaxes(-1, -2)) / sqrt(2)


def skew_symmetric(n: SizeLike) -> Tensor:
    r"""Sample a random skew-symmetric matrix, i.e. $A^⊤ = -A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    shape = (*size, dim, dim)

    A = torch.normal(mean=torch.zeros(shape), std=1 / sqrt(dim))
    return (A - A.swapaxes(-1, -2)) / sqrt(2)


def orthogonal(n: SizeLike) -> Tensor:
    r"""Sample a random orthogonal matrix, i.e. $A^⊤ = A$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A = stats.ortho_group.rvs(dim=dim, size=num).reshape(shape)
    return Tensor(A)


def special_orthogonal(n: SizeLike) -> Tensor:
    r"""Sample a random special orthogonal matrix, i.e. $A^⊤ = A^{-1}$ with $\det(A)=1$.

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Parameters
    ----------
    n: int

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    num = prod(size)
    shape = (*size, dim, dim)

    A = stats.special_ortho_group.rvs(dim=dim, size=num).reshape(shape)
    return Tensor(A)


def canonical_skew_symmetric(n: SizeLike) -> Tensor:
    r"""Return the canonical skew symmetric matrix of size $n=2k$.

    .. math:: 𝕁_n = 𝕀_n ⊗ \begin{bmatrix}0 & +1 \\ -1 & 0\end{bmatrix}

    Normalized such that if $x∼𝓝(0,1)$, then $A⋅x∼𝓝(0,1)$.

    Parameters
    ----------
    n: int or tuple[int]

    Returns
    -------
    Tensor
    """
    # convert to tuple
    tup = (n,) if isinstance(n, int) else tuple(n)
    dim, size = tup[-1], tup[:-1]
    assert dim % 2 == 0, "The dimension must be divisible by 2!"
    dim //= 2

    J1 = torch.tensor([[0, 1], [-1, 0]])
    J = torch.kron(J1, torch.eye(dim))
    ONES = torch.ones(size)
    return torch.einsum("..., de -> ...de", ONES, J)


def low_rank(size: SizeLike, rank: Optional[int] = None) -> Tensor:
    r"""Sample a random low-rank m×n matrix, i.e. $A = UV^⊤$.

    Parameters
    ----------
    size: tuple[int] = ()
        Optional batch dimensions.
    rank: int
        Rank of the matrix

    Returns
    -------
    Tensor
    """
    if isinstance(size, int):
        shape: tuple[int, ...] = (size, size)
    elif isinstance(size, Sequence) and len(size) == 1:
        shape = (size[0], size[0])
    else:
        shape = size

    *batch, m, n = shape

    if isinstance(rank, int) and rank > min(m, n):
        raise ValueError("Rank must be smaller than min(m,n)")

    rank = max(1, min(m, n) // 2) if rank is None else rank
    U = torch.normal(mean=torch.zeros((*batch, m, rank)), std=1 / sqrt(rank))
    V = torch.normal(mean=torch.zeros((*batch, rank, n)), std=1 / sqrt(n))

    return torch.einsum("...ij, ...jk -> ...ik", U, V)


baba = 2
