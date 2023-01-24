r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in functional form.
  - See `~linodenet.projections.modular` for modular implementations.
"""

__all__ = [
    # Functions
    "banded",
    "diagonal",
    "identity",
    "masked",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
]


import torch
from torch import BoolTensor, Tensor, jit


@jit.script
def identity(x: Tensor) -> Tensor:
    r"""Return x as-is.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2
    """
    return x


@jit.script
def symmetric(x: Tensor) -> Tensor:
    r"""Return the closest symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = Y

    One can show analytically that Y = ½(X + X^⊤) is the unique minimizer.
    """
    return (x + x.swapaxes(-1, -2)) / 2


@jit.script
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """
    return (x - x.swapaxes(-1, -2)) / 2


@jit.script
def normal(x: Tensor) -> Tensor:
    r"""Return the closest normal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^⊤Y = YY^⊤

    **The Lagrangian:**

    .. math:: ℒ(Y, Λ) = ½∥X-Y∥_F^2 + ⟨Λ, [Y, Y^⊤]⟩

    **First order necessary KKT condition:**

    .. math::
            0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λ^⊤) - (Λ + Λ^⊤)Y
        \\⟺ Y &= X + [Y, Λ]

    **Second order sufficient KKT condition:**

    .. math::
             ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@jit.script
def orthogonal(x: Tensor) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y^𝖳 Y = 𝕀 = YY^𝖳

    One can show analytically that $Y = UV^𝖳$ is the unique minimizer,
    where $X=UΣV^𝖳$ is the SVD of $X$.

    References
    ----------
    - `<https://math.stackexchange.com/q/2215359>`_
    """
    U, _, V = torch.svd(x, some=False, compute_uv=True)
    return torch.einsum("...ij, ...kj -> ...ik", U, V)


@jit.script
def diagonal(x: Tensor) -> Tensor:
    r"""Return the closest diagonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y⊙𝕀 = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕀⊙X$.
    """
    eye = torch.eye(x.shape[-1], dtype=torch.bool, device=x.device)
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return torch.where(eye, x, zero)


@jit.script
def banded(x: Tensor, u: int = 0, l: int = 0) -> Tensor:
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y⊙B = Y

    One can show analytically that the unique smallest norm minimizer is $Y = B⊙X$.
    """
    x = torch.triu(x, diagonal=u)
    x = torch.tril(x, diagonal=l)
    return x


@jit.script
def masked(x: Tensor, m: BoolTensor) -> Tensor:
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2 s.t. Y⊙M = Y

    One can show analytically that the unique smallest norm minimizer is $Y = M⊙X$.
    """
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return torch.where(m, x, zero)
