r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Lemma
~~~~~

In this case: $e^{A}$



Notes
-----
Contains initializations in both modular and functional form.
  - See `~linodenet.initializations.functional` for functional implementations.
  - See `~linodenet.initializations.modular` for modular implementations.
"""

__all__ = [
    # Constants
    "INITIALIZATIONS",
    "FUNCTIONAL_INITIALIZATIONS",
    "MODULAR_INITIALIZATIONS",
    # Types
    "Initialization",
    "FunctionalInitialization",
    "ModularInitialization",
    # Sub-Modules
    "functional",
    "modular",
    # Functions
    "canonical_skew_symmetric",
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
    # Classes
]

from typing import Callable, Final, TypeAlias

from torch import Tensor, nn

from linodenet.initializations import functional, modular
from linodenet.initializations.functional import (
    canonical_skew_symmetric,
    diagonally_dominant,
    gaussian,
    low_rank,
    orthogonal,
    skew_symmetric,
    special_orthogonal,
    symmetric,
)

FunctionalInitialization: TypeAlias = Callable[
    [int | tuple[int, ...]], Tensor
]  # SizeLike to matrix
r"""Type hint for Initializations."""

ModularInitialization: TypeAlias = nn.Module
r"""Type hint for modular regularizations."""

Initialization: TypeAlias = FunctionalInitialization | ModularInitialization
r"""Type hint for initializations."""

MODULAR_INITIALIZATIONS: Final[dict[str, type[ModularInitialization]]] = {}
r"""Dictionary of all available modular metrics."""

FUNCTIONAL_INITIALIZATIONS: Final[dict[str, FunctionalInitialization]] = {
    "canonical_skew_symmetric": canonical_skew_symmetric,
    "diagonally_dominant": diagonally_dominant,
    "gaussian": gaussian,
    "low_rank": low_rank,
    "orthogonal": orthogonal,
    "skew-symmetric": skew_symmetric,
    "special-orthogonal": special_orthogonal,
    "symmetric": symmetric,
}
r"""Dictionary containing all available initializations."""

INITIALIZATIONS: Final[
    dict[str, FunctionalInitialization | type[ModularInitialization]]
] = {
    **FUNCTIONAL_INITIALIZATIONS,
    **MODULAR_INITIALIZATIONS,
}
r"""Dictionary of all available initializations."""
