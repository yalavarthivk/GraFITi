r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in both modular and functional form.
  - See `~linodenet.projections.functional` for functional implementations.
  - See `~linodenet.projections.modular` for modular implementations.
"""


__all__ = [
    # Constants
    "PROJECTIONS",
    "FUNCTIONAL_PROJECTIONS",
    "MODULAR_PROJECTIONS",
    # Types
    "Projection",
    "FunctionalProjection",
    "ModularProjection",
    # Sub-Modules
    "functional",
    "modular",
    # Functions
    "banded",
    "diagonal",
    "identity",
    "masked",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    # Classes
    "Banded",
    "Diagonal",
    "Identity",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
]


from collections.abc import Callable
from typing import Final, TypeAlias

from torch import Tensor, nn

from linodenet.projections import functional, modular
from linodenet.projections.functional import (
    banded,
    diagonal,
    identity,
    masked,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
)
from linodenet.projections.modular import (
    Banded,
    Diagonal,
    Identity,
    Masked,
    Normal,
    Orthogonal,
    SkewSymmetric,
    Symmetric,
)

FunctionalProjection: TypeAlias = Callable[[Tensor], Tensor]
r"""Type hint for modular regularizations."""

ModularProjection: TypeAlias = nn.Module
r"""Type hint for modular regularizations."""

Projection: TypeAlias = FunctionalProjection | ModularProjection  # matrix to matrix
r"""Type hint for projections."""

MODULAR_PROJECTIONS: Final[dict[str, type[nn.Module]]] = {
    "Banded": Banded,
    "Diagonal": Diagonal,
    "Identity": Identity,
    "Masked": Masked,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
}
r"""Dictionary of all available modular metrics."""

FUNCTIONAL_PROJECTIONS: Final[dict[str, FunctionalProjection]] = {
    "banded": banded,
    "diagonal": diagonal,
    "identity": identity,
    "masked": masked,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
}
r"""Dictionary of all available modular metrics."""

PROJECTIONS: Final[dict[str, FunctionalProjection | type[ModularProjection]]] = {
    **FUNCTIONAL_PROJECTIONS,
    **MODULAR_PROJECTIONS,
}
r"""Dictionary containing all available projections."""
