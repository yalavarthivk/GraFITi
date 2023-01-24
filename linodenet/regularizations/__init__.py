r"""Regularizations for the Linear ODE Networks.

Notes
-----
Contains regularizations in both modular and functional form.
  - See `~linodenet.regularizations.functional` for functional implementations.
  - See `~linodenet.regularizations..modular` for modular implementations.
"""

__all__ = [
    # Constants
    "REGULARIZATIONS",
    "FUNCTIONAL_REGULARIZATIONS",
    "MODULAR_REGULARIZATIONS",
    # Types
    "Regularization",
    "FunctionalRegularization",
    "ModularRegularization",
    # Sub-Modules
    "functional",
    "modular",
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
    # Classes
    "Banded",
    "Diagonal",
    "Identity",
    "LogDetExp",
    "Masked",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
]


from collections.abc import Callable
from typing import Final, TypeAlias

from torch import Tensor, nn

from linodenet.regularizations import functional, modular
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
from linodenet.regularizations.modular import (
    Banded,
    Diagonal,
    Identity,
    LogDetExp,
    Masked,
    MatrixNorm,
    Normal,
    Orthogonal,
    SkewSymmetric,
    Symmetric,
)

FunctionalRegularization: TypeAlias = Callable[[Tensor], Tensor]
r"""Type hint for modular regularizations."""

ModularRegularization: TypeAlias = nn.Module
r"""Type hint for modular regularizations."""

Regularization: TypeAlias = FunctionalRegularization | ModularRegularization
r"""Type hint for projections."""

FUNCTIONAL_REGULARIZATIONS: Final[dict[str, FunctionalRegularization]] = {
    "banded": banded,
    "diagonal": diagonal,
    "identity": identity,
    "logdetexp": logdetexp,
    "masked": masked,
    "matrix_norm": matrix_norm,
    "normal": normal,
    "orthogonal": orthogonal,
    "skew_symmetric": skew_symmetric,
    "symmetric": symmetric,
}
r"""Dictionary of all available modular metrics."""

MODULAR_REGULARIZATIONS: Final[dict[str, type[nn.Module]]] = {
    "Banded": Banded,
    "Diagonal": Diagonal,
    "Identity": Identity,
    "LogDetExp": LogDetExp,
    "Masked": Masked,
    "MatrixNorm": MatrixNorm,
    "Normal": Normal,
    "Orthogonal": Orthogonal,
    "SkewSymmetric": SkewSymmetric,
    "Symmetric": Symmetric,
}
r"""Dictionary of all available modular metrics."""

REGULARIZATIONS: Final[
    dict[str, FunctionalRegularization | type[ModularRegularization]]
] = {
    **FUNCTIONAL_REGULARIZATIONS,
    **MODULAR_REGULARIZATIONS,
}
r"""Dictionary containing all available projections."""
