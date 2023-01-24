r"""Utility functions."""

__all__ = [
    # Sub-Modules
    "layers",
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten",
    "initialize_from",
    "initialize_from_config",
    "is_dunder",
    "pad",
    # Classes
    "ReZeroCell",
    "ReverseDense",
    "Repeat",
    "Series",
    "Parallel",
    "Multiply",
]

from linodenet.utils import layers
from linodenet.utils._util import (
    autojit,
    deep_dict_update,
    deep_keyval_update,
    flatten,
    initialize_from,
    initialize_from_config,
    is_dunder,
    pad,
)
from linodenet.utils.generic_layers import Multiply, Parallel, Repeat, Series
from linodenet.utils.layers import ReverseDense, ReZeroCell
