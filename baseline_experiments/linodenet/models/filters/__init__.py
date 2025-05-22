r"""Different Filter models to be used in conjunction with LinodeNet."""

__all__ = [
    # Types
    "Filter",
    # Constants
    "FILTERS",
    # Classes
    "FilterABC",
    "KalmanCell",
    "KalmanFilter",
    "PseudoKalmanFilter",
    "RecurrentCellFilter",
    "SequentialFilter",
    "SequentialFilterBlock",
]

from typing import Final, TypeAlias

from torch import nn

from linodenet.models.filters._filters import (
    FilterABC,
    KalmanCell,
    KalmanFilter,
    PseudoKalmanFilter,
    RecurrentCellFilter,
    SequentialFilter,
    SequentialFilterBlock,
)

Filter: TypeAlias = nn.Module
r"""Type hint for Filters"""

FILTERS: Final[dict[str, type[nn.Module]]] = {
    "FilterABC": FilterABC,
    "KalmanCell": KalmanCell,
    "KalmanFilter": KalmanFilter,
    "LinearFilter": PseudoKalmanFilter,
    "RecurrentCellFilter": RecurrentCellFilter,
    "SequentialFilter": SequentialFilter,
    "SequentialFilterBlock": SequentialFilterBlock,
}
r"""Dictionary of all available filters."""
