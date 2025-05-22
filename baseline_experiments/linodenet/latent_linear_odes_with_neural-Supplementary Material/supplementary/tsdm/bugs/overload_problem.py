#!/usr/bin/env python

from typing import overload

import torch
from pandas import DataFrame
from torch import Tensor


class TensorEncoder:
    def __init__(self):
        super().__init__()

    @overload
    def encode(self, x: DataFrame) -> Tensor:
        ...

    @overload
    def encode(self, x: tuple[DataFrame, ...]) -> tuple[Tensor, ...]:
        ...

    def encode(self, x):
        if isinstance(x, Tensor):
            return tuple(self.encode(y) for y in x)
        return torch.tensor(x.values)


@overload
def foo(x: Tensor) -> Tensor:
    ...


@overload
def foo(x: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    ...
