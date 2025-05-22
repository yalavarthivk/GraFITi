#!/usr/bin/env python

from collections.abc import Iterable, Sequence
from functools import singledispatchmethod
from typing import Union, cast, overload


class FloatSequence(Sequence[float]):
    __slots__ = ("data",)
    data: tuple[float, ...]

    def __init__(self, more: Iterable[float]) -> None:
        if isinstance(more, FloatSequence):
            self.data = cast(FloatSequence, more).data
        else:
            self.data = tuple(float(x) for x in more)

    @overload
    def __getitem__(self, index: int) -> float:
        pass

    @overload
    def __getitem__(self, index: slice) -> "FloatSequence":
        pass

    def __getitem__(self, index: Union[int, slice]) -> Union[float, "FloatSequence"]:
        if isinstance(index, slice):
            return FloatSequence(self.data[index])
        else:
            return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class IntSequence(Sequence[int]):
    __slots__ = ("data",)
    data: tuple[int, ...]

    def __init__(self, more: Iterable[int]) -> None:
        if isinstance(more, IntSequence):
            self.data = cast(IntSequence, more).data
        else:
            self.data = tuple(int(x) for x in more)

    @singledispatchmethod
    def __getitem__(  # error: Signature of "__getitem__" incompatible with supertype "Sequence"
        self, index: Union[int, slice]
    ) -> Union[int, "IntSequence"]:
        raise TypeError("unsupported index type")

    @__getitem__.register
    def _(self, index: int) -> int:
        return self.data[index]

    @__getitem__.register
    def _(self, index: slice) -> "IntSequence":
        return IntSequence(self.data[index])

    def __len__(self) -> int:
        return len(self.data)
