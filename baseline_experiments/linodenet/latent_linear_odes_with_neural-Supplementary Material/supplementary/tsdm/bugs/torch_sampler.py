#!/usr/bin/env python


from abc import abstractmethod
from collections.abc import Sized
from typing import Generic, Iterator, Optional, TypeVar

from torch.utils.data import Sampler, SequentialSampler

T_co = TypeVar("T_co", covariant=True)


class MySampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    @abstractmethod
    def __init__(self, data_source: Optional[Sized]) -> None:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class MySequentialSampler(MySampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


def demo_func(x: Sized) -> int:
    return len(x)


def sampler_breaks(x: Sampler) -> int:
    return demo_func(x)


def mysampler_works(x: MySampler) -> int:
    return demo_func(x)


Sized.register(Sampler)

A: Sized = SequentialSampler(range(9))
B: Sized = MySequentialSampler(range(9))
