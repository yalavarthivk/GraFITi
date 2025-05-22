#!/usr/bin/env python


from abc import ABC, abstractmethod
from typing import Generic, Literal, Sequence, TypeVar

KEYS = TypeVar("KeyType")
"""Type hint for index"""


class BaseClass(ABC, Generic[KEYS]):
    # index: Sequence[KeyType]
    # """The index"""

    @property
    @abstractmethod
    def keys(self) -> Sequence[KEYS]:
        ...

    @abstractmethod
    def values(self) -> dict[KEYS, int]:
        ...


class ExampleClass(BaseClass):
    KEYS = Literal["a", "b", "c"]
    # index: list[KeyType] = ["a", "b", "c"]
    #
    # def values(self) -> dict[KeyType, int]:
    #     return {key: 0 for key in self.index}
