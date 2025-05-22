#!/usr/bin/env python

from typing import *


class A:
    pass


X = TypeVar("X", bound=A | Sequence[A] | Mapping[str, A])


class Foo(Generic[X]):
    x: X

    def __init__(self, x: X):
        self.x = x

    def convert_to_default_types(self) -> X:
        if isinstance(self.x, A):
            return self.x
        if isinstance(self.x, Sequence):
            return list(self.x)
        if isinstance(self.x, Mapping):
            for k, v in self.x.items():
                ...
            return dict(**self.x)
        raise TypeError("Unexpected type")
