#!/usr/bin/env python


from abc import abstractmethod

from tsdm.util import PatchedABCMeta, abstractattribute


class Foo(metaclass=PatchedABCMeta):
    a: int = abstractmethod(abstractattribute())

    @abstractmethod
    @abstractattribute
    def b(self) -> int:
        ...


class Bar(Foo):
    a = 2
    b: int

    def __init__(self, b: int):
        self.b = b


Bar(b=3)
