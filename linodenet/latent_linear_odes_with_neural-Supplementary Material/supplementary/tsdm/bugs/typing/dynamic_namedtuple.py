#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from collections.abc import Iterable
from typing import NamedTuple, Protocol, TypeGuard, TypeVar, cast

T = TypeVar("T")


class _NamedTuple(tuple[T, ...], ABC):
    r"""To check for namedtuple."""
    __slots__ = ()

    @abstractmethod
    def __init__(self, *args: T) -> None:
        ...

    @classmethod
    @abstractmethod
    def _make(cls, iterable: Iterable[T]) -> _NamedTuple[T]:
        ...

    @abstractmethod
    def _replace(self, /, **kwds: dict[str, T]) -> None:
        ...

    @property
    @abstractmethod
    def _fields(self) -> tuple[T, ...]:
        ...

    @property
    @abstractmethod
    def _field_defaults(self) -> dict[str, T]:
        ...

    @abstractmethod
    def _asdict(self) -> dict[str, T]:
        ...


def register_namedtuple(obj: object, fields: list[str], /, *, name: str) -> None:
    if not name.isidentifier():
        raise ValueError(f"{name} is not a valid identifier!")
    tuple_type = cast(type[_NamedTuple], namedtuple(name, fields))
    # reveal_type(tuple_type)
    _NamedTuple.register(tuple_type)

    # create a unique identifier and store it in globals
    identifier = f"_{name}_{obj.__class__.__name__}_{hash(obj)}"
    tuple_type.__qualname__ = identifier
    if identifier in globals():
        raise RuntimeError(f"A class of name '{identifier}' exists!!")
    globals()[identifier] = tuple_type

    setattr(obj, "tuple_type", tuple_type)

    # if hasattr(obj, "__del__"):

    # return tuple_type


class Foo:
    tuple_type: type[_NamedTuple]

    def __init__(self, fields: list[str]):
        super().__init__()
        register_namedtuple(self, fields, name="FooTuple")

        # self.tuple_type = register_namedtuple(self, fields, name="FooTuple")


foo = Foo(["a", "b", "c"])
FooTup = foo.tuple_type
footup = foo.tuple_type(1, 2, 3)

assert isinstance(footup, tuple)
assert issubclass(FooTup, tuple)
assert isinstance(footup, _NamedTuple)
assert issubclass(FooTup, _NamedTuple)

from collections import namedtuple


class ClassWithTuple:
    tuple_id: str
    tuple_type: type[tuple]

    def __init__(self, name: str, fields: list[str]) -> None:
        tuple_type = namedtuple(name, fields)

        # create a unique identifier and store it in globals
        tuple_id = f"_{name}_{self.__class__.__name__}_{hash(self)}"
        tuple_type.__qualname__ = tuple_id

        if tuple_id in globals():
            raise RuntimeError(f"A class '{identifier}' exists!")
        globals()[tuple_id] = tuple_type

        self.tuple_id = tuple_id
        self.tuple_type = tuple_type

    def __call__(self, *args) -> tuple:
        return self.tuple_type(*args)

    def __del__(self):
        del globals()[self.tuple_id]
        del self


obj = ClassWithTuple("FooTuple", ["a", "b", "c"])
obj(1, 2, 3)
