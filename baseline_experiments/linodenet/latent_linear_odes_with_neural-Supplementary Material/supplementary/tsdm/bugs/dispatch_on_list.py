#!/usr/bin/env python

from collections.abc import Sequence
from functools import singledispatch
from typing import Union


@singledispatch
def foo(x: Union[Sequence[int], Sequence[str]]):
    print("fallback", x)


@foo.register
def _(x: Sequence[int]):
    print("all ints")


@foo.register
def _(x: Sequence[str]):
    print("all strings")


foo([1, 2, 3])
foo(["a", "b", "c"])
foo([1, "a"])
