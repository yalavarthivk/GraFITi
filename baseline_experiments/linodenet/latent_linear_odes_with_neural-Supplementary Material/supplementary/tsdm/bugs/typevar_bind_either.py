#!/usr/bin/env python

from typing import *


class A:
    pass


class B(A):
    pass


Z = TypeVar("Z", bound=A | Sequence[A] | Mapping[str, A])


class Foo(Generic[Z]):
    inst_or_list: Z

    def __init__(self, x: Z):
        self.inst_or_list = x


a_inst: Foo[A] = Foo(A())  # ✔
a_list: Foo[list[A]] = Foo([A(), A()])  # ✔
b_inst: Foo[B] = Foo(B())  # ✔
b_list: Foo[list[A]] = Foo([B(), B()])  # ✔

test_list: Foo[list[B]] = Foo([B(), B()])  # ✘ raises type-var error
test_mapp: Foo[dict[str, B]] = Foo({"1": B()})  # ✘ raises type-var error
