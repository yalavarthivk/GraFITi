#!/usr/bin/env python

from __future__ import annotations

from typing import *

SelfA = TypeVar("SelfA", bound="A")
SelfB = TypeVar("SelfB", bound="B")

# fmt: off
class A:
    def __add__(self: SelfA, other: SelfA) -> SelfA:  return self
    def __radd__(self: SelfA, other: SelfA) -> SelfA: return self

class B:
    def __add__(self: SelfB, other: SelfB) -> SelfB:  return self
    def __radd__(self: SelfB, other: SelfB) -> SelfB: return self

class SubA(A): ...
class SubB(B): ...


T = TypeVar("T", bound=A)
class Foo(Generic[T]):
    s: T
    def __init__(self, x: T, y: T):
        self.s = x + y                   # ✔

foo_a: Foo[A] = Foo(A(), A())            # ✔
foo_b: Foo[A] = Foo(SubA(), A())         # ✔
foo_e: Foo[SubA] = Foo(SubA(), SubA())   # ✔


AuB = TypeVar("AuB", bound=Union[A,B])
class Bar(Generic[AuB]):
    s: AuB
    def __init__(self, x: AuB, y: AuB):
        self.s = x + y                   # ✘ raises type[assignment, operator]

bar_a: Bar[A] = Bar(A(), A())            # ✔
bar_b: Bar[A] = Bar(SubA(), A())         # ✔
bar_e: Bar[SubA] = Bar(SubA(), SubA())   # ✔


AB = TypeVar("AB", A, B, covariant=True) # covariant=True does nothing?
class Baz(Generic[AB]):
    s: AB
    def __init__(self, x: AB, y: AB):
        self.s = x + y                   # ✔

baz_a: Baz[A] = Baz(A(), A())            # ✔
baz_b: Baz[A] = Baz(SubA(), A())         # ✔
baz_e: Baz[SubA] = Baz(SubA(), SubA())   # ✘ raises type[assignment, type-var]
# fmt: on


# X = TypeVar("X", A, B, covariant=True)
# X = TypeVar("X", bound=Union[A, B])

# obj_c: Foo[B] = Foo(B())
# obj_d: Foo[B] = Foo(SubB())
# obj_f: Foo[SubB] = Foo(SubB())


A() + A()  # ✔
B() + B()  # ✔
SubA() + SubA()  # ✔
SubB() + SubB()  # ✔
A() + SubA()  # ✔
B() + SubB()  # ✔
SubA() + A()  # ✔
SubB() + B()  # ✔
