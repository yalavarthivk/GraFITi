#!/usr/bin/env python

from typing import Generic, Optional, TypeVar


class A:
    pass


class B(A):
    pass


T = TypeVar("T", A, tuple[A, ...], covariant=True)
S = TypeVar("S", A, tuple[A, ...], covariant=True)


class Foo(Generic[S, T]):
    fallback_class: S
    fallback_tuple: T

    def __init__(self, x: Optional[S] = None, y: Optional[T] = None):
        if x is None:
            # Incompatible types in assignment (expression has type "int", variable has type "Tuple[int, ...]")
            self.fallback_class = A()
        else:
            self.fallback_class = x

        if y is None:
            # Incompatible types in assignment (expression has type "Tuple[]", variable has type "int")
            self.fallback_tuple = ()
        else:
            self.fallback_tuple = y
