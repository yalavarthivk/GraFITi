#!/usr/bin/env python
from __future__ import annotations

from typing import *

# Related to https://github.com/python/mypy/issues/3737


class A:
    pass


Z = TypeVar("Z", bound=A)


class Foo(Generic[Z]):
    x: Z

    def __init__(self, x: Optional[Z] = None) -> None:
        if x is None:
            # Z should be substituted with A in this case.
            self.x = cast(Z, A())
            # reveal_type(self.x)
        else:
            self.x = x


obj: Foo[A] = Foo()
# reveal_type(obj.x)

other = Foo(A())
# reveal_type(other.x)

# class A: ...
# class B(A): ...
#
#
# X = A | MutableMapping[str, A]
#
#
# foo: Mapping[str, A] = {"a": B()}
# bar: MutableMapping[str, A] = {"a": B()}
