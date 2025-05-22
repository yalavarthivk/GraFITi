#!/usr/bin/env python3

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(eq=False)
class Foo(Mapping):
    c: str

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __len__(self) -> int:
        return len(self.__dict__)


a = {"c": "test"}
b = Foo(**a)

print(b.__dict__)
assert b.__eq__(a)  # ✔
assert a.__eq__(b)  # DeprecationWarning: NotImplemented
assert dict(b) == a  # ✔
assert list(b.keys()) == list(a.keys())  # ✔
assert list(b.values()) == list(a.values())  # ✔
assert list(b.items()) == list(a.items())  # ✔
assert b.keys() == a.keys()  # ✔
assert b.values().__eq__(a.values())  # DeprecationWarning: NotImplemented
assert a.values().__eq__(b.values())  # DeprecationWarning: NotImplemented
assert b.values() == a.values()  # ✘  Fails!!
assert b.items() == a.items()  # ✔
assert dict(b.items()) == dict(a.items())  # ✔
assert b == a  # ✔
