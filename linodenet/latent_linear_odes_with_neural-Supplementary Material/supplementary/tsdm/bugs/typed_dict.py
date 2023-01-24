#!/usr/bin/env python3.10

from typing import TypedDict


class Foo(TypedDict):
    a: int
    b: str


foo: Foo = {"a": 1, "b": "2", "c": 3}
