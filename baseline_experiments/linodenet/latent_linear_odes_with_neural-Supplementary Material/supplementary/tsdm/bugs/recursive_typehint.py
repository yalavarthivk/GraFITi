#!/usr/bin/env python


import logging

__logger__ = logging.getLogger(__name__)


def f(x: int | Collection[int] | Mapping[str, int]) -> int:
    r"""Recursively sum up all the values of possibly nested data."""

    reveal_type(x)

    if isinstance(x, int):
        reveal_type(x)
        return x

    reveal_type(x)

    if isinstance(x, Mapping):
        reveal_type(x)  # <- Here, mypy thinks that x could be Mapping[Any, Any]
        return sum(f(y) for y in x.values())

    reveal_type(x)  # <- Here, mypy thinks this is Collection only!

    if isinstance(x, Collection):
        reveal_type(x)  # <- mypy thinks this is Collection | Mapping
        return sum(f(y) for y in x)

    raise TypeError(f"unsupported type: {type(x)}")
