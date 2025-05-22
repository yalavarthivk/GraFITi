#!/usr/bin/env python

from collections.abc import Callable
from functools import wraps
from typing import Optional, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def foo(
    func: Callable[P, R], before_func: Optional[Callable[P, None]] = None
) -> Callable[P, R]:

    if before_func is not None:

        reveal_type(before_func)  # <--  Revealed type is "def (*Any, **Any) -> Any"

        @wraps(func)  # problem persists when removing @wraps
        def wrapper(*args, **kwargs):
            reveal_type(
                before_func
            )  # Revealed type is "Union[def (*Any, **Any) -> Any, None]"
            before_func(*args, **kwargs)  # error: "None" not callable
            return func(*args, **kwargs)

        return wrapper

    return func
