#!/usr/bin/env python

# Already reported https://github.com/PyCQA/pycodestyle/issues/951


def f(x, y, /) -> None:
    ...


def g(x, y, /) -> None:
    ...
