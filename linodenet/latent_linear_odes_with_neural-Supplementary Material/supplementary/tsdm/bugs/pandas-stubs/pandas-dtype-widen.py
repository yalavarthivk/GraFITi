#!/usr/bin/env python

import pandas

x = pandas.Series([1.0, 3.1, float("nan")])
print(x.dtype)
y = x.replace(float("nan"), pandas.NA)
print(y.dtype)  # object ✘ (expected: Float64)
z = x.replace(1, 1 + 1j)
print(z.dtype)  # complex128 ✔
