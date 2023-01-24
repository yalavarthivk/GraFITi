#!/usr/bin/env python

import pandas as pd

s = pd.Series([1.2, float("nan")])
s = s.fillna(pd.NA)
print(s)
