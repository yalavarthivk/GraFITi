#!/usr/bin/env python

import pandas as pd

data = pd.date_range("2022-01-01", "2022-01-31", freq="D")
x = pd.Timestamp("2022-01-17")
idx = pd.Index(data, name="date")
print(data[x <= idx])  # error
dt_idx = pd.DatetimeIndex(data, name="date")
print(data[x <= dt_idx])  # ok
