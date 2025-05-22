#!/usr/bin/env python

import pandas as pd

print(pd.isna({1, 2, 3, float("nan")}))
