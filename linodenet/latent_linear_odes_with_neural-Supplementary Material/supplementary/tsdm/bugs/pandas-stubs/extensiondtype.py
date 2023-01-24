#!/usr/bin/env python

import pandas as pd
from pandas.api.extensions import ExtensionDtype

assert issubclass(pd.Int64Dtype, ExtensionDtype)  # ✔
x: type[ExtensionDtype] = pd.Int64Dtype
