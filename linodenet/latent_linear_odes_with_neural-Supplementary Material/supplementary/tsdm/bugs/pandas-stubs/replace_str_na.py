#!/usr/bin/env python

import pandas as pd

frame = pd.DataFrame(["N/A", "foo", "bar"])
frame = frame.replace("N/A", pd.NA)

print(frame)
