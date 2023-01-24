#!/usr/bin/env python
# coding: utf-8

# # Title

# In[1]:


# get_ipython().run_line_magic('config', "InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.")
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import logging

logging.basicConfig(level=logging.INFO)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, DatetimeIndex, Series

np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
rng = np.random.default_rng()


# In[3]:


from tsdm.datasets import Electricity

ds = Electricity()


# In[4]:


dt = ds.index
time_resolution = [
    dt.year,
    dt.month,
    dt.day,
    dt.hour,
    dt.minute,
    dt.second,
    dt.microsecond,
    dt.nanosecond,
]


# In[5]:


import pandas as pd

pd.Timedelta(ds.index.inferred_freq)


# In[6]:


class SocialTime:
    level_codes = {
        "Y": "year",
        "M": "month",
        "W": "weekday",
        "D": "day",
        "h": "hour",
        "m": "minute",
        "s": "second",
        "µ": "microsecond",
        "n": "nanosecond",
    }

    def __init__(self, levels: str = "YMWDhms") -> None:
        self.levels = [self.level_codes[k] for k in levels]

    def fit(self, x: Series, /) -> None:
        self.original_type = type(x)
        self.original_name = x.name
        self.original_dtype = x.dtype
        self.rev_cols = [l for l in self.levels if l != "weekday"]
        # self.new_names = {level:f"{x.name}_{level}" for level in self.levels}
        # self.rev_names = {f"{x.name}_{level}":level for level in self.levels if level != "weekday"}

    def encode(self, x, /):
        if isinstance(x, DatetimeIndex):
            res = {level: getattr(x, level) for level in self.levels}
        else:
            res = {level: getattr(x, level) for level in self.levels}
        return DataFrame.from_dict(res)

    def decode(self, x, /):
        x = x[self.rev_cols]
        s = pd.to_datetime(x)
        return self.original_type(s, name=self.original_name, dtype=self.original_dtype)


# In[7]:


enc = SocialTime()
enc.fit(ds.index)
encoded = enc.encode(ds.index)


# In[8]:


enc.decode(encoded)


# In[9]:


ds.index


# In[10]:


from tsdm.encoders import *

# In[11]:


enc = FrameEncoder(PeriodicEncoder(), duplicate=True) @ SocialTimeEncoder()
enc.fit(ds.index)
enc.encode(ds.index)


# In[12]:


# In[14]:


PeriodicEncoder(5)


# In[15]:


enc = PeriodicSocialTimeEncoder()
enc.fit(ds.index)
encoded = enc.encode(ds.index)


# In[17]:


decoded = enc.decode(encoded)


# In[ ]:


from pandas.core.indexes.frozen import FrozenList

encoded[FrozenList(["cos_year", "sin_year"])]


# In[ ]:
