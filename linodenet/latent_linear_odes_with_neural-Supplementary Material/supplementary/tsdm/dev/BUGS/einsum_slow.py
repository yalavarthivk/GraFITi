#!/usr/bin/env python

# # Title

# In[1]:


# %config InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.
# %config InlineBackend.figure_format = 'svg'
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# import logging
# logging.basicConfig(level=logging.INFO)


# In[2]:


import gc
from itertools import product
from time import perf_counter

import numpy as np
import torch
from pandas import DataFrame, MultiIndex, Series
from tqdm.auto import tqdm

np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
rng = np.random.default_rng()


# In[3]:


reductions = []
for a, b in product("ijkl", "ijkl"):
    if a == b:
        continue
    reduction = f"{a}{b},ijkl->" + "ijkl".replace(a, "").replace(b, "")
    reductions.append(reduction)

framework = ["numpy", "torch"]
dtypes = ["float32", "float64"]
sizes = [64, 128, 256]

TORCH_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}

devices = [torch.device("cpu"), torch.device("cuda")]
columns = Series(reductions, name="reduction")
index = MultiIndex.from_product(
    [sizes, dtypes, framework], names=["size", "dtype", "framework"]
)
results = DataFrame(index=index, columns=columns, dtype=float)
results.to_csv("einsum_slow.csv")


# In[ ]:


# torch_results
for size in tqdm(sizes):
    _mat1 = torch.randn((size, size, size, size), device="cpu")
    _mat2 = torch.randn((size, size), device="cpu")

    for dtype in tqdm(dtypes, leave=False):
        mat1 = _mat1.to(dtype=TORCH_DTYPES[dtype])
        mat2 = _mat2.to(dtype=TORCH_DTYPES[dtype])

        for reduction in tqdm(reductions, leave=False):
            gc.disable()
            start = perf_counter()
            torch.einsum(reduction, mat2, mat1)
            stop = perf_counter()
            gc.enable()
            results.loc[(size, dtype, "torch"), reduction] = stop - start


# In[ ]:


# numpy results
for size in tqdm(sizes):
    _mat1 = np.random.normal(size=(size, size, size, size))
    _mat2 = np.random.normal(size=(size, size))

    for dtype in tqdm(dtypes, leave=False):
        mat1 = _mat1.astype(dtype)
        mat2 = _mat2.astype(dtype)

        for reduction in tqdm(reductions, leave=False):
            gc.disable()
            start = perf_counter()
            np.einsum(reduction, mat2, mat1, optimize=False)
            stop = perf_counter()
            gc.enable()
            results.loc[(size, dtype, "numpy"), reduction] = stop - start


# In[ ]:


df = results.round(3).sort_values(["size", "dtype", "framework"])
df.to_csv("einsum_slow.csv")


# In[ ]:
