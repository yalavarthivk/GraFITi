#!/usr/bin/env python
r"""Tests regarding projection components."""


import torch

from linodenet import projections

B, N = 32, 5

x = torch.randn(B, N, N)

m = torch.randn(B, N, N) > 0.5

x_obs = torch.where(m, x, float("nan"))


projections.diagonal(x)
