#!/usr/bin/env python

from typing import Final

import torch
from torch import Tensor, jit, nn


class PositionalEncoder(nn.Module):
    num_dim: Final[int]
    scale: Final[float]
    scales: Tensor

    def __init__(self, num_dim: int, scale: float) -> None:
        super().__init__()
        self.num_dim = num_dim
        self.scale = float(scale)
        scales = self.scale ** (-torch.arange(0, num_dim + 2, 2) / num_dim)
        assert scales[0] == 1.0, "Something went wrong."
        self.register_buffer("scales", scales, persistent=True)

    @jit.export
    def forward(self, t: Tensor) -> Tensor:
        z = torch.einsum("..., d -> ...d", t, self.scales)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    @jit.script
    @staticmethod
    def inverse(t: Tensor) -> Tensor:
        return torch.asin(t[..., 0])


encoder = jit.script(PositionalEncoder(5, 1.23))

x = torch.randn(5)
y = encoder(x)
encoder.inverse(y)
