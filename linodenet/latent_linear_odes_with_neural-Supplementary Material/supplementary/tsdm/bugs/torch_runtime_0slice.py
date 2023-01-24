#!/user/bin/env python3

import torch
from torch import Tensor, jit, nn


class MWE(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(2 * input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        r"""[..., input_size] -> [..., output_size]"""
        z = torch.cat([x, x], dim=-1)
        # print("")  # FIXME: Commenting this line causes RunTimeError
        return self.linear(z)


xdim, ydim = 7, 10
device = torch.device("cuda")  # bug happens both on CPU and GPU
model = jit.script(MWE(xdim, ydim)).to(device=device)  # bug only occurs with JIT

for k in range(100):
    num_observations = torch.randint(0, 3, (1,)).item()
    sample = torch.randn(num_observations, xdim, device=device)
    print(f"Sample {k=} of shape {sample.shape}, {sample.device=}")
    model.zero_grad()
    y = model(sample)
    y.norm().backward()
