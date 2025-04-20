import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from grafiti import grafiti_layers
from torch.nn.utils.rnn import pad_sequence
import pdb


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×(N+K):   the input timestamps.
    x_vals: Tensor  # B×(N+K)×D: the input values.
    x_mask: Tensor  # B×(N+K)×D: the input mask.

    y_vals: Tensor  # B×(N+K)×D: the target values.
    y_mask: Tensor  # B×(N+K)×D: teh target mask.


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor


def tsdm_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_vals.append(y)
        y_mask.append(mask_y)
        context_x.append(torch.cat([t, t_target], dim=0))

        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0),
        x_mask=pad_sequence(context_mask, batch_first=True),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0),
        y_mask=pad_sequence(target_mask, batch_first=True),
    )


class GraFITi(nn.Module):

    def __init__(
        self, input_dim=41, attn_head=4, latent_dim=128, n_layers=2, device="cuda"
    ):
        super().__init__()
        self.dim = input_dim  # input dimensions
        self.attn_head = attn_head  # no. of attention heads
        self.latent_dim = latent_dim  # latend dimension
        self.n_layers = n_layers  # number of grafiti layers
        self.device = device  # cpu or gpu
        self.grafiti_ = grafiti_layers.grafiti_(
            self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device
        )  # applying grafiti

    def forward(self, x_time, x_vals, x_mask, y_mask):
        yhat = self.grafiti_(x_time, x_vals, x_mask, y_mask)
        return yhat
