import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
import tripletformer_layers
from torch.nn.utils.rnn import pad_sequence
import pdb

class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

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
    originals: tuple[Tensor, Tensor]



def ushcn_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
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

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)
        
        context_x.append(torch.cat([t, t_target], dim = 0))
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
        x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(context_x, batch_first=True).squeeze(),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
        y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
    )



class TripletFormer(nn.Module):

    def __init__(
        self,
        input_dim=41,
        enc_attn_head=4,
        dec_attn_head=2,
        latent_dim = 128,
        dec_dim = 128,
        n_ref_points=32,
        n_layers=2,
        out_layers=2,
        out_dim=128,
        device='cuda'):
        super().__init__()
        self.dim=input_dim
        self.enc_attn_head=enc_attn_head
        self.dec_attn_head=dec_attn_head
        self.latent_dim = latent_dim
        self.n_ref_points = n_ref_points
        self.n_layers=n_layers
        self.out_layers=out_layers
        self.dec_dim=dec_dim
        self.out_dim=out_dim
        self.device=device
        self.enc = tripletformer_layers.Encoder(self.dim, self.latent_dim, self.n_ref_points, self.n_layers, self.enc_attn_head, device=device)
        self.dec = tripletformer_layers.Decoder(self.dim, self.latent_dim, self.dec_dim, self.dec_attn_head, device=device)
        self.out = tripletformer_layers.Out(self.dec_dim, self.out_dim, self.out_layers, device=device)

    def get_extrapolation(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        enc_val, enc_mask = self.enc(x_time, x_vals, x_mask)
        dec_val, true_val, target_mask = self.dec(enc_val, enc_mask, y_time, y_vals, y_mask)
        pred_val = self.out(dec_val)
        return pred_val, true_val, target_mask

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        if len(x_vals.shape) == 2:
            x_time = x_time.unsqueeze(0)
            x_vals = x_vals.unsqueeze(0)
            x_mask = x_mask.unsqueeze(0)
            y_time = y_time.unsqueeze(0)
            y_vals = y_vals.unsqueeze(0)
            y_mask = y_mask.unsqueeze(0)
        pred_val, true_val, target_mask = self.get_extrapolation(x_time, x_vals, x_mask, y_time, y_vals, y_mask)

        return pred_val, true_val, target_mask.to(torch.bool)