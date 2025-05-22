import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from linodenet.models import gratif_layers
from torch.nn.utils.rnn import pad_sequence
import pdb

class Gaussian:
    mean = None
    logvar = None

class LossInfo: 
    loglik = None
    mse = None
    mae = None
    composite_loss = None

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



class GrATiF(nn.Module):

    def __init__(
        self,
        input_dim=41,
        enc_num_heads=4,
        dec_num_heads=4,
        num_ref_points=128,
        mse_weight=1.,
        norm=True,
        imab_dim = 128,
        cab_dim = 128,
        decoder_dim = 128,
        n_layers=2,
        device='cuda'):
        super().__init__()
        self.dim=input_dim
        self.enc_num_heads=enc_num_heads
        self.dec_num_heads=dec_num_heads
        self.num_ref_points=num_ref_points
        self.mse_weight=mse_weight
        self.norm=norm
        self.imab_dim = imab_dim
        self.cab_dim = cab_dim
        self.decoder_dim = decoder_dim
        self.n_layers=n_layers
        self.device=device
        self.enc = gratif_layers.Encoder(self.dim, self.imab_dim, self.n_layers, self.num_ref_points, self.enc_num_heads, device=device)
        # self.dec_att = layers.Decoder_att(self.dim, self.imab_dim, self.cab_dim, self.dec_num_heads, device=device)
        # self.O = layers.output(self.cab_dim, self.decoder_dim, device=device)

    def get_extrapolation(self, context_x, context_w, target_x, target_y):
        context_mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        X = X*context_mask
        context_mask = context_mask + target_y[:,:,self.dim:]
        output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:,:,:self.dim], target_y[:,:,self.dim:])
        return output, target_U_, target_mask_

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        # context_x = torch.cat([x_time, y_time], dim = 1)
        # x_vals_temp = torch.zeros_like(x_vals)
        # y_vals_temp = torch.zeros_like(y_vals)
        # context_vals = torch.cat([x_vals, y_vals_temp], dim=1)
        # context_mask = torch.cat([x_mask, y_vals_temp], dim=1)
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        # target_vals = torch.cat([x_vals_temp, y_vals], dim=1)
        # target_mask = torch.cat([x_vals_temp, y_mask], dim=1)
        # target_y = torch.cat([target_vals, target_mask], dim=2)

        # return context_x, context_y, context_x, target_y

        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):

        context_x, context_y, target_x, target_y = self.convert_data(x_time, x_vals, x_mask, y_time, y_vals, y_mask)
        # pdb.set_trace()
        output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y)

        return output, target_U_, target_mask_.to(torch.bool)