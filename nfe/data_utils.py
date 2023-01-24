import torch
import numpy as np

def tsdm_collate(batch):
    context_x = []
    context_vals = []
    context_mask = []
    target_vals = []
    target_mask = []
    idx_ = []
    x_vals = []
    x_time = []
    x_mask = []
    y_time = []
    y_vals = []
    y_mask = []
    num_obs = []
    res = dict()
    for i, sample in enumerate(batch):
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
        idx_.append(torch.ones_like(torch.cat([t, t_target], dim = 0))*i)
        context_x.append(np.concatenate((t.numpy(), t_target.numpy()),0))
        num_obs.append(t.shape[0] + t_target.shape[0])
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))

    idx = torch.cat(idx_, dim = 0)
    values = torch.cat(context_vals, dim=0) + torch.cat(target_vals, dim=0)
    mask = torch.cat(context_mask, dim=0) + torch.cat(target_mask, dim=0)

    res = dict()
    res["num_obs"]  = torch.Tensor(num_obs)
    res["times"]    = context_x
    res["X"]        = values
    res["M"]        = mask
    res["y"]        = torch.zeros([i+1,1])
    res["cov"]      = torch.zeros([i+1,1])
    res["times_val"] = None
    return res

def tsdm_collate_val(batch):
    context_x = []
    target_x = []
    context_vals = []
    context_mask = []
    target_vals = []
    target_mask = []
    idx_ = []
    x_vals = []
    x_time = []
    x_mask = []
    y_time = []
    y_vals = []
    y_mask = []
    num_obs = []
    res = dict()
    for i, sample in enumerate(batch):
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
        idx_.append(torch.ones_like(torch.cat([t, t_target], dim = 0))*i)
        context_x.append(t.numpy())
        target_x.append(t_target.numpy())

        num_obs.append(t.shape[0])
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(x)
        context_mask.append(mask_x)

        target_vals.append(y)
        target_mask.append(mask_y)

    values = torch.cat(context_vals, dim=0) 
    values_val = torch.cat(target_vals, dim=0)
    mask = torch.cat(context_mask, dim=0) 
    mask_val = torch.cat(target_mask, dim=0)

    res = {}
    res["num_obs"]  = torch.Tensor(num_obs)
    res["times"]    = context_x
    res["X"]        = values
    res["M"]        = mask
    res["y"]        = torch.zeros([i+1,1])
    res["cov"]      = torch.zeros([i+1,1])
    res["X_val"]    = values_val
    res["M_val"]    = mask_val
    res["times_val"] = target_x

    return res
