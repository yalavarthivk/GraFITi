import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from torch.nn.utils.rnn import pad_sequence
import pdb

plt.switch_backend('agg')


class Batch(NamedTuple):
    r"""A single sample of the data."""
    batch_x: Tensor
    batch_x_mark: Tensor
    batch_x_mask: Tensor
    batch_y: Tensor
    batch_y_mark: Tensor
    batch_y_mask: Tensor 

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

class collate():
    def __init__(self, dataset, cond_time):
        self.dset = dataset
        if self.dset == "physionet2012":
            self.t_max = 48*60
            self.mf = 60
            self.time_cond = cond_time*1
        elif self.dset == "mimiciii":
            self.t_max = 96
            self.mf = 2
            self.time_cond = cond_time*2
        elif self.dset == "mimiciv":
            self.t_max = 48*60
            self.mf = 60
            self.time_cond = cond_time*60
        

    def custom_collate_fn(self, batch: list[Sample]) -> Batch:
        r"""Collate tensors into batch.

        Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
        """
        x_vals: list[Tensor] = []
        y_vals: list[Tensor] = []
        x_mark: list[Tensor] = []
        y_mark: list[Tensor] = []
        x_mask: list[Tensor] = []
        y_mask: list[Tensor] = []
        
        for sample in batch:
            t, x, t_target = sample.inputs
            y = sample.targets
            t *= self.t_max
            t_target *= self.t_max



            tim_cond = torch.arange(0, self.time_cond+1)
            x_cond = torch.zeros([tim_cond.shape[0], x.shape[-1]])*torch.nan
            inds = np.where(np.in1d(tim_cond, t.int()))[0]
            tim_cond[inds] = t.long()
            tim_cond = tim_cond/self.t_max
            x_cond[inds] = x

            tim_forc = torch.arange(self.time_cond+1, self.t_max+1)
            y_forc = torch.zeros([tim_forc.shape[0], y.shape[-1]])*torch.nan
            inds2 = np.where(np.in1d(tim_forc, t_target.int()))[0]
            tim_forc[inds2] = t_target.long()
            tim_forc = tim_forc/self.t_max
            y_forc[inds2] = y

            
            # get whole time interval
            sorted_idx = torch.argsort(tim_cond)

            # create a mask for looking up the target values
            mask_y = y_forc.isfinite()
            mask_x = x_cond.isfinite()

            # nan to zeros
            x = torch.nan_to_num(x_cond)
            y = torch.nan_to_num(y_forc)

            # x_vals.append(torch.cat([x[sorted_idx], mask_x[sorted_idx]], -1))\
            x_vals.append(x[sorted_idx])
            x_mark.append(tim_cond[sorted_idx][:,None])
            x_mask.append(mask_x[sorted_idx])

            y_mark.append(tim_forc[:,None])
            y_vals.append(y)
            y_mask.append(mask_y)

        return Batch(
            batch_x = pad_sequence(x_vals, batch_first=True),
            batch_x_mark = pad_sequence(x_mark, batch_first=True),
            batch_x_mask = pad_sequence(x_mask, batch_first=True),
            batch_y = pad_sequence(y_vals, batch_first=True),
            batch_y_mark = pad_sequence(y_mark, batch_first=True),
            batch_y_mask = pad_sequence(y_mask, batch_first=True),
        )

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj =='type3':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '_checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
