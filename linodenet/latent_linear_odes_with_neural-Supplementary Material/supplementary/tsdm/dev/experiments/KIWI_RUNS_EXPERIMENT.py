#!/usr/bin/env python

# In[1]:


# get_ipython().run_line_magic('config', "InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.")
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
from datetime import datetime

# enable JIT compilation - must be done before loading torch!
os.environ["PYTORCH_JIT"] = "1"


# In[3]:


from pathlib import Path
from time import time

import numpy as np
import pandas
import torch
import torchinfo
from linodenet.models import LinODE, LinODECell, LinODEnet
from linodenet.projections.functional import skew_symmetric, symmetric
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from torch import Tensor, jit, tensor
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

import tsdm
from encoders.functional import time2float
from tsdm.datasets import DATASETS
from tsdm.logutils import (
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)
from tsdm.metrics import LOSSES
from tsdm.tasks import KIWI_RUNS_TASK
from tsdm.utils import grad_norm, multi_norm

# # Initialize Task

# In[4]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NAN = tensor(float("nan"), dtype=DTYPE, device=DEVICE)
BATCH_SIZE = 128
PRD_HORIZON = 30
OBS_HORIZON = 90
HORIZON = SEQLEN = OBS_HORIZON + PRD_HORIZON


# In[5]:


TASK = KIWI_RUNS_TASK(
    forecasting_horizon=PRD_HORIZON,
    observation_horizon=OBS_HORIZON,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=1024,
)

DATASET = TASK.dataset
ts = TASK.timeseries
md = TASK.metadata
NUM_PTS, NUM_DIM = ts.shape


# ## Initialize Loss

# In[6]:


LOSS = TASK.test_metric.to(device=DEVICE)

TASK.loss_weights


# ## Initialize DataLoaders

# In[7]:


TRAINLOADER = TASK.batchloader
EVALLOADERS = TASK.dataloaders


# ## Initialize Model

# In[8]:


MODEL = LinODEnet
model = MODEL(
    input_size=NUM_DIM,
    hidden_size=128,
    embedding_type="concat",
    Encoder_cfg={"nblocks": 10},
    Decoder_cfg={"nblocks": 10},
)
model.to(device=DEVICE, dtype=DTYPE)
torchinfo.summary(model)


# ## Initalize Optimizer

# In[9]:


# OPTIMIZER = AdamW
# # defaults: lr=0.001, betas=(0.9, 0.999), eps=1e-08
# optimizer = OPTIMIZER(model.parameters(), lr=0.01, betas=(0.9, 0.999))


OPTIMIZER = SGD
# defaults: lr=0.001, betas=(0.9, 0.999), eps=1e-08
optimizer = OPTIMIZER(model.parameters(), lr=0.001)


# ## Utility functions

# In[10]:


batch = next(iter(TRAINLOADER[0]))
T, X = batch
targets = X[..., OBS_HORIZON:, TASK.targets.index].clone()
# assert targets.shape == (BATCH_SIZE, PRD_HORIZON, len(TASK.targets))

inputs = X.clone()
inputs[:, OBS_HORIZON:, TASK.targets.index] = NAN
inputs[:, OBS_HORIZON:, TASK.observables.index] = NAN
# assert inputs.shape == (BATCH_SIZE, HORIZON, NUM_DIM)


# In[11]:


targets = X[..., OBS_HORIZON:, TASK.targets.index].clone()
targets.shape


# In[12]:


def prep_batch(batch: tuple[Tensor, Tensor]):
    """Get batch and create model inputs and targets"""
    T, X = batch
    targets = X[..., OBS_HORIZON:, TASK.targets.index].clone()
    # assert targets.shape == (BATCH_SIZE, PRD_HORIZON, len(TASK.targets))

    inputs = X.clone()
    inputs[:, OBS_HORIZON:, TASK.targets.index] = NAN
    inputs[:, OBS_HORIZON:, TASK.observables.index] = NAN
    # assert inputs.shape == (BATCH_SIZE, HORIZON, NUM_DIM)
    return T, inputs, targets


def get_all_preds(model, dataloader):
    Y, Ŷ = [], []
    for batch in (pbar := tqdm(dataloader, leave=False)):
        with torch.no_grad():
            model.zero_grad()
            times, inputs, targets = prep_batch(batch)
            outputs = model(times, inputs)
            predics = outputs[:, OBS_HORIZON:, TASK.targets.index]
            loss = LOSS(targets, predics)
            Y.append(targets)
            Ŷ.append(predics)
        if pbar.n == 5:
            break

    targets, predics = torch.cat(Y, dim=0), torch.cat(Ŷ, dim=0)
    mask = torch.isnan(targets)
    targets[mask] = torch.tensor(0.0)
    predics[mask] = torch.tensor(0.0)
    # scale = 1/torch.mean(mask.to(dtype=torch.float32))
    # targets *= scale
    # predics *= scale
    return targets, predics


# ## Logging Utilities

# In[13]:


def log_all(i, model, writer, optimizer):
    kernel = model.system.kernel.clone().detach().cpu()
    log_kernel_information(i, writer, kernel, log_histograms=True)
    log_optimizer_state(i, writer, optimizer, histograms=True)


# In[14]:


print("WARMUP")
t = torch.randn(NUM_DIM).to(DEVICE)
x = torch.randn(1, NUM_DIM).to(device=DEVICE)
y = model(t, x)
torch.linalg.norm(y).backward()
model.zero_grad()


# In[15]:


RUN_START = tsdm.utils.now()
CHECKPOINTDIR = Path(f"checkpoints/{RUN_START}/")
CHECKPOINTDIR.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(f"runs/{MODEL.__name__}/{DATASET.__name__}{RUN_START}")
metrics = {key: LOSSES[key] for key in ("ND", "NRMSE", "MSE", "MAE")}
# assert any(isinstance(TASK.test_metric, metric) for metric in metrics.values())
metrics = {key: LOSSES[key]() for key in ("ND", "NRMSE", "MSE", "MAE")} | {
    "WRMSE": LOSS
}


# ### Training Start

# In[16]:


i = -1

for epoch in (epochs := trange(100)):
    # log
    with torch.no_grad():
        # log optimizer state first !!!
        # log_optimizer_state(epoch, writer, optimizer, histograms=True)
        log_kernel_information(epoch, writer, model.system.kernel, log_histograms=True)

        for key in ((0, "train"), (0, "test")):
            dataloader = EVALLOADERS[key]
            y, ŷ = get_all_preds(model, dataloader)
            log_metrics(
                epoch, writer, metrics=metrics, targets=y, predics=ŷ, prefix=key[1]
            )

    for batch in (batches := tqdm(TRAINLOADER[0])):
        i += 1
        # Optimization step
        model.zero_grad()
        times, inputs, targets = prep_batch(batch)

        forward_time = time()
        outputs = model(times, inputs)
        forward_time = time() - forward_time

        predics = outputs[:, OBS_HORIZON:, TASK.targets.index]

        # get rid of nan-values in teh targets.
        mask = torch.isnan(targets)
        targets[mask] = torch.tensor(0.0)
        predics[mask] = torch.tensor(0.0)

        # # compensate NaN-Value with upscaling
        # scale = 1/torch.mean(mask.to(dtype=torch.float32))
        # targets *= scale
        # predics *= scale

        loss = LOSS(targets, predics)

        backward_time = time()
        loss.backward()
        backward_time = time() - backward_time

        optimizer.step()

        # batch logging
        logging_time = time()
        with torch.no_grad():
            i += 1
            log_metrics(
                i,
                writer,
                metrics=metrics,
                targets=targets,
                predics=predics,
                prefix="batch",
            )
            log_optimizer_state(i, writer, optimizer, prefix="batch")

            lval = loss.clone().detach().cpu().numpy()
            gval = grad_norm(list(model.parameters())).clone().detach().cpu().numpy()
            if torch.any(torch.isnan(loss)):
                raise RuntimeError("NaN-value encountered!!")
        logging_time = time() - logging_time

        batches.set_postfix(
            loss=f"{lval:.2e}",
            gnorm=f"{gval:.2e}",
            Δt_forward=f"{forward_time:.1f}",
            Δt_backward=f"{backward_time:.1f}",
            Δt_logging=f"{logging_time:.1f}",
        )

    with torch.no_grad():
        # log optimizer state first !!!
        log_optimizer_state(epoch, writer, optimizer, histograms=True)
        log_kernel_information(epoch, writer, model.system.kernel, log_histograms=True)

        for key in ((0, "train"), (0, "test")):
            dataloader = EVALLOADERS[key]
            y, ŷ = get_all_preds(model, dataloader)
            log_metrics(
                epoch, writer, metrics=metrics, targets=y, predics=ŷ, prefix=key[1]
            )

        # Model Checkpoint
        torch.jit.save(model, CHECKPOINTDIR.joinpath(f"{MODEL.__name__}-{epochs.n}"))
        torch.save(
            {
                "optimizer": optimizer,
                "epoch": epoch,
                "batch": i,
            },
            CHECKPOINTDIR.joinpath(f"{OPTIMIZER.__name__}-{epochs.n}"),
        )


# In[ ]:
