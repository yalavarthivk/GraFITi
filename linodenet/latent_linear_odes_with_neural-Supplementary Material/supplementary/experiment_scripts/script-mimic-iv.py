#!/usr/bin/env python
# coding: utf-8

# # MIMIC-IV

# ## Input Parsing (for command line use)

# In[ ]:


import argparse

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for MIMIC-IV dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=50,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=64,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=128,    type=int,   help="hidden-size")
parser.add_argument("-ls", "--latent-size",  default=256,    type=int,   help="latent-size")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
# fmt: on

try:
    get_ipython().run_line_magic(
        "config", "InteractiveShell.ast_node_interactivity='last_expr_or_assign'"
    )
except NameError:
    ARGS = parser.parse_args()
else:
    ARGS = parser.parse_args("")

print(ARGS)


# ### Load from config file if provided

# In[ ]:


import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)


# ## Global Variables

# In[ ]:


import logging
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
from tqdm.autonotebook import tqdm, trange

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")


# ## Hyperparameter choices

# In[ ]:


if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

hparam_dict = {
    "dataset": (DATASET := "MIMIC-IV"),
    "model": (MODEL_NAME := "LinODEnet"),
    "fold": ARGS.fold,
    "seed": ARGS.seed,
    "max_epochs": ARGS.epochs,
    "batch_size": ARGS.batch_size,
    "hidden_size": ARGS.hidden_size,
    "latent_size": ARGS.latent_size,
    "kernel-initialization": ARGS.kernel_init,
} | OPTIMIZER_CONFIG


# In[ ]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_STR = f"f={ARGS.fold}_bs={ARGS.batch_size}_lr={ARGS.learn_rate}_hs={ARGS.hidden_size}_ls={ARGS.latent_size}"
RUN_ID = ARGS.run_id or datetime.now().isoformat(timespec="seconds")
CFG_ID = 0 if ARGS.config is None else ARGS.config[1]
HOME = Path.cwd()

LOGGING_DIR = HOME / "tensorboard" / DATASET / MODEL_NAME / RUN_ID / CONFIG_STR
CKPOINT_DIR = HOME / "checkpoints" / DATASET / MODEL_NAME / RUN_ID / CONFIG_STR
RESULTS_DIR = HOME / "results" / DATASET / MODEL_NAME / RUN_ID
LOGGING_DIR.mkdir(parents=True, exist_ok=True)
CKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ## Initialize Task

# In[ ]:


from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

TASK = MIMIC_IV_Bilos2021()


# ## Initialize DataLoaders

# In[ ]:


from tsdm.tasks.mimic_iv_bilos2021 import mimic_collate as task_collate_fn

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": task_collate_fn,
}

dloader_config_infer = {
    "batch_size": 256,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": task_collate_fn,
}

TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


# ## Initialize Loss

# In[ ]:


def MSE(y: Tensor, yhat: Tensor) -> Tensor:
    return torch.mean((y - yhat) ** 2)


def MAE(y: Tensor, yhat: Tensor) -> Tensor:
    return torch.mean(torch.abs(y - yhat))


def RMSE(y: Tensor, yhat: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((y - yhat) ** 2))


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


# ## Initialize Model

# In[ ]:


from linodenet.models import LinODEnet, ResNet, embeddings, filters, system

MODEL_CONFIG = {
    "__name__": "LinODEnet",
    "input_size": TASK.dataset.shape[-1],
    "hidden_size": ARGS.hidden_size,
    "latent_size": ARGS.latent_size,
    "Filter": filters.SequentialFilter.HP | {"autoregressive": True},
    "System": system.LinODECell.HP | {"kernel_initialization": ARGS.kernel_init},
    "Encoder": ResNet.HP,
    "Decoder": ResNet.HP,
    "Embedding": embeddings.ConcatEmbedding.HP,
    "Projection": embeddings.ConcatProjection.HP,
}

MODEL = LinODEnet(**MODEL_CONFIG).to(DEVICE)
MODEL = torch.jit.script(MODEL)
torchinfo.summary(MODEL)


# ### Warm-Up

# In[ ]:


def predict_fn(model, batch) -> tuple[Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, _, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    YHAT = model(T, X)
    return Y[MY], YHAT[M]


batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

# Forward
Y, YHAT = predict_fn(MODEL, batch)

# Backward
R = LOSS(Y, YHAT)
assert torch.isfinite(R).item(), "Model Collapsed!"
R.backward()

# Reset
MODEL.zero_grad(set_to_none=True)


# ## Initialize Optimizer

# In[ ]:


from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)


# ## Initialize Logging

# In[ ]:


from torch.utils.tensorboard import SummaryWriter

from tsdm.logutils import StandardLogger

WRITER = SummaryWriter(LOGGING_DIR)
LOGGER = StandardLogger(
    writer=WRITER,
    model=MODEL,
    optimizer=OPTIMIZER,
    metrics=METRICS,
    dataloaders=EVAL_LOADERS,
    hparam_dict=hparam_dict,
    checkpoint_dir=CKPOINT_DIR,
    predict_fn=predict_fn,
    results_dir=RESULTS_DIR,
)
LOGGER.log_epoch_end(0)


# ## Training

# In[ ]:


total_num_batches = 0
for epoch in trange(1, ARGS.epochs, desc="Epoch", position=0):
    for batch in tqdm(
        TRAIN_LOADER, desc="Batch", leave=False, position=1, disable=ARGS.quiet
    ):
        total_num_batches += 1
        MODEL.zero_grad(set_to_none=True)

        # Forward
        Y, YHAT = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT)
        assert torch.isfinite(R).item(), "Model Collapsed!"

        # Backward
        R.backward()
        OPTIMIZER.step()

        # Logging
        LOGGER.log_batch_end(total_num_batches, targets=Y, predics=YHAT)
    LOGGER.log_epoch_end(epoch)

LOGGER.log_history(CFG_ID)
LOGGER.log_hparams(CFG_ID)

