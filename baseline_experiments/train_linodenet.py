import argparse
import sys
from random import SystemRandom
from IPython.core.display import HTML

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for MIMIC dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=50,    type=int,   help="maximum epochs")
parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=128,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=64,    type=int,   help="hidden-size")
parser.add_argument("-ls", "--latent-size",  default=64,    type=int,   help="latent-size")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ft", "--forc_time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond_time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")

# fmt: on

ARGS = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)
print(ARGS)



import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)

import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchinfo
from torch import Tensor, jit
from tqdm.autonotebook import tqdm, trange

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")


if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)


OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARGS.DEVICE = DEVICE

if ARGS.dataset=="ushcn":
    from tsdm.tasks import USHCN_DeBrouwer2019
    TASK = USHCN_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=="mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
    TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=="mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
    TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=='physionet2012':
    from tsdm.tasks.physionet2012 import Physionet2012
    TASK = Physionet2012(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)


from linodenet.utils.data_utils import linodenet_collate as task_collate_fn

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

from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)


def eval_model(MODEL, DATA_LOADER, ARGS):
    loss_list = []
    MODEL.eval()
    with torch.no_grad():
        count = 0
        for i, batch in enumerate(DATA_LOADER):
            Y, YHAT = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT)*len(Y)
            count += len(Y)
            loss_list.append([R])
    val_loss = torch.sum(torch.Tensor(loss_list).to(ARGS.DEVICE)/count)
    return val_loss



total_num_batches = 0
print("Start Training")
print(MODEL)
early_stop = 0
best_val_loss = 1e8

for epoch in range(ARGS.epochs):
    loss_list = []
    for i, batch in enumerate(TRAIN_LOADER):
        MODEL.zero_grad(set_to_none=True)

        # Forward
        Y, YHAT = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        # Backward
        R.backward()
        OPTIMIZER.step()
    train_loss = torch.mean(torch.Tensor(loss_list))
    count = 0
    
    # After each epoch compute validation error
    val_loss = eval_model(MODEL, VALID_LOADER, ARGS)
    print(epoch," Train: ", train_loss.item(), " VAL: ",val_loss.item())

    # if current val_loss is less than the best val_loss save the parameters
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # best_model = deepcopy(model.state_dict())
        torch.save({    'ARGS': ARGS,
                        'epoch': epoch,
                        'state_dict': MODEL.state_dict(),
                        'optimizer_state_dict': OPTIMIZER.state_dict(),
                        'loss': train_loss,
                    }, 'saved_models/'+ARGS.dataset + '_' + str(experiment_id) + '.h5')
        early_stop = 0
    else:
        early_stop += 1
    # Compute test_loss if all the epochs or completed or early stop if val_loss did not improve for # many appochs
    if (early_stop == ARGS.patience) or (epoch == ARGS.epochs-1):
        if early_stop == ARGS.patience:
            print("Early stopping because of no improvement in val. metric for 30 epochs")
        else:
            print("Exhausted all the epochs")
        chp = torch.load('saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
        MODEL.load_state_dict(chp['state_dict'])
        test_loss = eval_model(MODEL, TEST_LOADER, ARGS)
        print("Best Val Loss: ", best_val_loss.item(), ", TEST: ",test_loss.item())
        break

    