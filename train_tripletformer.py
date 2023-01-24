import argparse
import sys
from random import SystemRandom
# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=30,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=2,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=32,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-nrefp", "--n-ref-points",  default=32,    type=int,   help="number of reference points")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-nl",  "--nlayers", default=2,   type=int,   help="")
parser.add_argument("-eahd",  "--enc-attn-head", default=2,   type=int,   help="")
parser.add_argument("-dahd",  "--dec-attn-head", default=2,   type=int,   help="")
parser.add_argument("-ldim",  "--latent-dim", default=128,   type=int,   help="")
parser.add_argument("-olayers", "--noutlayers", default=2, type=int, help="")
parser.add_argument("-od", "--out-dim", default=128, type=int, help="")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ddim", "--dec-dim", default="128", type=int, help="Decoder dimension")
parser.add_argument("-ft", "--forc_time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond_time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")

# fmt: on

ARGS = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)


# ### Load from config file if provided

# In[52]:


import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)




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

# In[55]:


if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}
# In[57]:



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




# ## Initialize DataLoaders

# In[58]:

from gratif import ushcn_collate

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": ushcn_collate,
}

dloader_config_infer = {
    "batch_size": 64,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": ushcn_collate,
}


TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


# def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
#     mask_sum = torch.sum(mask,1)
#     total_vals = mask_sum.to(torch.bool).sum()
#     mask_sum[mask_sum==0] = 1
#     err = torch.sum(mask*(y-yhat)**2, 1)/mask_sum
#     return err.sum()/total_vals

def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask] )**2)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask *torch.abs( y -yhat), 1 )/(torch.sum(mask ,1))
    return torch.mean(err)

def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask *( y -yhat )**2, 1 )/(torch.sum(mask ,1)))
    return torch.mean(err)


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


# ## Initialize Model

# In[60]:


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[61]:


# from linodenet.models import LinODEnet, ResNet, embeddings, filters, system
# from gratif import GrATiF
from tripletformer import TripletFormer

MODEL_CONFIG = {
    "input_dim" :TASK.dataset.shape[-1],
    "enc_attn_head" :ARGS.enc_attn_head,
    "dec_attn_head" :ARGS.dec_attn_head,
    "latent_dim" : ARGS.latent_dim,
    "n_layers" :ARGS.nlayers,
    "n_ref_points":ARGS.n_ref_points,
    "out_dim":ARGS.out_dim,
    "dec_dim":ARGS.dec_dim,
    "out_layers":ARGS.noutlayers,
    "device": DEVICE
}

MODEL = TripletFormer(**MODEL_CONFIG).to(DEVICE)
# MODEL = torch.jit.script(MODEL)
torchinfo.summary(MODEL)


# ### Warm-Up

# In[62]:

# logging.basicConfig(filename="logs/gratif_f_ " +str(ARGS.fold ) +"_nl_ " +str(MODEL_CONFIG["n_layers"] ) +"_enh_ " +str
#     (MODEL_CONFIG["attn_head"] ) +"_idm_ " +str(MODEL_CONFIG["latent_dim"] ) +".log", encoding='utf-8', level=logging.DEBUG, force = True)
#
# logging.info(str(ARGS))
# logging.info(str(MODEL_CONFIG))


import pdb
def predict_fn(model, batch) -> tuple[Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    output, target_U_, target_mask_ = model(T, X, M, TY, Y, MY)
    return target_U_, output.squeeze(), target_mask_


batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

# Forward
Y, YHAT, MASK = predict_fn(MODEL, batch)
# pdb.set_trace()
# Backward
R = LOSS(Y, YHAT, MASK)
assert torch.isfinite(R).item(), "Model Collapsed!"
# R.backward()

# Reset
MODEL.zero_grad(set_to_none=True)


# ## Initialize Optimizer

# In[63]:


from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min', patience=10, factor=0.5, min_lr=0.00001, verbose=True)

# ## Training
early_stop = 0
# In[ ]:
es = False
best_val_loss = 10e8
total_num_batches = 0
for epoch in range(1, ARGS.epochs +1):
    loss_list = []
    for batch in (TRAIN_LOADER):
        total_num_batches += 1
        OPTIMIZER.zero_grad()
        # print(batch[1].abs().max())

        # Forward
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        # Backward
        R.backward()
        OPTIMIZER.step()

    train_loss = torch.mean(torch.Tensor(loss_list))
    loss_list = []
    count = 0
    # count = count.to(DEVICE)
    with torch.no_grad():
        for batch in (VALID_LOADER):
            total_num_batches += 1
            # Forward
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT, MASK)
            if R.isnan():
                pdb.set_trace()
            loss_list.append([ R *MASK.sum()])
            count += MASK.sum()
    val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE ) /count)
    print(epoch ,"Train: ", train_loss.item(), " VAL: " ,val_loss.item())

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save({    'args': ARGS,
                        'epoch': epoch,
                        'state_dict': MODEL.state_dict(),
                        'optimizer_state_dict': OPTIMIZER.state_dict(),
                        'loss': train_loss,
                        }, 'saved_models/' +ARGS.dataset + '_' + str(experiment_id) + '.h5')
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == 30:
        print("Early stopping because of no improvement in val. metric for 30 epochs")
        es = True
        # test_loss = utils.test_result(net,dim,test_loader,args.sample_type, args.sample_tp,shuffle=False,k_iwae=1,device=device)
        # print('best_val_loss: ', best_val_loss.cpu().detach().numpy(), ' test_loss: ', test_loss.cpu().detach().numpy())
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
    if (epoch == ARGS.epochs) or (es == True):
        # experiment_id = 5806940
        chp = torch.load('saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
        MODEL.load_state_dict(chp['state_dict'])
        loss_list = []
        count = 0
        # count = count.to(DEVICE)
        # pdb.set_trace()
        with torch.no_grad():
            for batch in (TEST_LOADER):

                total_num_batches += 1
                # Forward
                Y, YHAT, MASK = predict_fn(MODEL, batch)
                R = LOSS(Y, YHAT, MASK)
                assert torch.isfinite(R).item(), "Model Collapsed!"
                # loss_list.append([R*Y.shape[0]])
                loss_list.append([ R *MASK.sum()])
                # count += Y.shape[0]
                count += MASK.sum()
        # test_loss = torch.sum(torch.Tensor(loss_list)/count)
        test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE ) /count)
        print("Best_val_loss: " ,best_val_loss.item(), " test_loss : ", test_loss.item())
        # logging.info("BEST VAL:  " +str(val_loss.item() )+ " TEST :  "+ str(test_loss.item()))
        break
    # LOGGER.log_epoch_end(epoch)



# LOGGER.log_history(CFG_ID)
# LOGGER.log_hparams(CFG_ID)
