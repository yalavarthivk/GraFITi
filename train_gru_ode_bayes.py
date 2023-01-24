
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
parser.add_argument("-bs", "--batch-size",   default=128,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-dr",  "--dropout", default=0.2,   type=int,   help="")
parser.add_argument("-hs",  "--hidden-size", default=50,   type=int,   help="")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ft", "--forc_time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond_time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")


# fmt: on

ARGS = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

params_dict = {}
params_dict["hidden_size"] = ARGS.hidden_size
params_dict["p_hidden"] = 25
params_dict["prep_hidden"] = 10
params_dict["logvar"] = True
params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
params_dict["delta_t"]=0.0006
params_dict["T"]=1
params_dict["lambda"] = 0 #Weighting between classification and MSE loss.

params_dict["classification_hidden"] = 2
params_dict["cov_hidden"] = 50
params_dict["weight_decay"] = ARGS.weight_decay
params_dict["dropout_rate"] = ARGS.dropout
params_dict["lr"]=0.001
params_dict["full_gru_ode"] = True
params_dict["no_cov"] = True
params_dict["impute"] = False
params_dict["verbose"] = 0 #from 0 to 3 (highest)

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
import pdb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

import gru_ode_bayes.gru_ode_bayes as gru_ode_bayes
import gru_ode_bayes.torchdiffeq as torchdiffeq
import gru_ode_bayes.gru_ode_bayes.data_utils as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
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
    ARGS.batch_size = 64
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
    TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=="mimiciv":
    ARGS.batch_size = 64
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
    TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
elif ARGS.dataset=='physionet2012':
    from tsdm.tasks.physionet2012 import Physionet2012
    TASK = Physionet2012(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)



# ## Initialize DataLoaders


dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": data_utils.tsdm_collate_val,
}

dloader_config_infer = {
    "batch_size": 64,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": data_utils.tsdm_collate_val,
}

    
TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}



def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask])**2)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask*torch.abs(y-yhat), 1)/(torch.sum(mask,1))
    return torch.mean(err)

def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask*(y-yhat)**2, 1)/(torch.sum(mask,1)))
    return torch.mean(err)


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


b = next(iter(TRAIN_LOADER))

params_dict["input_size"]=b["M"].shape[-1]
params_dict["cov_size"] = b["cov"].shape[-1]


nnfwobj = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                        p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                        logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                        classification_hidden=params_dict["classification_hidden"],
                                        cov_size = params_dict["cov_size"], cov_hidden = params_dict["cov_hidden"],
                                        dropout_rate = params_dict["dropout_rate"],full_gru_ode= params_dict["full_gru_ode"], impute = params_dict["impute"])
nnfwobj.to(DEVICE)

optimizer = torch.optim.Adam(nnfwobj.parameters(), lr=params_dict["lr"], weight_decay= params_dict["weight_decay"])
class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
print("Start Training")
val_metric_prev = -1000

def test_evaluation(model, params_dict, class_criterion, DEVICE, dl_test):
    chp = torch.load('saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
    model.load_state_dict(chp['state_dict'])
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test  = 0
        corr_test = 0
        num_obs = 0
        for i, b in enumerate(dl_test):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(DEVICE)
            M        = b["M"].to(DEVICE)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(DEVICE)
            labels   = b["y"].to(DEVICE)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val     = b["X_val"].to(DEVICE)
                M_val     = b["M_val"].to(DEVICE)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(DEVICE)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _ , _ = model(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size

            try:
                auc_test=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
            except ValueError:
                if params_dict["verbose"]>=3:
                    print("Only one class. AUC is wrong")
                auc_test = 0
                pass

            if params_dict["lambda"]==0:
                t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                m, v = torch.chunk(p_val,2,dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs=1

            total_loss_test += total_loss.cpu().detach().numpy()
            auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /=  num_obs
        auc_total_test /= (i+1)

        return(loss_test, auc_total_test, mse_test)

early_stop = 0
for epoch in range(ARGS.epochs):
    nnfwobj.train()
    total_train_loss = 0
    auc_total_train  = 0
    tot_loglik_loss = 0
    for i, batch in enumerate(TRAIN_LOADER):

        optimizer.zero_grad()
        times    = b["times"]
        time_ptr = b["time_ptr"]
        X        = b["X"].to(DEVICE)
        M        = b["M"].to(DEVICE)
        obs_idx  = b["obs_idx"]
        cov      = b["cov"].to(DEVICE)
        labels = b["y"].to(DEVICE)
        batch_size = labels.size(0)
        h0 = 0# torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(DEVICE)
        hT, loss, class_pred, mse_loss  = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov = cov)

        total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
        total_train_loss += total_loss
        tot_loglik_loss +=mse_loss
        try:
            auc_total_train += roc_auc_score(labels.detach().cpu(),torch.sigmoid(class_pred).detach().cpu())
        except ValueError:
            if params_dict["verbose"]>=3:
                print("Single CLASS ! AUC is erroneous")
            pass
        
        total_loss.backward()
        optimizer.step()


    
    info = { 'training_loss' : total_train_loss.detach().cpu().numpy()/(i+1), 'AUC_training' : auc_total_train/(i+1), "loglik_loss" :tot_loglik_loss.detach().cpu().numpy()}
    # for tag, value in info.items():
    #     logger.scalar_summary(tag, value, epoch)
    print(f"NegLogLik Loss train : {tot_loglik_loss.detach().cpu().numpy()}")

    data_utils.adjust_learning_rate(optimizer,epoch,params_dict["lr"])

    with torch.no_grad():
        nnfwobj.eval()
        total_loss_val = 0
        auc_total_val = 0
        loss_val = 0
        mse_val  = 0
        corr_val = 0
        num_obs = 0
        for i, b in enumerate(VALID_LOADER):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(DEVICE)
            M        = b["M"].to(DEVICE)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(DEVICE)
            labels   = b["y"].to(DEVICE)
            batch_size = labels.size(0)
            # pdb.set_trace()
            if b["X_val"] is not None:
                X_val     = b["X_val"].to(DEVICE)
                M_val     = b["M_val"].to(DEVICE)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(DEVICE)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _  = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size

            try:
                auc_val=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
            except ValueError:
                auc_val = 0.5
                if params_dict["verbose"]>=3:
                    print("Only one class : AUC is erroneous")
                pass

            if params_dict["lambda"]==0:
                t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                # pdb.set_trace()
                p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                m, v = torch.chunk(p_val,2,dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                corr_val_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_val += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_val += mse_loss.cpu().numpy()
                corr_val += corr_val_loss.cpu().numpy()
            else:
                num_obs=1

            total_loss_val += total_loss.cpu().detach().numpy()
            auc_total_val += auc_val

        loss_val /= num_obs
        mse_val /=  num_obs
        info = { 'validation_loss' : total_loss_val/(i+1), 'AUC_validation' : auc_total_val/(i+1),
                 'loglik_loss' : loss_val, 'validation_mse' : mse_val, 'correlation_mean' : np.nanmean(corr_val),
                'correlation_max': np.nanmax(corr_val), 'correlation_min': np.nanmin(corr_val)}
        print(f"Total validation loss at epoch {epoch}: {total_loss_val/(i+1)}")
        print(f"Validation AUC at epoch {epoch}: {auc_total_val/(i+1)}")
        print(f"Validation loss (loglik) at epoch {epoch}: {loss_val:.5f}. MSE : {mse_val:.5f}. Correlation : {np.nanmean(corr_val):.5f}. Num obs = {num_obs}")

        val_metric = mse_val

        if val_metric < val_metric_prev:
            print(f"New highest validation metric reached ! : {val_metric}")
            print("Saving Model")
            torch.save({    'args': ARGS,
                            'epoch': epoch,
                            'state_dict': nnfwobj.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                        }, 'saved_models/'+ARGS.dataset + '_' + str(experiment_id) + '.h5')
            # torch.save(nnfwobj.state_dict(),f"trained_models/{simulation_name}_MAX.pt")
            val_metric_prev = val_metric
            early_stop = 0
        else:
            early_stop += 1
        if (early_stop == 30) or (epoch == ARGS.epochs-1):
            test_loglik, test_auc, test_mse = test_evaluation(nnfwobj, params_dict, class_criterion, DEVICE, TEST_LOADER)
            print(f"Test loglik loss at epoch {epoch} : {test_loglik}")
            print(f"Test AUC loss at epoch {epoch} : {test_auc}")
            print(f"Test MSE loss at epoch{epoch} : {test_mse}")
            print("Best_val_loss: ",val_metric_prev.item(), " test_loss : ", test_mse.item())


