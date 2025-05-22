import argparse
import datetime
import os
import random
import sys
import time
from random import SystemRandom

sys.path.append("../")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import model_selection

import lib.utils as utils
from grafiti.grafiti import *
from lib import evaluation
from lib.parse_datasets import parse_datasets

parser = argparse.ArgumentParser("IMTS Forecasting")

parser.add_argument("--state", type=str, default="def")
parser.add_argument("-n", type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument("--hop", type=int, default=1, help="hops in GNN")
parser.add_argument("--nhead", type=int, default=1, help="heads in Transformer")
parser.add_argument("--tf_layer", type=int, default=1, help="# of layer in Transformer")
parser.add_argument("--nlayer", type=int, default=1, help="# of layer in TSmodel")
parser.add_argument("--epoch", type=int, default=1000, help="training epoches")
parser.add_argument("--patience", type=int, default=10, help="patience for early stop")
parser.add_argument(
    "--history",
    type=int,
    default=24,
    help="number of hours (months for ushcn and ms for activity) as historical window",
)
parser.add_argument("--logmode", type=str, default="a", help="File mode of logging.")

parser.add_argument("--lr", type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument("--w_decay", type=float, default=0.0, help="weight decay.")
parser.add_argument("-b", "--batch_size", type=int, default=32)

parser.add_argument(
    "--save", type=str, default="experiments/", help="Path for save checkpoints"
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="ID of the experiment to load for evaluation. If None, run a new experiment.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument(
    "--dataset",
    type=str,
    default="physionet",
    help="Dataset to load. Available: physionet, mimic, ushcn",
)

# value 0 means using original time granularity, Value 1 means quantization by 1 hour,
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument(
    "--quantization",
    type=float,
    default=0.0,
    help="Quantization on the physionet dataset.",
)
parser.add_argument("--model", type=str, default="GraFITi", help="Model name")
parser.add_argument("-hs", "--hidden-size", default=32, type=int, help="hidden-size")
parser.add_argument(
    "-ki", "--kernel-init", default="skew-symmetric", help="kernel-inititialization"
)
parser.add_argument("--note", default="", type=str, help="Note that can be added")
parser.add_argument("-nl", "--nlayers", default=2, type=int, help="")
parser.add_argument("-ahd", "--attn-head", default=2, type=int, help="")
parser.add_argument("-ldim", "--latent-dim", default=128, type=int, help="")

parser.add_argument("--gpu", type=str, default="0", help="which gpu to use.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)
print(" ".join(sys.argv))

#####################################################################################################

if __name__ == "__main__":
    utils.setup_seed(args.seed)
    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2) :]
    input_command = " ".join(input_command)

    # utils.makedirs("results/")

    ##################################################################
    data_obj = parse_datasets(args, patch_ts=False)
    input_dim = data_obj["input_dim"]

    ### Model setting ###
    args.ndim = input_dim
    MODEL_CONFIG = {
        "input_dim": input_dim,
        "attn_head": args.attn_head,
        "latent_dim": args.latent_dim,
        "n_layers": args.nlayers,
        "device": args.device,
    }
    model = GraFITi(**MODEL_CONFIG).to(args.device)
    ##################################################################

    # # Load checkpoint and evaluate the model
    # if args.load is not None:
    # 	utils.get_ckpt_model(ckpt_path, model, args.device)
    # 	exit()

    ##################################################################

    if args.n < 12000:
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}lr.log".format(
            args.dataset,
            args.model,
            args.lr,
        )

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(
        logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode
    )
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]  # n_sample / batch_size
    print("n_train_batches:", num_batches)

    best_val_mse = np.inf
    test_res = None
    for itr in range(args.epoch):
        st = time.time()

        ### Training ###
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = evaluation.compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            optimizer.step()

        ### Validation ###
        model.eval()
        with torch.no_grad():
            val_res = evaluation.evaluation(
                model, data_obj["val_dataloader"], data_obj["n_val_batches"]
            )

            ### Testing ###
            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation.evaluation(
                    model, data_obj["test_dataloader"], data_obj["n_test_batches"]
                )

            logger.info("- Epoch {:03d}, ExpID {}".format(itr, experimentID))
            logger.info(
                "Train - Loss (one batch): {:.5f}".format(train_res["loss"].item())
            )
            logger.info(
                "Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%".format(
                    val_res["loss"],
                    val_res["mse"],
                    val_res["rmse"],
                    val_res["mae"],
                    val_res["mape"] * 100,
                )
            )
            if test_res != None:
                logger.info(
                    "Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%".format(
                        best_iter,
                        test_res["loss"],
                        test_res["mse"],
                        test_res["rmse"],
                        test_res["mae"],
                        test_res["mape"] * 100,
                    )
                )
            logger.info("Time spent: {:.2f}s".format(time.time() - st))

        if itr - best_iter >= args.patience:
            print("Exp has been early stopped!")
            break

    print(f"VAL-MSE: {best_val_mse}")

    print("TEST-MSE:", test_res["mse"])
    print("TEST-RMSE:", test_res["rmse"])
    print("TEST-MAE:", test_res["mae"])
    print("TEST-MAPE:", test_res["mape"] * 100)
