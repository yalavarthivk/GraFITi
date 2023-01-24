import argparse
import os
import torch
import random
import numpy as np
from FEDformer.models import FEDformer, Autoformer, Informer, Transformer, DLinear, NLinear
from FEDformer.utils.tools import EarlyStopping, adjust_learning_rate
from random import SystemRandom
import sys
import torch.nn as nn
from torch import optim
import pdb
import os
import time

# fix_seed = 2021
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--task_id', type=str, default='test', help='task id')
parser.add_argument('--model', type=str, default='Informer',
                    help='model name, options: [FEDformer, Autoformer, Informer, Transformer, DLinear, NLinear]')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')


# data loader
parser.add_argument('--dataset', type=str, default='mimiciii', help='dataset')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='a',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq-len', type=int, default=73, help='input sequence length')
parser.add_argument('--label-len', type=int, default=12, help='start token length')
parser.add_argument('--pred-len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--embed-type', type=int, default=0, help='prediction sequence length')
parser.add_argument("-ft", "--forc-time", default=12, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond-time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-f",  "--fold",         default=2,      type=int,   help="fold number")

# parser.add_argument('--cross_activation', type=str, default='tanh'
# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# model define
parser.add_argument('--enc-in', type=int, default=192, help='encoder input size')
parser.add_argument('--dec-in', type=int, default=96, help='decoder input size')
parser.add_argument('--c-out', type=int, default=96, help='output size')
parser.add_argument('--d-model', type=int, default=512, help='dimension of model')
parser.add_argument('--n-heads', type=int, default=8, help='num of heads')
parser.add_argument('--e-layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d-layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d-ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving-avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output-attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do-predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num-workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use-gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use-multi-gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

args = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
args.experiment_id = experiment_id
# print(ARGS, experiment_id)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.label_len = args.pred_len//2
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print(args, experiment_id)

class Exp_Main():
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        super(Exp_Main, self).__init__()

    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        if self.args.dataset=="ushcn":
            from tsdm.tasks import USHCN_DeBrouwer2019
            TASK = USHCN_DeBrouwer2019(normalize_time=True, condition_time=self.args.cond_time, forecast_horizon = self.args.forc_time, num_folds=self.args.nfolds)
        elif self.args.dataset=="mimiciii":
            from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
            TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=self.args.cond_time, forecast_horizon = self.args.forc_time, num_folds=self.args.nfolds)
        elif self.args.dataset=="mimiciv":
            from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
            TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=self.args.cond_time, forecast_horizon = self.args.forc_time, num_folds=self.args.nfolds)
        elif self.args.dataset=='physionet2012':
            from tsdm.tasks.physionet2012 import Physionet2012
            TASK = Physionet2012(normalize_time=True, condition_time=self.args.cond_time, forecast_horizon = self.args.forc_time, num_folds=self.args.nfolds)
        
        from FEDformer.utils.tools import collate

        collate_fn = collate(self.args.dataset, self.args.cond_time)
        tsdm_collate = collate_fn.custom_collate_fn

        dloader_config_train = {
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 4,
            "collate_fn": tsdm_collate,
        }

        dloader_config_infer = {
            "batch_size": 64,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": tsdm_collate,
        }

        TRAIN_LOADER = TASK.get_dataloader((self.args.fold, "train"), **dloader_config_train)
        VALID_LOADER = TASK.get_dataloader((self.args.fold, "valid"), **dloader_config_infer)
        TEST_LOADER = TASK.get_dataloader((self.args.fold, "test"), **dloader_config_infer)
        pdb.set_trace()
        # EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}

        return TRAIN_LOADER, VALID_LOADER, TEST_LOADER

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def predict_fn(self, batch):
        """Get targets and predictions."""
        batch_x, batch_x_mark, batch_x_mask, batch_y, batch_y_mark,  batch_y_mask = (tensor.to(self.device) for tensor in batch)
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mask = batch_x_mask.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

        batch_y_mark = torch.cat([batch_x_mark[:,-self.args.label_len:,:], batch_y_mark[:, -self.args.pred_len:, :]], dim=1)
        
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if 'Linear' in self.args.model:
                    batch_ip = torch.cat([batch_x, batch_x_mask], 1)
                    outputs = self.model(batch_ip)
                # outputs = self.model(batch_x)
                else:
                    batch_x = torch.cat([batch_x, batch_x_mask], -1)
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if 'Linear' in self.args.model:
                batch_ip = torch.cat([batch_x, batch_x_mask], 1)
                outputs = self.model(batch_ip)
                # outputs = self.model(batch_x)
            else:
                batch_x = torch.cat([batch_x, batch_x_mask], -1)
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        pred = outputs
        true = batch_y
        mask = batch_y_mask.bool()
        return pred, true, mask

    def train(self):
        train_loader, vali_loader, test_loader = self._get_data()
        pdb.set_trace()

        path = 'saved_models/'+str(self.args.experiment_id)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            count = 0
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                outputs, batch_y, mask = self.predict_fn(batch)
                loss = criterion(outputs[mask], batch_y[mask])
                train_loss.append(loss.item())
                count += mask.sum()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            total_loss = []
            count = 0
            with torch.no_grad():
                for batch in (vali_loader):
                    outputs, batch_y, mask = self.predict_fn(batch)
                    loss = criterion(outputs[mask], batch_y[mask])
                    total_loss.append(loss*mask.sum())
                    count += mask.sum()
                vali_loss = torch.sum(torch.Tensor(total_loss))/count

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss.item()))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        total_loss = []
        count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in (test_loader):
                outputs, batch_y, mask = self.predict_fn(batch)
                loss = criterion(outputs[mask], batch_y[mask])
                count += mask.sum()
                total_loss.append(loss*mask.sum())
            test_loss = torch.sum(torch.Tensor(total_loss))/count
        print("Best_val_loss: ",early_stopping.val_loss_min, " test_loss : ", test_loss.item())


exp = Exp_Main(args)
exp.train()
torch.cuda.empty_cache()