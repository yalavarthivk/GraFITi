# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.optim as optim
import pdb
from random import SystemRandom
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--only-periodic', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ft", "--forc_time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond_time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")

args = parser.parse_args()

if args.dataset=="ushcn":
    from tsdm.tasks import USHCN_DeBrouwer2019
    TASK = USHCN_DeBrouwer2019(normalize_time=True, condition_time=args.cond_time, forecast_horizon = args.forc_time, num_folds=args.nfolds)
elif args.dataset=="mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
    TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=args.cond_time, forecast_horizon = args.forc_time, num_folds=args.nfolds)
elif args.dataset=="mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
    TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=args.cond_time, forecast_horizon = args.forc_time, num_folds=args.nfolds)
elif args.dataset=='physionet2012':
    from tsdm.tasks.physionet2012 import Physionet2012
    TASK = Physionet2012(normalize_time=True, condition_time=args.cond_time, forecast_horizon = args.forc_time, num_folds=args.nfolds)




if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'toy':
        data_obj = utils.kernel_smoother_data_gen(args, alpha=100., seed=0)
    elif args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)


    from utils import mTAN_collate as task_collate_fn

    dloader_config_train = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": task_collate_fn,
    }

    dloader_config_infer = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": 0,
        "collate_fn": task_collate_fn,
    }

    train_loader = TASK.get_dataloader((args.fold, "train"), **dloader_config_train)
    # INFER_LOADER = TASK.get_dataloader((args.fold, "train"), **dloader_config_infer)
    val_loader = TASK.get_dataloader((args.fold, "valid"), **dloader_config_infer)
    test_loader = TASK.get_dataloader((args.fold, "test"), **dloader_config_infer)
    # EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}

    # train_loader = data_obj["train_dataloader"]
    # test_loader = data_obj["test_dataloader"]
    dim = TASK.dataset.shape[-1]
    # pdb.set_trace()
    # model
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)
   
        
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)


    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec))
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))
    best_val = 10e8
    es = 0
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch in train_loader:
            subsampled_tp, subsampled_data, subsampled_mask, observed_tp, observed_data, observed_mask = (tensor.to(device) for tensor in train_batch)
            # pdb.set_trace()
            # train_batch = train_batch.to(device)
            batch_len = subsampled_data.shape[0]
            # observed_data = train_batch[:, :, :dim]
            # observed_mask = train_batch[:, :, dim:2 * dim]
            # observed_tp = train_batch[:, :, -1]
            # if args.sample_tp and args.sample_tp < 1:
            #     subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
            #         observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            # else:
            #     subsampled_data, subsampled_tp, subsampled_mask = \
            #         observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            # epsilon = torch.randn(qz0_mean.size()).to(device)
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            # compute loss
            rec_mask = observed_mask + subsampled_mask
            rec_val = observed_data + subsampled_data
            train_batch = torch.cat([rec_val, rec_mask, observed_tp[:,:,None]], -1)
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len

        
        val_error = utils.evaluate(dim, rec, dec, val_loader, args, 1)
        if val_error < best_val:
            best_val = val_error
            es = 0
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
            }, 'saved_models/'+ args.dataset + '_' + args.enc + '_' + args.dec + '_' +
                str(experiment_id) + '.h5')
        else:
            es+=1
        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, train mse: {:.6f}, val mse: {:.6f}'
            .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n, val_error))
        
        if es == 30:
            print('Early stopping because validation loss did not improve for 30 epochs')
            checkpoint = torch.load('saved_models/'+args.dataset + '_' + args.enc + '_' + args.dec + '_' +
                str(experiment_id) + '.h5')
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            test_loss = utils.evaluate(dim, rec, dec, test_loader, args, 1)
            print("Best Val Loss: ", best_val.item(), ", TEST: ",test_loss.item())
            break
        elif itr == args.niters:
            print('Completed training')
            checkpoint = torch.load('saved_models/'+args.dataset + '_' + args.enc + '_' + args.dec + '_' +
                str(experiment_id) + '.h5')
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            test_loss = utils.evaluate(dim, rec, dec, test_loader, args, 1)
            print("Best Val Loss: ", best_val.item(), ", TEST: ",test_loss.item())
            # test_loss = utils.evaluate(dim, rec, dec, test_loader, args, 1)