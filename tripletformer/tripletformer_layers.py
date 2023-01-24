import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention, IMAB
import pdb


class Out(nn.Module):
    def __init__(self, dim=128, nkernel=128, nlayers=1, device='cuda'):
        super(Out, self).__init__()
        self.device=device
        self.nlayers=nlayers
        self.dense = nn.ModuleList()
        self.dim = dim
        self.nkernel=nkernel
        for i in range(self.nlayers):
            self.dense.append(nn.Linear(dim, nkernel))
            dim = nkernel
        self.dense.append(nn.Linear(dim,1))
        self.relu = nn.ReLU()
    def forward(self, hid):
        for i in range(self.nlayers):
            hid = self.relu(self.dense[i](hid))
        outs = self.dense[-1](hid)
        return outs[:,:,0]


class Encoder(nn.Module):
    def __init__(self, dim=41, latent_dim=128, n_ref_points=128, n_layers=3, enc_num_heads=4, device="cuda"):
        super(Encoder, self).__init__()
        self.dim = dim
        self.nheads = enc_num_heads
        self.iFF = nn.Linear(self.dim+2, latent_dim)
        self.n_layers = n_layers
        self.SAB = nn.ModuleList()
        for i in range(self.n_layers):
            self.SAB.append(IMAB(latent_dim, latent_dim, self.nheads, n_ref_points))
        
        self.relu = nn.ReLU()

    def forward(self, time_x, value_x, mask_x):

        seq_len = time_x.size(-1)
        ndims = value_x.shape[-1]

        T = time_x[:,:,None].repeat(1,1,mask_x.shape[-1])
        C = torch.cumsum(torch.ones_like(value_x).to(torch.int64), -1) - 1
        mk_bool = mask_x.to(torch.bool)
        full_len = torch.max(mask_x.sum((1,2))).int()
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)
        T_ = torch.stack([pad(r[m]) for r, m in zip(T, mk_bool)]).contiguous()
        U_ = torch.stack([pad(r[m]) for r, m in zip(value_x, mk_bool)]).contiguous()
        C_ = torch.stack([pad(r[m]) for r, m in zip(C, mk_bool)]).contiguous()
        mk_ = torch.stack([pad(r[m]) for r, m in zip(mask_x, mk_bool)]).contiguous()
        
        C_ = torch.nn.functional.one_hot(C_.to(torch.int64), num_classes=self.dim)

        X = torch.cat([C_, T_.unsqueeze(-1), U_.unsqueeze(-1)], -1).contiguous()
        X = X*mk_[:,:,None].repeat(1,1,X.size(-1)).contiguous()
        mk_ = mk_.unsqueeze(-1).contiguous()

        # iFF layer
        Y_e = self.relu(self.iFF(X))
        Y_e = Y_e*mk_.repeat(1,1, Y_e.shape[-1])


        attn_mask = mk_[:,:,0]
        attn_input = Y_e
        for i in range(self.n_layers):
            Z_e = self.SAB[i](attn_input, attn_input, mask1=attn_mask, mask2=attn_mask)[0]
            Z_e = Z_e*mk_.repeat([1, 1, Z_e.shape[-1]])
            Z_e = Z_e + attn_input
            attn_input = Z_e

        return Z_e, mk_



class Decoder(nn.Module):
    def __init__(self, dim=41, latent_dim=128, dec_dim=128, dec_num_heads = 2, device="cuda"):
        super(Decoder, self).__init__()
        self.dim = dim
        self.dec_dim = dec_dim
        self.latent_dim = latent_dim
        self.device=device
        self.oFF = nn.Linear(self.dim+1, self.dec_dim)
        self.input_dense_key = nn.Linear(self.latent_dim,self.dec_dim)
        self.nheads = dec_num_heads
        self.CAB = MultiHeadAttention(dec_dim, self.nheads)
        self.relu = nn.ReLU()
        self.res_con = nn.Linear(dec_dim, dec_dim)

    def forward(self, enc_val, enc_mask, y_time, y_vals, y_mask):

        # mk_d = y_mask[:,:,1:]
        T = y_time[:, :, None].repeat(1, 1, y_mask.shape[-1])
        C = torch.cumsum(torch.ones_like(y_vals).to(torch.int64), -1) - 1
        mk_bool = y_mask.to(torch.bool)
        full_len = torch.max(y_mask.sum((1, 2))).int()
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)

        T_ = torch.stack([pad(r[m]) for r, m in zip(T, mk_bool)]).contiguous()
        U_ = torch.stack([pad(r[m]) for r, m in zip(y_vals, mk_bool)]).contiguous()
        C_ = torch.stack([pad(r[m]) for r, m in zip(C, mk_bool)]).contiguous()
        mk_d = torch.stack([pad(r[m]) for r, m in zip(y_mask, mk_bool)]).contiguous()

        C_ = torch.nn.functional.one_hot(C_.to(torch.int64), num_classes=self.dim)

        X = torch.cat([C_, T_.unsqueeze(-1)], -1).contiguous()
        X = X * mk_d[:, :, None].repeat(1, 1, X.size(-1)).contiguous()
        mk_d = mk_d.unsqueeze(-1).contiguous()

        # oFF layer
        Y_d = self.relu(self.oFF(X))
        Y_d = Y_d * mk_d.repeat(1, 1, Y_d.shape[-1])
        # pdb.set_trace()
        attn_mask = mk_d.matmul(enc_mask.transpose(-2,-1).contiguous())
        Z_e_ = self.relu(self.input_dense_key(enc_val))

        q_len = Y_d.size(1)

        attn_ = []

        for i in range(q_len//1000+1):
            q = Y_d[:,i*1000:(i+1)*1000,:]
            attn_mask_i = attn_mask[:,i*1000:(i+1)*1000,:]
            attn_.append(self.CAB(q,Z_e_,Z_e_, mask = attn_mask_i)[0])
        Z_d = torch.cat(attn_, 1)

        Z_d = Z_d*mk_d.repeat(1,1,Z_d.shape[-1])
        Z_d_ = self.relu(Z_d + Y_d)

        Z_d__ = self.res_con(Z_d_)
        Z_d__ *= mk_d.repeat(1,1,Z_d__.shape[-1])
        Z_d__ += Z_d_
        Z_d__ = self.relu(Z_d__)
        return Z_d__, U_, mk_d[:,:,0]