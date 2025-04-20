
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention', 'IMAB', 'MAB', 'MAB2']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        # pdb.set_trace()
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10e9)


        attention = F.softmax(scores, dim=-1)

        return attention.matmul(value), attention


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        # pdb.set_trace()
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)

        y = self._reshape_from_batches(y)
        
        y = self.linear_o(y)
        # pdb.set_trace()
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class MAB2(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super(MAB2, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim =n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K = torch.cat(K.split(dim_split, 2), 0)
        V = torch.cat(V.split(dim_split, 2), 0)

        Att_mat = Q_.bmm(K.transpose(1,2))/math.sqrt(self.n_dim)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(mask.repeat(self.num_heads,1,1) == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        Att_mat = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(mask == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class indMAB(nn.Module):
    def __init__(self, induced_dims, value_dims, hidden_dims, num_heads, ln=False):
        super(indMAB, self).__init__()
        self.mab0 = MAB(induced_dims, value_dims, hidden_dims, num_heads, ln=ln)
        self.mab1 = MAB(value_dims, hidden_dims, hidden_dims, num_heads, ln=ln)
        self.head_num = num_heads

    def forward(self, X, Y, att_mask):
        induced_points = X.shape[-2]
        if att_mask is not None:
            mask_r = att_mask.unsqueeze(-2).repeat(self.head_num, induced_points, 1)
            mask_o = att_mask.unsqueeze(-1).repeat(self.head_num, 1, induced_points)
        I = self.mab0(X, Y, mask_r)
        H = self.mab1(Y, I, mask_o)
        return I, H



class IMAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(IMAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.head_num = num_heads
        self.num_inds = num_inds

    def forward(self, X, Y, mask1, mask2):
        
        if mask1 is not None:
            mask_r = mask1.unsqueeze(-2).repeat(self.head_num, self.num_inds, 1)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), Y, mask_r)
        if mask2 is not None:
            mask_o = mask2.unsqueeze(-1).repeat(self.head_num,1, self.num_inds)
        return self.mab1(X, H, mask_o)
