import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gratif.attention import indMAB
from gratif.attention import MAB2
import pdb


class bipartitegraph_encoder(nn.Module):
	def __init__(self, dim = 41, tim_dims=64, nkernel = 128, n_induced_points=32, n_layers=3, attn_head = 4, device="cuda"):
		super(bipartitegraph_encoder, self).__init__()
		self.dim = dim
		self.nheads = attn_head
		self.n_induced_points = n_induced_points
		self.nkernel = nkernel
		self.time_init = nn.Linear(1, tim_dims)
		self.n_layers = n_layers
		self.attn_blocks = nn.ModuleList()
		self.device = device
		self.out_layer = nn.Linear(nkernel, 1)
		induced_dims = self.n_induced_points
		value_dims = 1+1+tim_dims+dim #valuedims(1), targetindicator(1), timeembed(tim_dims), channelembed(#channels)

		for i in range(self.n_layers):
			self.attn_blocks.append(indMAB(induced_dims, value_dims, nkernel, self.nheads))
			induced_dims = nkernel
			value_dims = nkernel
		self.relu = nn.ReLU()
		temp = torch.arange(self.n_induced_points, dtype=torch.int64)[None,:].to(self.device)
		self.induced_points = torch.nn.functional.one_hot(temp, num_classes=self.n_induced_points).to(torch.float32)

	def forward(self, context_x, value, mask, target_value, target_mask):
		bsize = context_x.shape[0]
		# seq_len = context_x.size(-1) #T, sequence length
		ndims = value.shape[-1] # C, number of channels
		T = context_x[:,:,None].repeat(1,1,ndims) # BxTxC, observed time points in the input
		C_inds = torch.cumsum(torch.ones_like(value).to(torch.int64).to(self.device), -1) - 1 #BxTxC init for channel indices
		mk_bool = mask.to(torch.bool) # BxTxC boolean mask for the input
		full_len = torch.max(mask.sum((1,2))).to(torch.int64) # flattened TxC max length possible
		pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0) # defined function for padding the sequences

		# flattening to 2D, removed masked values and flattened for each series and padded the smaller series to the max possible length
		T_ = torch.stack([pad(r[m]) for r, m in zip(T, mk_bool)]).contiguous() #BxTxC (Times) -> Bxfull_len , flattened time
		U_ = torch.stack([pad(r[m]) for r, m in zip(value, mk_bool)]).contiguous() #BxTxC (values) -> Bxfull_len, flattened values
		target_U_ = torch.stack([pad(r[m]) for r, m in zip(target_value, mk_bool)]).contiguous() #BxK_, flattened target values
		target_mask_ = torch.stack([pad(r[m]) for r, m in zip(target_mask, mk_bool)]).contiguous() #BxK_, flattened target mask
		C_inds_ = torch.stack([pad(r[m]) for r, m in zip(C_inds, mk_bool)]).contiguous() #BxK_, flattened channel indices
		mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous() #BxK_, flattened mask


		C_ = torch.nn.functional.one_hot(C_inds_.to(torch.int64), num_classes=ndims).to(torch.float32) #BxCxC #channel one hot encoding
		U_indicator = target_mask_	# indicator for target values, 0 for the observed and 1 for the forecast
		T_emb = torch.sin(self.time_init(T_[:,:,None])) # learned time embedding
		U_ = torch.cat([T_emb, C_, U_[:,:,None], U_indicator[:,:,None]], -1) #BxK_max x 3 , OHE of channels, values and indicators
		
		#creating induced input at the beginning of the forward propagation		
		att_input = U_
		induced_input = self.induced_points.repeat(bsize, 1, 1)
		att_mask = mk_ # atttention mask is the input mask
		for i in range(self.n_layers):
			induced_input, att_input = self.attn_blocks[i](induced_input, att_input, att_mask) #performing induced multihead attention
			att_input *= mk_[:,:,None].repeat(1,1,self.nkernel)	# multiplying with the mask
		output = self.out_layer(att_input) # passing attention outputs to the linear layer
		return output.squeeze(-1), target_U_, target_mask_


class fullgraph_encoder(nn.Module):
	def __init__(self, dim = 41, tim_dims=64, nkernel = 128, n_layers=3, attn_head = 4, device="cuda"):
		super(fullgraph_encoder, self).__init__()
		# self.dim = dim+2
		self.nheads = attn_head
		self.nkernel = nkernel
		self.time_init = nn.Linear(1, tim_dims)
		self.n_layers = n_layers
		self.attn_blocks = nn.ModuleList()
		self.device = device
		self.out_layer = nn.Linear(nkernel, 1)
		q_dims = 1+1+tim_dims+dim #valuedims(1), targetindicator(1), timeembed(tim_dims), channelembed(#channels)
		for i in range(self.n_layers):
			self.attn_blocks.append(MAB2(q_dims, q_dims, q_dims, nkernel, self.nheads))
			q_dims = nkernel
		self.relu = nn.ReLU()
	def forward(self, context_x, value, mask, target_value, target_mask):
		# seq_len = context_x.size(-1) #T, sequence length
		ndims = value.shape[-1] # C, number of channels
		T = context_x[:,:,None].repeat(1,1,ndims) # BxTxC, observed time points in the input
		C_inds = torch.cumsum(torch.ones_like(value).to(torch.int64).to(self.device), -1) - 1 #BxTxC init for channel indices
		mk_bool = mask.to(torch.bool) # BxTxC boolean mask for the input
		full_len = torch.max(mask.sum((1,2))).to(torch.int64) # flattened TxC max length possible
		pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0) # defined function for padding the sequences

		# flattening to 2D, removed masked values and flattened for each series and padded the smaller series to the max possible length
		T_ = torch.stack([pad(r[m]) for r, m in zip(T, mk_bool)]).contiguous() #BxTxC (Times) -> Bxfull_len , flattened time
		U_ = torch.stack([pad(r[m]) for r, m in zip(value, mk_bool)]).contiguous() #BxTxC (values) -> Bxfull_len, flattened values
		target_U_ = torch.stack([pad(r[m]) for r, m in zip(target_value, mk_bool)]).contiguous() #BxK_, flattened target values
		target_mask_ = torch.stack([pad(r[m]) for r, m in zip(target_mask, mk_bool)]).contiguous() #BxK_, flattened target mask
		C_inds_ = torch.stack([pad(r[m]) for r, m in zip(C_inds, mk_bool)]).contiguous() #BxK_, flattened channel indices
		mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous() #BxK_, flattened mask


		C_ = torch.nn.functional.one_hot(C_inds_.to(torch.int64), num_classes=ndims).to(torch.float32) #BxCxC #channel one hot encoding
		U_indicator = target_mask_	# indicator for target values, 0 for the observed and 1 for the forecast
		T_emb = torch.sin(self.time_init(T_[:,:,None])) # learned time embedding
		U_ = torch.cat([T_emb, C_, U_[:,:,None], U_indicator[:,:,None]], -1) #BxK_max x 3 , OHE of channels, values and indicators
		att_input = U_
		att_mask = torch.matmul(mk_[:,:,None], mk_[:,None,:]) # creating attention mask
		for i in range(self.n_layers):
			att_input = self.attn_blocks[i](att_input, att_input, att_mask) #performing induced multihead attention
			att_input *= mk_[:,:,None].repeat(1,1,self.nkernel)	# multiplying with the mask
		output = self.out_layer(att_input) # passing attention outputs to the linear layer
		return output.squeeze(-1), target_U_, target_mask_

class Encoder(nn.Module):
	def __init__(self, dim = 41, nkernel = 128, n_layers=3, attn_head = 4, device="cuda"):
		super(Encoder, self).__init__()
		self.dim = dim+2
		self.nheads = attn_head
		self.nkernel = nkernel
		self.edge_init = nn.Linear(2, nkernel)
		self.chan_init = nn.Linear(dim, nkernel)
		self.time_init = nn.Linear(1, nkernel)
		self.n_layers = n_layers
		self.channel_time_attn = nn.ModuleList()
		self.time_channel_attn = nn.ModuleList()
		self.edge_nn = nn.ModuleList()
		self.channel_attn = nn.ModuleList()
		self.device = device
		self.output = nn.Linear(3*nkernel, 1)
		for i in range(self.n_layers):
			self.channel_time_attn.append(MAB2(nkernel, 2*nkernel, 2*nkernel, nkernel, self.nheads))
			self.time_channel_attn.append(MAB2(nkernel, 2*nkernel, 2*nkernel, nkernel, self.nheads))
			self.edge_nn.append(nn.Linear(3*nkernel, nkernel))
			self.channel_attn.append(MAB2(nkernel, nkernel, nkernel, nkernel, self.nheads))
		self.relu = nn.ReLU()

	def gather(self, x, inds):
		# inds =  # keep repeating until the embedding len as a new dim
		return x.gather(1, inds[:,:,None].repeat(1,1,x.shape[-1]))

	def forward(self, context_x, value, mask, target_value, target_mask):
		
		seq_len = context_x.size(-1) #T
		ndims = value.shape[-1] # C
		T = context_x[:,:,None] # BxTx1
		C = torch.ones([context_x.shape[0], ndims]).cumsum(1).to(self.device) - 1 #BxC intialization for one hot encoding channels
		T_inds = torch.cumsum(torch.ones_like(value).to(torch.int64), 1) - 1 #BxTxC init for time indices
		C_inds = torch.cumsum(torch.ones_like(value).to(torch.int64), -1) - 1 #BxTxC init for channel indices
		mk_bool = mask.to(torch.bool) # BxTxC
		full_len = torch.max(mask.sum((1,2))).to(torch.int64) # flattened TxC max length possible
		pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0) 

		# flattening to 2D
		T_inds_ = torch.stack([pad(r[m]) for r, m in zip(T_inds, mk_bool)]).contiguous() #BxTxC -> Bxfull_len 
		U_ = torch.stack([pad(r[m]) for r, m in zip(value, mk_bool)]).contiguous() #BxTxC (values) -> Bxfull_len 
		target_U_ = torch.stack([pad(r[m]) for r, m in zip(target_value, mk_bool)]).contiguous() #BxK_
		target_mask_ = torch.stack([pad(r[m]) for r, m in zip(target_mask, mk_bool)]).contiguous() #BxK_
		C_inds_ = torch.stack([pad(r[m]) for r, m in zip(C_inds, mk_bool)]).contiguous() #BxK_
		mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous() #BxK_

		obs_len = full_len

		C_ = torch.nn.functional.one_hot(C.to(torch.int64), num_classes=ndims).to(torch.float32) #BxCxC #channel one hot encoding
		U_indicator = 1-mk_+target_mask_
		U_ = torch.cat([U_[:,:,None], U_indicator[:,:,None]], -1) #BxK_max x 2 #todo: correct

		
		# creating Channel mask and Time mask
		C_mask = C[:,:,None].repeat(1,1,obs_len)
		temp_c_inds = C_inds_[:,None,:].repeat(1,ndims,1)
		C_mask = (C_mask == temp_c_inds).to(torch.float32) #BxCxK_
		C_mask = C_mask*mk_[:,None,:].repeat(1,C_mask.shape[1],1)

		T_mask = T_inds_[:,None,:].repeat(1,T.shape[1],1)
		temp_T_inds = torch.ones_like(T[:,:,0]).cumsum(1)[:,:,None].repeat(1,1,C_inds_.shape[1]) -1
		T_mask = (T_mask == temp_T_inds).to(torch.float32) #BxTxK_
		T_mask = T_mask*mk_[:,None,:].repeat(1,T_mask.shape[1],1)

		U_ = self.relu(self.edge_init(U_)) * mk_[:,:,None].repeat(1,1,self.nkernel) # 
		T_ = torch.sin(self.time_init(T)) # learned time embedding
		C_ = self.relu(self.chan_init(C_)) # embedding on one-hot encoded channel

		del temp_T_inds
		del temp_c_inds
		
		
		for i in range(self.n_layers):

			# channels as queries
			q_c = C_
			k_t = self.gather(T_, T_inds_) # BxK_max x embd_len
			k = torch.cat([k_t, U_], -1) # BxK_max x 2 * embd_len

			C__ = self.channel_time_attn[i](q_c, k, C_mask) # attn (channel_embd, concat(time, values)) along with the mask

			# times as queries
			q_t = T_
			k_c = self.gather(C_, C_inds_)
			k = torch.cat([k_c, U_], -1)
			T__ = self.time_channel_attn[i](q_t, k, T_mask)

			# updating edge weights
			U_ = self.relu(U_ + self.edge_nn[i](torch.cat([U_, k_t, k_c], -1))) * mk_[:,:,None].repeat(1,1,self.nkernel)

			# updating only channel nodes

			C_ = self.channel_attn[i](C__, C__)
			T_ = T__

		k_t = self.gather(T_, T_inds_)
		k_c = self.gather(C_, C_inds_)
		output = self.output(torch.cat([U_, k_t, k_c], -1))
		return output, target_U_, target_mask_
