import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
# from transformers import TransfoXLModel, TransfoXLConfig
from mindreadingautobots.transformer_pipeline.components.attention_utils import MultiHeadedAttention
from mindreadingautobots.transformer_pipeline.components.transformer_encoder import Encoder, EncoderLayer, EncoderLayerFFN
from mindreadingautobots.transformer_pipeline.components.positional_encodings import 	PositionalEncoding, CosineNpiPositionalEncoding, LearnablePositionalEncoding


class TransformerCLF(nn.Module):
	def __init__(self, ntoken, noutputs, d_model, nhead, d_ffn, nlayers, dropout=0.25, pos_encode_type ='absolute', bias=True, mask=False):
		super(TransformerCLF, self).__init__()
		self.model_type = 'SAN'
		if pos_encode_type == 'absolute':
			self.pos_encoder = PositionalEncoding(d_model, dropout, 10000.0)
		elif pos_encode_type == 'cosine_npi':
			self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
		elif pos_encode_type == 'learnable':
			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout, 400)
		
		self.pos_encode = True
		self.pos_mask = mask
		self.d_model = d_model

		self.encoder= nn.Embedding(ntoken, d_model)

		self_attn = MultiHeadedAttention(nhead, d_model, dropout)

		feedforward= nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model) )
		encoder_layers = EncoderLayerFFN(d_model, self_attn, feedforward, dropout)

		self.transformer_encoder=  Encoder(encoder_layers, nlayers)

		self.decoder= nn.Linear(d_model, noutputs, bias=bias)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.LogSoftmax(dim=1)

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def _generate_square_subsequent_mask(self, size):
		"Mask out subsequent positions."
		attn_shape = (1, size, size)
		subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
		return torch.from_numpy(subsequent_mask) == 0

	def forward(self, src, lengths):
		src_mask = None
		if self.pos_mask:
			src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
		
		src = self.encoder(src) * math.sqrt(self.d_model)
		if self.pos_encode:
			src= self.pos_encoder(src)
		
		src = src.transpose(0,1)
		output= self.transformer_encoder(src, src_mask)
		slots = src.size(1)
		out_flat= output.view(-1, self.d_model)
		out_idxs= [(i*slots)+lengths[i].item() -1 for i in range(len(lengths))]
		out_vecs = out_flat[out_idxs]
		out = self.decoder(out_vecs)
		# out = self.softmax(out)

		
		return out