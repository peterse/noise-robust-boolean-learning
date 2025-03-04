import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch import optim

class TransformerWrapper(nn.Module):
	def __init__(self, config=None, voc=None, device=None, logger=None):
		super(TransformerWrapper, self).__init__()

		self.config = config
		self.device = device
		self.logger = logger
		self.voc = voc
		self.threshold = 0.5

		if self.logger:
			self.logger.debug('Initalizing Model...')
		self._initialize_model()

		if self.logger:
			self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# self.criterion = nn.NLLLoss()
		# Use this to save computation, the model does not compute softmax.
		self.criterion = torch.nn.CrossEntropyLoss()

	def _initialize_model(self):

		# self.config.d_ff = 2*self.config.d_model # uh this attr was spelled wrong when I found it, yikes?
		self.model = TransformerCLF(self.voc.nwords, 2, self.config.d_model,
		self.config.heads, self.config.d_ffn, self.config.depth, 
		self.config.dropout, self.config.pos_encode, mask= self.config.mask ).to(self.device)


	def _initialize_optimizer(self):
		self.params = self.model.parameters()

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		elif self.config.opt =='rmsprop':
			self.optimizer = optim.RMSprop(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=self.config.decay_rate, patience=self.config.decay_patience, verbose=True)
	

	def trainer(self, source, targets, lengths, config, device = None, logger=None):

		self.optimizer.zero_grad()
		output = self.model(source, lengths)
		loss = self.criterion(output, targets)
		loss.backward()

		if self.config.max_grad_norm >0:   
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
		
		self.optimizer.step()
		
		return loss.item()
	

	def evaluator(self, source, targets, lengths, config, device=None, hidden=None):
		
		# if config.model_type == 'RNN':
		# 	output, hidden = self.model(source, hidden, lengths)
		
		output = self.model(source, lengths)
		preds = output.cpu().numpy()
		preds = preds.argmax(axis=1)
		labels= targets.cpu().numpy()
		acc= np.array(preds==labels, np.int32).sum() / len(targets)

		return acc
	
	def predict(self, source, lengths, config, device=None, hidden=None):
		output = self.model(source, lengths)
		preds = output.cpu().numpy()
		preds = preds.argmax(axis=1)
		return preds
		


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
	

    
class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_period = 10000.0, max_len=500):
		super(PositionalEncoding, self).__init__()
		odd_flag=False
		if int(d_model%2) !=0:
			odd_flag=True
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_period) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		if odd_flag:
			pe[:, 1::2] = torch.cos(position * div_term[:-1])
		else:
			pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class CosineNpiPositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(CosineNpiPositionalEncoding, self).__init__()
		odd_flag=False
		if int(d_model%2) !=0:
			odd_flag=True
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.ones(max_len, d_model)
		for i in range(max_len):
			pe[i] = pe[i] * math.cos(i * math.pi)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=900, init_range = 0.1):
		super(LearnablePositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pos_embeds = torch.FloatTensor(max_len, 1, d_model).uniform_(-init_range, init_range)
		pe = nn.Parameter(pos_embeds, requires_grad = True)
		self.pe = pe

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)
	

    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn"

    def __init__(self, self_attn):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        #self.feed_forward = feed_forward

    def forward(self, x, mask):
        return self.self_attn(x, x, x, mask)


class EncoderLayerFFN(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayerFFN, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        # return self.feed_forward(self.self_attn(x, x, x, mask))


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""

	def __init__(self, d_model, dropout=0.1):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return self.dropout(sublayer(self.norm(x))) + x


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores= scores.masked_fill(mask ==0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout= 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model %h ==0

        self.d_k = d_model //h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn= None
        self.dropout= nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):
        
        if mask is not None:
            mask= mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask= mask, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
    
    
            
