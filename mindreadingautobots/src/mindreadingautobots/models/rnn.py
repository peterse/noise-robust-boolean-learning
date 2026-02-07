from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class RNNWrapper(nn.Module):
	# This wrapper is specific to RNNModel!
	def __init__(self, config=None, voc=None, device=None, logger=None):
		super(RNNWrapper, self).__init__()

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

		self.criterion = nn.NLLLoss()
	

	def _initialize_model(self):

		self.model = RNNModel(self.config.cell_type, 2, 2, 
			self.config.emb_size, self.config.hidden_size, self.config.depth, 
			self.config.dropout, self.config.tied).to(self.device)

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
	
	def trainer(self, source, targets, lengths, hidden, config, device = None, logger=None):
		"""trainer for a single batch. Note that `hidden` is re-initialized per batch."""
		self.optimizer.zero_grad()
		output, hidden = self.model(source, hidden, lengths)
		loss = self.criterion(output, targets)
		reg_term = 0
		# Optionalal regularization
		if self.config.lambda_sens > 0:
			n_samp = self.config.reg_samples_per_batch
			n_bits = source.shape[0]  # sequence length
			if self.config.loss == 'sensitivity_reg':
				# estimate the sensitivity, with a squared difference differentiable estimate
				x = torch.randint(0, 2, (n_bits, n_samp), device=self.device).long()
				x_lengths = torch.full((n_samp,), n_bits, dtype=torch.int64, device=self.device)
				idx = torch.randint(0, n_bits, (n_samp,), device=self.device)
				x_flip = x.clone()
				x_flip[idx, torch.arange(n_samp)] = 1 - x_flip[idx, torch.arange(n_samp)]
				# convert the (log) probabilities to probabilities
				
				# Initialize hidden state based on RNN type
				if self.model.rnn_type == 'LSTM':
					temp_hidden = (
						torch.zeros(self.model.nlayers, n_samp, self.model.nhid, device=self.device), 
						torch.zeros(self.model.nlayers, n_samp, self.model.nhid, device=self.device)
					)
				else:
					temp_hidden = torch.zeros(self.model.nlayers, n_samp, self.model.nhid, device=self.device)
				
				output_x, _ = self.model(x, temp_hidden, x_lengths)
				output_flip, _ = self.model(x_flip, temp_hidden, x_lengths)
				p      = torch.exp(output_x)        
				p_flip = torch.exp(output_flip)   
				mse = F.mse_loss(p, p_flip, reduction='mean')
				reg_term = self.config.lambda_sens * mse
			# the goal of the regulaization is to _increase_ sensitivity, so we negate it
		if logger:
			logger.info(f'final batch loss: {str(loss.item())}, final reg term: {str(reg_term.item())}')

		loss -= reg_term
		loss.backward()
 
		if self.config.max_grad_norm >0:   
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
		self.optimizer.step()

		hidden = self.repackage_hidden(hidden)
		if self.config.lambda_sens > 0:
			temp_hidden = self.repackage_hidden(temp_hidden)
		return loss.item()
	
	def evaluator(self, source, targets, lengths, config, device=None, hidden=None):
		
		output, hidden = self.model(source, hidden, lengths)
		preds = output.cpu().numpy()
		preds = preds.argmax(axis=1)
		labels= targets.cpu().numpy()
		acc= np.array(preds==labels, np.int32).sum() / len(targets)
		return acc
	
	def predict(self, source, lengths, config, hidden=None):
		output, hidden = self.model(source, hidden, lengths)
		preds = output.cpu().numpy()
		preds = preds.argmax(axis=1)
		return preds
		
	def repackage_hidden(self, h):
		"""Wraps hidden states in new Tensors, to detach them from their history."""

		if isinstance(h, torch.Tensor):
			return h.detach()
		else:
			return tuple(self.repackage_hidden(v) for v in h)


class RNNModel(nn.Module):
	"""Container module with an embedder, a recurrent module, and a classifier."""

	def __init__(self, rnn_type, ntoken, noutputs, ninp, nhid, nlayers, dropout=0.1, tie_weights=False, is_embedding=True):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		if is_embedding:
			self.encoder = nn.Embedding(ntoken, ninp)
		else:
			ninp = ntoken
			self.encoder = nn.Embedding(ntoken, ninp)
			self.encoder.weight.data =torch.eye(ntoken)
			self.encoder.weight.requires_grad = False
		
		self.bi = bidirectional

		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		
		
		self.decoder = nn.Linear(nhid, noutputs)

		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.LogSoftmax(dim=1)

		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			self.decoder.weight = self.encoder.weight
		

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange = 0.1

		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden, lengths):
		
		lengths = lengths.cpu()
		inp_emb = self.drop(self.encoder(input))


		emb_packed = nn.utils.rnn.pack_padded_sequence(inp_emb, lengths.cpu(), enforce_sorted = False)
		output_packed, hidden = self.rnn(emb_packed, hidden)
		output_padded, _ = nn.utils.rnn.pad_packed_sequence(output_packed)

		output_flat = output_padded.view(-1, self.nhid)
		slots = input.size(1)
		out_idxs= [(lengths[i].item() -1)*slots + i for i in range(len(lengths))]   # Indices of last hidden state
		out_vecs= output_flat[out_idxs]
		output = self.drop(out_vecs)

		decoded = self.decoder(output)
		decoded = self.softmax(decoded)
		return decoded, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
				weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)



