import pdb
from time import time
from typing import OrderedDict
import numpy as np
import os
import copy

import torch
import torch.nn as nn
from torch import optim

# from ray.util.check_serialize import inspect_serializability
from mindreadingautobots.models.transformer import TransformerWrapper
from mindreadingautobots.models.rnn import RNNWrapper
from mindreadingautobots.utils.logger import store_results, print_log
from mindreadingautobots.utils.dataloader import Corpus, Sampler
from mindreadingautobots.utils.helper import Voc
from argparse import Namespace

config = Namespace(
    mode='tune',
    debug=False,
    noiseless_validation=True,
    results=True,
    savei=False,
    dataset='sparse_parity_k4_nbits21_n5000_bf0_seed1234',
    itr=False,
    gpu=0,
    seed=1729,
    logging=1,
    ckpt='model',
    model_type='SAN',
    depth=4,
    dropout=0.1,
    cell_type=None,
    emb_size=None,
    hidden_size=None,
    tied=False,
    d_model=32,
    d_ffn=64,
    heads=8,
    pos_encode='learnable',
    mask=False,
    init_range=0.08,
    lr=0.003981,
    decay_patience=3,
    decay_rate=0.2,
    max_grad_norm=0.25,
    batch_size=32,
    epochs=2,
    iters=40000,
    opt='adam',
    project='Bool',
    entity='your_entity',
    hyper_config_path='/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml',
    sensitivity=True,
    run_name='20250305130005',
    model_path='train_results/models/sparse_majority_k5_nbits21_n2000_bf45_seed1234/20250305130005',
    abs_path='/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots',
    data_path='/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/data',
    tune_directory='/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots/tune_results/SAN_sparse_majority_k5_nbits21_n2000_bf45_seed1234/run_2025-03-05-13-00-05-178272',
    patience=300
)

class DummyModel:
    def __init__(self):
        pass

    def predict(self, source, word_lens, config, idx=[2, 3, 13, 16]):
        # Convert to numpy array if not already
        source_np = np.array(source)
        
        # Extract relevant bits from the given indices
        selected_bits = source_np[idx, :]  # Shape (4, 32)
        
        # Compute parity (sum mod 2) along axis 0 (column-wise)
        parity = np.sum(selected_bits, axis=0) % 2  # Shape (32,)
        
        return parity

def compute_sensitivity(model, data_loader, config, device):
	"""
	Compute the sensitivity of the model to input bit flips.
	For each input bit string, flip each bit and see if the output bit changes.
	Sensitivity is the fraction of bit flips that cause the output bit to change.
	"""
	# model.eval()

	sensitivity = 0
	with torch.no_grad():
		for i in range(0, len(data_loader), data_loader.batch_size):
			batch_size = min(data_loader.batch_size, len(data_loader) - i)
			
			source, targets, word_lens = data_loader.get_batch(i)
			# source = source[:batch_size]
			# word_lens = word_lens[:batch_size]

			source, word_lens = source.to(device), word_lens.to(device)

			# Get original output bits
			original_outputs = model.predict(source, word_lens, config)
			# For each input in batch
			# for idx in range(batch_size):

			# input_len = word_lens[idx].item()
			input_len = source.size(0)
			# For each bit in the input string
			for pos in range(input_len):
				modified_source = source.clone()

				# modified_source[pos, idx] = 1 - modified_source[pos, idx]  # flip bit
				modified_source[pos] = 1 - modified_source[pos]  # flip the same position of all the inputs in the batch
				# Get output bit after flipping input bit
				modified_outputs = model.predict(modified_source, word_lens, config)
				

				# How many outputs changed in this batch: 
				num_flipped_output = (modified_outputs != original_outputs).sum().item()
				sensitivity += num_flipped_output

  # Not normalized by the length of the input string

	return sensitivity / len(data_loader)


def test_compute_sensitivity():
  model = DummyModel() 		
  voc= Voc()
	
  data_path = config.data_path
  train_path = os.path.join(data_path, config.dataset, 'train.pkl')
  voc.create_vocab_dict(config, path= train_path, debug = config.debug)
  noiseless_val_path = os.path.join(data_path, config.dataset, 'noiseless_val.pkl')

  noiseless_val_corpus = Corpus(noiseless_val_path, voc, debug = config.debug)
  noiseless_val_loader = Sampler(noiseless_val_corpus, voc, config.batch_size) 
  device = "cpu"
  sensitivity = compute_sensitivity(model, noiseless_val_loader, config, device)
  print(sensitivity)

test_compute_sensitivity()
