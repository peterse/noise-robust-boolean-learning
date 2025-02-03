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

def build_model(config, voc, device, logger):
	if config.model_type == 'RNN':
		logger.info('Building RNN Model')
		model = RNNWrapper(config, voc, device, logger)
	elif config.model_type == 'SAN':
		logger.info('Building Transformer Model')
		model = TransformerWrapper(config, voc, device, logger)
	model = model.to(device)
	return model


def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	if config.mode == 'train' or config.mode == 'tune':
		logger.debug('Loading Training Data...')

		'''Create Vocab'''
		data_path = config.data_path
		train_path = os.path.join(data_path, config.dataset, 'train.pkl')
		val_path = os.path.join(data_path, config.dataset, 'val.pkl')
		noiseless_val_path = os.path.join(data_path, config.dataset, 'noiseless_val.pkl')
		# test_path = os.path.join(data_path, config.dataset, 'test.tsv')
		voc= Voc()
		voc.create_vocab_dict(config, path= train_path, debug = config.debug)
		# voc.create_vocab_dict(config, path= val_path, debug = config.debug)
		# voc.create_vocab_dict(config, path= test_path, debug = config.debug)

		'''Load Datasets'''
		train_corpus = Corpus(train_path, voc, debug = config.debug)
		train_loader = Sampler(train_corpus, voc, config.batch_size)

		val_corpus = Corpus(val_path, voc, debug = config.debug)		
		val_loader = Sampler(val_corpus, voc, config.batch_size)

		noiseless_val_corpus = Corpus(noiseless_val_path, voc, debug = config.debug)
		noiseless_val_loader = Sampler(noiseless_val_corpus, voc, config.batch_size)

		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(len(train_corpus.data), len(val_corpus.data))
		logger.info(msg)
		
		return voc, train_loader, val_loader, noiseless_val_loader
	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))
	

def train_model(model, train_loader, val_loader, noiseless_val_loader, voc, device, 
				config, logger, epoch_offset=0, manager=None):
	"""Train a model on the given dataset and validate it on the validation set.
	
	Returns:
		best_results (dict): Dictionary of key metrics for the best epoch (by validation acc).
	"""

	if manager is not None:
		logger = manager.logger
		log_print = manager.log_print
	else:
		log_print = logger.info

	max_val_acc = 0
	best_epoch = 0
	itr= 0
	early_stopping = EarlyStopping(patience=50, delta=0.0, logger=logger)
	best_results = None # dictionary of key metrics for the best epoch (by validation acc)
	for epoch in range(1, config.epochs+1):

		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		val_acc_epoch = 0.0
		model.train()
		start_time = time()
		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']

		for batch, i in enumerate(range(0, len(train_loader), config.batch_size)):
			if config.model_type == 'RNN':
				hidden = model.model.init_hidden(config.batch_size)
			else:
				hidden = None
			source, targets, word_lens = train_loader.get_batch(i)
			source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)
			loss = model.trainer(source, targets, word_lens, hidden, config)
			train_loss_epoch += loss 
			itr += 1
		
		train_loss_epoch = train_loss_epoch/train_loader.num_batches
		# print time in mins and seconds
		time_taken = time() - start_time
		time_taken = '{:5.0f}m {:2.0f}s'.format(time_taken // 60, time_taken % 60)
		log_print('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		log_print('Starting Validation')

		val_acc_epoch = run_validation(config, model, val_loader, voc, device, logger)
		train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)

		# If noiseless validation is not enabled, the model will report '0'
		noiseless_val_acc_epoch = 0
		if config.noiseless_validation is not None:
			noiseless_val_acc_epoch = run_validation(config, model, noiseless_val_loader, voc, device, logger)

		epoch_results = {
				"epoch": epoch,
				"train_loss": train_loss_epoch,
				"train_acc": train_acc_epoch,
				"noiseless_val_acc": noiseless_val_acc_epoch,
				"val_acc": val_acc_epoch,
			}

		# Early stopping is based on _loss_, so we negate the accuracy
		early_stopping( (-1) * val_acc_epoch, model)

		if config.opt == 'sgd':
			model.scheduler.step(val_acc_epoch)

		if config.mode == 'tune':
			manager.report(epoch_results)
		
		if val_acc_epoch > max_val_acc :
			max_val_acc = val_acc_epoch
			best_epoch = epoch
			curr_train_acc= train_acc_epoch
			best_results = copy.deepcopy(epoch_results)

		# Break if we haven't had consistent progress 
		if early_stopping.early_stop:
			break

		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['train_loss'] = train_loss_epoch
		od['train_acc'] = train_acc_epoch
		od['val_acc_epoch']= val_acc_epoch
		od['noiseless_val_acc_epoch'] = noiseless_val_acc_epoch
		od['max_val_acc']= max_val_acc
		od['lr_epoch'] = lr_epoch
		print_log(logger, od)

	logger.info('Training Completed for {} epochs'.format(epoch))
	if config.results and config.mode == 'train':
		# These results are redundant with the tuning directory structure
		store_results(config, max_val_acc, curr_train_acc, best_epoch, noiseless_val_acc_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))
	
	return best_results
		

def run_validation(config, model, data_loader, voc, device, logger):
	"""
	Run validation on the given dataset and compute accuracy.

	Args:
		config: Configuration object with training parameters.
		model: PyTorch model to evaluate.
		data_loader: DataLoader object for validation data.
		voc: Vocabulary object.
		device: Device to run the model on (e.g., "cpu" or "cuda").
		logger: Logger object for logging messages.

	Returns:
		val_acc_epoch (float): Average accuracy over the validation dataset.
	"""
	logger.info("Starting validation...")
	model.eval()  # Switch model to evaluation mode
	batch_num = 0
	val_acc_epoch = 0.0
	
	with torch.no_grad():  # Disable gradient computation for validation
		for batch, i in enumerate(range(0, len(data_loader), data_loader.batch_size)):
			if config.model_type == 'RNN':
				hidden = model.model.init_hidden(config.batch_size)
			else:
				hidden = None
			source, targets, word_lens = data_loader.get_batch(i)
			source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)

			# Note/warning: `hidden` is not used in the evaluation function for the SAN model
			# this is just for expediency.
			acc = model.evaluator(source, targets, word_lens, config, hidden=hidden)

			# Log individual batch accuracy
			# log_print(f"Batch {batch_num}: Accuracy={acc}")
			val_acc_epoch += acc
			batch_num += 1

	# Ensure all batches were processed
	if batch_num != data_loader.num_batches:
		logger.warning(
			f"Number of processed batches ({batch_num}) does not match total batches ({data_loader.num_batches})"
		)

	# Compute average validation accuracy
	val_acc_epoch = val_acc_epoch / max(1, data_loader.num_batches)
	logger.info(f"Validation completed: Average Accuracy={val_acc_epoch:.4f}")
	return val_acc_epoch


class EarlyStopping:
	def __init__(self, patience=5, delta=0, logger=None):
		"""Implement early stopping if the validation loss doesn't improve after a given patience period.
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 5
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
						   Default: 0
		"""
		self.patience = patience
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = float('inf')
		self.delta = delta
		self.logger = logger

	def __call__(self, val_loss, model):
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score <= self.best_score + self.delta:
			self.counter += 1
			if self.logger:
				self.logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		'''Saves model when validation loss decrease.'''
		if self.logger:
			self.logger.debug(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), 'checkpoint.pt')
		self.val_loss_min = val_loss
