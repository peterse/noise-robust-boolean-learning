import pdb
from time import time
from typing import OrderedDict
import numpy as np
import wandb
import copy

import torch
import torch.nn as nn
from torch import optim

# from ray.util.check_serialize import inspect_serializability

from mindreadingautobots.utils.logger import store_results, print_log
from mindreadingautobots.utils.training import EarlyStopping


def train_model(model, train_loader, val_loader, noiseless_val_loader, voc, device, 
				config, logger, epoch_offset=0):

	max_val_acc = 0
	best_epoch = 0
	early_stop_count=0
	itr= 0

	num_batches = int(train_loader.num_batches)
	early_stopping = EarlyStopping(patience=50, delta=0.0, logger=logger)

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
		print('Time Taken for epoch {} : {}'.format(epoch, time_taken))
	
		logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		logger.debug('Starting Validation')

		val_acc_epoch = run_validation(config, model, val_loader, voc, device, logger)
		train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)

		# If noiseless validation is not enabled, the model will report '0'
		noiseless_val_acc_epoch = 0
		if config.noiseless_validation is not None:
			noiseless_val_acc_epoch = run_validation(config, model, noiseless_val_loader, voc, device, logger)

		gen_gap = train_acc_epoch - val_acc_epoch
		# Early stopping is based on _loss_, so we negate the accuracy
		early_stopping( (-1) * val_acc_epoch, model)

		if config.opt == 'sgd':
			model.scheduler.step(val_acc_epoch)

		if config.mode == 'tune':
			raise NotImplementedError('Tuning not implemented yet')
		
		if val_acc_epoch > max_val_acc :
			max_val_acc = val_acc_epoch
			best_epoch= epoch
			curr_train_acc= train_acc_epoch

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
	if config.results:
		store_results(config, max_val_acc, curr_train_acc, best_epoch, noiseless_val_acc_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))
		

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
			logger.debug(f"Batch {batch_num}: Accuracy={acc}")
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