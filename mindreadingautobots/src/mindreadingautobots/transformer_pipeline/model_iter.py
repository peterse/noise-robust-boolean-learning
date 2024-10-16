import os
import sys
import math
import logging
import ipdb as pdb
import random
from time import time
from typing import OrderedDict
import numpy as np
import wandb
import copy

from mindreadingautobots.utils.helper import save_checkpoint
from mindreadingautobots.transformer_pipeline.components.transformer import TransformerCLF

from mindreadingautobots.utils.logger import store_results, print_log
import torch
import torch.nn as nn
from torch import optim

from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray import tune, train
from ray.train import RunConfig
from ray.train.torch import TorchTrainer, get_device
from ray.util.check_serialize import inspect_serializability
from mindreadingautobots.utils.sentence_processing import idx_to_sent, idxs_to_sent
from mindreadingautobots.utils.dataloader import SamplerIter


class SeqClassifier(nn.Module):
	def __init__(self, config=None, voc=None, device=None, logger=None):
		super(SeqClassifier, self).__init__()

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

		self.model = TransformerCLF(self.voc.nwords, self.config.nlabels, self.config.d_model,
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
	

	def evaluator(self, source, targets, lengths, config, device=None):
		
		# if config.model_type == 'RNN':
		# 	output, hidden = self.model(source, hidden, lengths)
		
		output = self.model(source, lengths)
		preds = output.cpu().numpy()
		preds = preds.argmax(axis=1)
		labels= targets.cpu().numpy()
		acc= np.array(preds==labels, np.int32).sum() / len(targets)

		return acc
		

####################################


def build_model(config, voc, device, logger):
	model = SeqClassifier(config, voc, device, logger)
	model = model.to(device)
	return model


def train_model(model, train_loader, val_loader, voc, device, 
				config, logger, epoch_offset= 0, min_val_loss=1e7, 
				max_val_acc=0.0, writer= None):

	best_epoch = 0
	curr_train_acc=0.0
	early_stop_count=0
	max_train_acc = 0.0
	if config.wandb:
		wandb.watch(model, log_freq= 1000)

	itr= 0
	gen_success=False
	conv_time = -1
	conv = False
	data_size = len(train_loader)
	estop_lim = 1000 * (config.batch_size // data_size)

	for epoch in range(1, config.epochs):

		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		val_acc_epoch = 0.0
		model.train()
		start_time = time()
		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']

		for batch, i in enumerate(range(0, len(train_loader), config.batch_size)):

			# if config.model_type == 'RNN':
			# 	hidden = model.model.init_hidden(config.batch_size)
			# else:
			# 	hidden = None
		
			source, targets, word_lens = train_loader.get_batch(i)			
			source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)
			loss = model.trainer(source, targets, word_lens, config)
			train_loss_epoch += loss 
			itr +=1
		
		train_loss_epoch = train_loss_epoch/train_loader.num_batches
		time_taken = time() - start_time
		time_mins = int(time_taken/60)
		time_secs= time_taken%60

		logger.debug('Training for epoch {} completed...\nTime Taken: {} mins and {} secs'.format(epoch, time_mins, time_secs))
		logger.debug('Starting Validation')

		val_acc_epoch = run_validation(config, model, val_loader, voc, device, logger)
		train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)
		gen_gap = train_acc_epoch- val_acc_epoch

		if config.opt == 'sgd':
			model.scheduler.step(val_acc_epoch)

		if config.wandb:
			wandb.log({
				'train-loss': train_loss_epoch,
				'train-acc': train_acc_epoch,
				'val-acc':val_acc_epoch,
				'gen-gap': gen_gap,
				})

		if config.mode == 'tune':
			train.report({
				"train_loss": (train_loss_epoch),
				"train_acc": train_acc_epoch,
				"val_acc": val_acc_epoch
			})

		if val_acc_epoch > max_val_acc :
			max_val_acc = val_acc_epoch
			best_epoch= epoch
			curr_train_acc= train_acc_epoch

		if val_acc_epoch> 0.9999:
			early_stop_count +=1

		else:
			early_stop_count=0

		if early_stop_count > estop_lim:
			break
		if train_acc_epoch> 0.999:
			early_stop_tr +=1
		else:
			early_stop_tr=0

		if early_stop_tr > 30:
			break

		
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['train_loss'] = train_loss_epoch
		od['train_acc'] = train_acc_epoch
		od['val_acc_epoch']= val_acc_epoch
		od['max_val_acc']= max_val_acc
		od['lr_epoch'] = lr_epoch
		# od['conv_time'] = conv_time
		print_log(logger, od)

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.wandb:
		wandb.log({
			'max-val-acc': max_val_acc,
			# 'gen-success': gen_success,
			# 'conv-time': conv_time,
			})
	


	if config.results:
		store_results(config, max_val_acc, curr_train_acc, best_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))

	
def build_and_train_model_raytune(hyper_config, config, train_loader, val_loader, voc, 
				logger, epoch_offset= 0, min_val_loss=1e7, 
				max_val_acc=0.0,
				):
	"""Build and train a model with the given hyperparameters
	
	Reasons why this exists:
	 - ray (or pickle) cannot serialize torch models
	 - We need to distribute the model training/building pipeline to multiple workers
	"""
	device = get_device()
	for key, value in hyper_config.items():
		setattr(config, key, value)
	model = build_model(config, voc, device, logger)
	train_model(model, train_loader, val_loader, voc, device, config, logger, epoch_offset, min_val_loss, max_val_acc)

def trial_dirname_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"

def tune_model(hyper_settings, hyper_config, train_loader, val_loader, voc, 
				config, logger, epoch_offset= 0, min_val_loss=1e7, 
				max_val_acc=0.0):	
	
	
	# config should have tune=True
	scheduler = ASHAScheduler(
		metric="train_loss",
		mode="min",
		max_t=hyper_settings.get("epochs"), # I'm not sure what this kwarg does and neither is the documentation
		grace_period=1,
		reduction_factor=2)

	reporter = CLIReporter(
		metric_columns=["loss", "training_iteration", "mean_accuracy"],
		print_intermediate_tables=False,
		)

	tune_config = tune.TuneConfig(
		num_samples=hyper_settings.get("num_samples"),
		trial_dirname_creator=trial_dirname_creator,
		scheduler=scheduler,
		max_concurrent_trials=hyper_settings.get("max_concurrent_trials"),
		)
		
	run_config = RunConfig(
		progress_reporter=reporter,
		stop={"training_iteration": config["epochs"], "mean_accuracy": 0.8},
	)
	trainable = tune.with_parameters(
					build_and_train_model_raytune, 
					config=config,
					train_loader=train_loader,
					val_loader=val_loader,
					voc=voc,
					logger=logger,
					epoch_offset=epoch_offset,
					min_val_loss=min_val_loss,
					max_val_acc=max_val_acc
					)
	
	resources = tune.with_resources(
		trainable,
		{"cpu": hyper_settings.get("cpus_per_worker"), "gpu": hyper_settings.get("gpus_per_worker")},
	)
	
	tuner = Tuner(
		resources,
		param_space=hyper_config,
		tune_config=tune_config,
		run_config=run_config,
	)
	result = tuner.fit()

	return result			

# def train_model_iter(model, voc, device, config, logger):

# 	best_epoch = 0
# 	curr_train_acc=0.0
# 	early_stop_count=0

# 	conv_time = -1
# 	conv = False

# 	max_val_acc=0.0
# 	max_train_acc = 0.0
# 	if config.wandb:
# 		wandb.watch(model, log_freq= 1000)

# 	init_model = copy.deepcopy(model.model)
# 	init_distance= 0.0
# 	max_init_dist= 0.0

# 	sampler = SamplerIter(voc, config.batch_size)

# 	gen_success = False
# 	start_time = time()

# 	epoch_iter = 100
# 	estop_lim = 15


# 	for itr in range(1, config.iters+1):
# 		val_acc_epoch = 0.0
# 		model.train()

# 		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']
# 		source, targets, word_lens = sampler.get_batch()
# 		source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)
# 		loss = model.trainer(source, targets, word_lens, config)

# 		# train_loss_epoch = train_loss_epoch/train_loader.num_batches		

# 		if config.opt == 'sgd':
# 			model.scheduler.step(val_acc_epoch)
		
# 		if itr%epoch_iter != 0:

# 			if config.wandb:
# 				wandb.log({
# 					'train-loss': loss,
# 					})
			
# 		else:
# 			time_taken = time() - start_time
# 			time_mins = int(time_taken/60)
# 			time_secs= time_taken%60
# 			start_time = time()

# 			logger.debug('Training for {} iters at {} completed...\nTime Taken: {} mins and {} secs'.format(epoch_iter, itr, time_mins, time_secs))
# 			logger.debug('Starting Validation')

# 			val_acc_epoch = run_validation_iter(config, model, 10000, voc, device)


# 			if config.wandb:
# 				wandb.log({
# 					'train-loss': loss,						
# 					'val-acc':val_acc_epoch,
# 					'init-dist': init_distance,
# 					})

# 			if val_acc_epoch > max_val_acc :
# 				max_val_acc = val_acc_epoch
# 				# best_epoch= epoch
				
# 			if val_acc_epoch> 0.9999:
# 				if not conv:
# 					gen_success = True
# 					conv_time= itr
# 					conv = True

# 				early_stop_count +=1
# 			else:
# 				early_stop_count=0

# 			if early_stop_count > estop_lim:
# 				break

# 			od = OrderedDict()
# 			od['Iterations'] = itr
# 			od['train_loss'] = loss
			
# 			od['val_acc_epoch']= val_acc_epoch
# 			od['max_val_acc']= max_val_acc
# 			# od['lr_epoch'] = lr_epoch
# 			if config.init_dist>0:
# 				od['init_dist'] = init_distance
# 			print_log(logger, od)

			
# ### After Training loop

# 	logger.info('Training Completed for {} iterations'.format(itr))

# 	if config.wandb:
# 		if config.init_dist >0:
# 			wandb.log({
# 				'max-val-acc': max_val_acc,
# 				'max-init-dist': max_init_dist,	
# 				'gen-success': gen_success,
# 				'conv-time': conv_time,
# 				})
# 		else:
# 			wandb.log({
# 				'max-val-acc': max_val_acc,
# 				})


# 	if config.results:
# 		store_results(config, max_val_acc, curr_train_acc, best_epoch)
# 		logger.info('Scores saved at {}'.format(config.result_path))

		
def run_validation(config, model, data_loader, voc, device, logger):
	model.eval()
	batch_num = 0
	val_acc_epoch = 0.0

	with torch.no_grad():
		for batch, i in enumerate(range(0, len(data_loader), data_loader.batch_size)):


			source, targets, word_lens= data_loader.get_batch(i)
			source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)

			acc = model.evaluator(source, targets, word_lens, config)

			val_acc_epoch+= acc
			batch_num+=1
	
	if batch_num != data_loader.num_batches:
		pdb.set_trace()

	val_acc_epoch = val_acc_epoch/data_loader.num_batches

	return val_acc_epoch


# def run_validation_iter(config, model, samples, voc, device):
# 	model.eval()
# 	batch_num = 0
# 	val_acc_epoch = 0.0

# 	sampler = SamplerIter(voc, config.batch_size)
# 	itrs = samples//config.batch_size
# 	with torch.no_grad():
		
# 		for i in range(itrs):
		
# 			source, targets, word_lens = sampler.get_batch()
# 			source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)

# 			acc = model.evaluator(source, targets, word_lens, config)

# 			val_acc_epoch+= acc
# 			batch_num+=1

# 	val_acc_epoch = val_acc_epoch/itrs

# 	return val_acc_epoch





	