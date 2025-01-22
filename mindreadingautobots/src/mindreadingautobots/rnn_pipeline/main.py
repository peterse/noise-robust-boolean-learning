import os
import sys
import math
import logging
import pdb
import random
import numpy as np
# from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
# from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

import wandb
from ray import tune

from mindreadingautobots.rnn_pipeline.args import build_parser
from mindreadingautobots.utils.dataloader import Corpus, Sampler
from mindreadingautobots.utils.helper import Voc, gpu_init_pytorch, create_save_directories, get_latest_checkpoint, count_parameters
from mindreadingautobots.utils.logger import get_logger
from mindreadingautobots.rnn_pipeline.model import build_model, train_model, tune_model
from mindreadingautobots.models import hyperparameters


global log_folder
global model_folder
global result_folder
global data_path

log_folder = 'logs'
model_folder = 'models'
result_folder = './out/'

data_path = 'data/'
data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../..', data_path))

# navigate to two directories up where /data is stored
# cwd = os.getcwd()
# data_path = os.path.join(os.path.dirname(os.path.dirname(cwd)), data_path)

# board_path = './runs/'


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


def main():
	'''Read arguments'''

	print('Starting....')
	parser = build_parser()
	args = parser.parse_args()
	config = args
	mode= config.mode # train, test, tune

	if mode == 'train':
		is_train = True
	else:
		is_train= False

	is_tune = False
	if mode == 'tune':
		is_tune = True
	
	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''device initialization'''
	if config.gpu is not None:
		device = gpu_init_pytorch(config.gpu)
	else:
		device = torch.device('cpu')

	'''Run Config files/paths'''
	run_name = config.run_name
	config.log_path = os.path.join(log_folder, run_name)
	config.model_path = os.path.join(model_folder, config.dataset, run_name)

	vocab_path = os.path.join(config.model_path, 'vocab.p')
	config_file = os.path.join(config.model_path, 'config.p')
	log_file = os.path.join(config.log_path, 'log.txt')

	if config.results:
		config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))
	
	if is_train or is_tune:
		create_save_directories(config.log_path, config.model_path, result_folder)
	else:
		create_save_directories(config.log_path, None, result_folder)
	
	logger = get_logger(run_name, log_file, logging.DEBUG)

	logger.debug('Created Relevant Directories')
	logger.info('Experiment Name: {}'.format(config.run_name))
	
	if is_train or is_tune:
		voc, train_loader, val_loader, noiseless_val_loader = load_data(config, logger)
		config.nlabels= train_loader.corpus.nlabels
		logger.info('Vocab Created with number of words : {}'.format(voc.nwords))		

	# 	with open(vocab_path, 'wb') as f:
	# 		pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)
	# else:
	# 	# FIXME: this seems broken? nothing else happens if is_train is False...
	# 	test_dataloader = load_data(config, logger)
	# 	# logger.info('Loading Vocab File...')

	# 	with open(vocab_path, 'rb') as f:
	# 		voc = pickle.load(f)

		# logger.info('Vocab Files loaded from {}'.format(vocab_path))
	
	if is_train:
		checkpoint = get_latest_checkpoint(config.model_path, logger)

		min_val_loss = torch.tensor(float('inf')).item()
		# min_val_ppl = float('inf')
		epoch_offset= 0

		if checkpoint:
			ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
			config.lr = ckpt['lr']
			model = build_model(config=config, voc=voc, device=device, logger=logger)
			model.load_state_dict(ckpt['model_state_dict'])
			model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
		else:
			model = build_model(config=config, voc=voc, device=device, logger=logger)

		logger.info('Initialized Model')
		with open(config_file, 'wb') as f:
			pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)
		logger.debug('Config File Saved')

		num_params =count_parameters(model)
		logger.info('Number of parameters {}'.format(num_params))

		### Wandb Initialization ###
		if config.wandb:
			metrics = dict(
				num_params= num_params,
				learning_rate = config.lr,
				hidden_size= config.hidden_size,
				depth= config.depth,
				emb_size = config.emb_size,
				dataset= config.dataset,
				dropout= config.dropout,
				optimizer =config.opt,
				epochs = config.epochs,
				batch_size= config.batch_size,
			)
			
			wandb.init(
				project= config.project,
				group= config.dataset,
				name= config.run_name,
				config= metrics,
				entity=config.entity,
			)
		
		logger.info('Starting Training Procedure')
		train_model(model, train_loader, val_loader, noiseless_val_loader, voc,
					device, config, logger, epoch_offset)

	elif is_tune:
		# Hyperparameter tuning happens here. 
		# The actual hyperparameters usued will be sampled from hyper_config randomly
		# for each trial in `num_samples`. 
		hyper_config = {
			'lr': tune.loguniform(1e-4, 1e-2),
			'hidden_size': tune.choice([16, 32, 64]),
			'depth': tune.choice([1, 2, 3]),
		}
		#If you want to deterministically visit
		# all possible hyperparameters, you can use tune.grid_search, e.g. the following:
		# hyper_config = {
		# 	'hidden_size': tune.grid_search([16, 32, 64, 128]),
		# 	'depth': tune.grid_search([1, 2, 3]),
		# }
		# WARNING: IF YOU USE GRID SEARCH, IT WILL RUN ALL COMBINATIONS OF THE PARAMETERS
		# `num_samples` TIMES EACH. e.g. if num_samples=60, the above will run 720 trials!

		# The way raytune distributes compute resources is to use all possible resources,
		# then maximize the number of trials such that cpus/gpus per worker below are satisfied.
		# To avoid overutilization, set `max_concurrent_trials`
		# https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html
		hyper_settings = {
			"total_cpus": 1,
			"total_gpus": 0,
			"cpus_per_worker": 1, #i.e. cpus per trial
			"gpus_per_worker": 0,
			"max_concurrent_trials": 1,
			"grace_period": 25, # minimum epochs to give each trial
			"max_iterations": 1000, # this is the max epochs any trial is allowed to run
			"num_samples": 1, # this is equal to total trials if no grid search
		}

		min_val_loss = torch.tensor(float('inf')).item()
		epoch_offset= 0

		logger.info('Starting Tuning Procedure')

		# Overwriting the config settings with the hyperparameters
		config.tune = True
		for key, value in hyper_config.items():
			setattr(config, key, value)

		tune_model(hyper_settings, hyper_config, train_loader, val_loader, noiseless_val_loader, voc, config, logger, epoch_offset)
		


if __name__ == '__main__':
	main()