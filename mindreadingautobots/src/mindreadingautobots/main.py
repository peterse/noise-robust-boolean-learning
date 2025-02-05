import os
import sys
import logging
import pdb
import random
import numpy as np
import datetime
# from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
# from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

from ray import tune
from mindreadingautobots.pipelines.training import train_model, load_data, build_model
from mindreadingautobots.pipelines import tuning


from mindreadingautobots.utils.helper import Voc, gpu_init_pytorch, create_save_directories, get_latest_checkpoint, count_parameters, validate_tuning_parameters
from mindreadingautobots.utils.logger import init_logger

from mindreadingautobots.pipelines.args import build_parser

import yaml 


global log_folder
global model_folder
global result_folder
global data_path



def load_hyperparameters(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
	print('Starting....')

	'''Read arguments'''
	parser = build_parser()
	args = parser.parse_args()
	config = args
	
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
	model_folder = 'train_results/models'
	result_folder = '/out/'
	data_path = 'data/'

	config.run_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	config.model_path = os.path.join(model_folder, config.dataset, config.run_name)
	config.abs_path = os.path.dirname(os.path.abspath(__file__)) # current file's path
	config.data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', data_path))

	# use a timestamp YYYYMMDDHHMMSS to identify the run
	if config.mode == 'train':
		config.log_path = 'train_results/'
		log_file = os.path.join(config.log_path, f'{config.run_name}.txt')
	elif config.mode == 'tune':
		config.tune_directory = tuning.make_tune_directory(config, config.abs_path) # makes tune_results/{model}_{dataset}/run_{timestamp}
		log_file = os.path.join(config.tune_directory, 'log.txt')

	logger = init_logger(config.run_name, log_file_path=log_file, logging_level=logging.DEBUG)

	# # # # # MANUAL SETTINGS
	config.patience = 50

	if config.mode == 'train':
		vocab_path = os.path.join(config.model_path, 'vocab.p')
		config_file = os.path.join(config.model_path, 'config.p')
		log_file = os.path.join(config.log_path, 'log.txt')

		if config.results:
			config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

		if config.mode == 'train' or config.mode == 'tune':
			create_save_directories(config.log_path, config.model_path, result_folder)
		else:
			create_save_directories(config.log_path, None, result_folder)
		
		logger.debug('Created Relevant Directories')
		logger.info('Experiment Name: {}'.format(config.run_name))

		voc, train_loader, val_loader, noiseless_val_loader = load_data(config, logger)
		config.nlabels= train_loader.corpus.nlabels
		logger.info('Vocab Created with number of words : {}'.format(voc.nwords))	

		checkpoint = get_latest_checkpoint(config.model_path, logger)
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
		
		num_params = count_parameters(model)
		logger.info('Number of parameters {}'.format(num_params))		
		logger.info('Starting Training Procedure')
		train_model(model, train_loader, val_loader, noiseless_val_loader, voc,
					device, config, logger, epoch_offset)

	elif config.mode == "tune":
		# these are the parameterized hyperparameters we want to tune over
		# Comment out anything that you are not tuning over, to save redundant information
		# from the tuning results.  

		yaml = load_hyperparameters(config.hyper_config_path)
		model_type = config.model_type 
		model_type_from_yaml = yaml["model_type"] 

		if model_type != model_type_from_yaml:
			raise ValueError(f"Model type {model_type} from args is different from model type {model_type_from_yaml} in hyperparameter config file")
		hyper_config = yaml["hyperparameters"]
		
		# if config.model_type == 'RNN':
		# 	hyper_config = {
		# 		'lr': np.logspace(-4,-2, num=20, base=10.0),
		# 		'emb_size': np.array([16, 32, 64]),
		# 		'hidden_size': np.array([16, 32, 64]),
		# 		'dropout': [0.05], # dropout is default 0.05
		# 		'depth': np.array([1,2,3, 4, 5, 6]),
		# 		'cell_type': ['LSTM']
		# 	}
		# elif config.model_type == 'SAN':
		# 	hyper_config = {
		# 		'lr': np.logspace(-5,-2, num=20, base=10.0),
		# 		'depth': np.array([1,2, 3]),
		# 		'd_model': np.array([32, 64]),
		# 		'dropout': [0.05, 0.1],# dropout is default 0.05
		# 		'heads': np.array([2, 4]),
		# 		'd_ffn': np.array([32, 64, 128]),
		# 	}
		if model_type == 'SAN':
			for h in hyper_config['heads']:
				for d_model in hyper_config['d_model']:
					if d_model % h != 0:
						raise ValueError(f"d_model must be divisible by heads. Cannot have d_model={d_model} and heads={h}")
		
		print('Hyperparameters to tune over: ', hyper_config)
		print('Model Type: ', model_type)	
		# Verification
		validate_tuning_parameters(config, hyper_config, logger)

		# these specify how tune will work
		hyper_settings = {
			"total_cpus": 160, 
			"total_gpus": 0,
			"num_samples": 300, 
		}
		tuning.tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, config, logger)
		


if __name__ == '__main__':
	main()