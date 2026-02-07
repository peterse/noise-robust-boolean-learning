import os
import multiprocessing as mp
import numpy as np
import json
import datetime
import torch
import pandas as pd
import itertools

from mindreadingautobots.pipelines.training import train_model, load_data, build_model
from mindreadingautobots.utils.logger import init_logger, get_logger
from mindreadingautobots.entropy_and_bayesian import boolean
from mindreadingautobots.sample_f_example import generate_data
from mindreadingautobots.entropy_and_bayesian.entropy import empirical_entropy_estimate


def make_tune_results_path(path):
    return os.path.join(path, "tune_results.csv")

def hyper_config_path(path):
    return os.path.join(path, f"hyper_config.json")

def get_header():
    return "epoch,train_loss,train_acc,val_acc,noiseless_val_acc,final_train_acc,final_val_acc,final_noiseless_val_acc,sensitivity"

def get_header_sample_f():
    return "epoch,train_loss,train_acc,val_acc,noiseless_val_acc,final_train_acc,final_val_acc,final_noiseless_val_acc,sensitivity,sens_f,sens_g,err_f,err_g,entropy"

class ThreadManager:
	"""
	Each thread has its own manager to handle file I/O operations and stdout.

	We will write to `tune_directory`, which is a directory that is unique to this thread
	(timestamp for uniqueness). This directory will be populated with:
	- a CSV file for the results of each epoch: job_results.csv
	- a JSON file for the hyperparameters: hyper_config_{thread_id}.json
	"""
	def __init__(self, thread_info):
		self.thread_id = thread_info.get("thread_id")
		self.tune_path = thread_info.get("tune_path")  # tune_results/{model}_{dataset}/run_{timestamp}/threads/job_<thread_id>/
		self.logger_name = thread_info.get("logger_name")
		self.tune_results_path = os.path.join(self.tune_path, f"job_results.csv")
		self.log_path = os.path.join(self.tune_path, f"log.txt")

		# initialize logging
		log_path = os.path.join(self.tune_path, "log.txt") # where the train_model will log
		logger_name = f"train_model_{self.thread_id}"
		self.logger = init_logger(logger_name, log_path)

		self.logger.info(f"Thread {self.thread_id} writing to {self.tune_path}")
		with open(self.tune_results_path, "w") as f:
			f.write(get_header() + "\n")

	def report(self, epoch_results):
		"""Report results from the training loop, according to the header in `get_header()`"""
		with open(self.tune_results_path, "a") as f:
			sensitivity = f",{epoch_results['sensitivity']}" if 'sensitivity' in epoch_results else ""
			f.write(f"{epoch_results['epoch']},{epoch_results['train_loss']},{epoch_results['train_acc']},{epoch_results['val_acc']},{epoch_results['noiseless_val_acc']},{epoch_results['final_train_acc']},{epoch_results['final_val_acc']},{epoch_results['final_noiseless_val_acc']}{sensitivity}\n")

	def save_configs(self, hyper_config):
		with open(hyper_config_path(self.tune_path), "w") as f:
			f.write(json.dumps(hyper_config))

	def log_print(self, message):
		self.logger.info(message)


def make_tune_directory(config, abs_path):
	"""Build top-level tuning directories: tune_results/{model}_{dataset_module}/run_<run_id>"""
	tune_directory = os.path.join(abs_path, "tune_results")
	model_name = config.model_type
	dataset_module = config.dataset
	model_subdir = f"{model_name}_{dataset_module}"
	tune_directory = os.path.join(tune_directory, model_subdir)

	# stamp this run with YYYY-MM-DD-HH-MM-SS-MS
	run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
	run_directory = os.path.join("run_" + run_id) # makes tune_results/{model}_{dataset}/run_{timestamp}
	tune_directory = os.path.join(tune_directory, run_directory)
	if not os.path.exists(tune_directory):
		os.makedirs(tune_directory)
	return tune_directory


def train_model_multiprocessing(package):
	"""The function to train a model called by a specific thread.
	
	This function is designed to be called from within a thread, and therefore must be 
	initialized from purely serializable args. 
	"""
	# Note: Function wrapping within `tune_hyperparameters_multiprocessing` is dangerous
	# because of serializability issues, e.g. pickling a function that is not defined at the module level
	(hyper_config, config, thread_info) = package

	# Initialize a logging system for this thread
	device = torch.device("cpu") # gpu is not implemented
	manager = ThreadManager(thread_info)
	manager.log_print(f"initializing multiprocessing on: {device}")
	manager.log_print(f"Process ID: {os.getpid()}, Process Name: {mp.current_process().name}")
	
	# merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
	# Then dispatch to the training pipeline
	for k, v in hyper_config.items():
		setattr(config, k, v)
		manager.log_print(f"Hyperparameter {k} set to {v}")

	voc, train_loader, val_loader, noiseless_val_loader = load_data(config, manager.logger)
	model = build_model(config=config, voc=voc, device=device, logger=manager.logger)
	best_results = train_model(model, train_loader, val_loader, noiseless_val_loader, voc,
					device, config, manager.logger, 0, manager=manager)
	
	# wrap-up operations: save config as json, save hyper_config as json
	manager.save_configs(hyper_config)
	return best_results


def tune_hyperparameters_multiprocessing(hyper_config, hyper_settings, config, logger):
	"""
	hyper_config should contain a sequence-type for each hyperparameter in the search, e.g.
	hyper_config = {
		'lr': np.loguniform(1e-4, 1e-2),
		'emb_size': [8, 16, 32, 64],
		'depth': [1, 2, 3, 4],
}
	"""
	num_cpus=hyper_settings.get("total_cpus")
	num_gpus=hyper_settings.get("total_gpus") 
	if num_gpus != 0:
		raise NotImplementedError("GPU support not yet implemented")

	# Number of CPUs to use
	tot_cpus = mp.cpu_count()  # Get the number of available CPUs
	logger.debug(f"Number of available CPUs: {tot_cpus}")
	logger.debug(f"Number of CPUs to use: {num_cpus}")

	# build the hyperparameter space by sampling (gridsearch not yet supported)
	# Each element of this list is a dictionary of hyperparameters with the same 
	# keys as `hyper_config`
	hyper_list = []
	for _ in range(hyper_settings.get("num_samples")):
		hyperparameter_slice = {}
		for key, value in hyper_config.items():
			hyperparameter_slice[key] = np.random.choice(value).item() # keep serializable
		hyper_list.append(hyperparameter_slice)

	# Assemble local variables to be called from within a thread
	tune_directory = config.tune_directory
	threads_directory = os.path.join(tune_directory, "threads/")
	package_list = []
	paths_list = []
	for thread_id, hyperparameter_slice in enumerate(hyper_list):
		# Configure tuning path for this specific thread
		# note that the thread id does not belong to a specific thread, but rather to a job
		tune_path = os.path.join(threads_directory, f"job_{thread_id}") # where this thread's results live: .../threads/job_<thread_id>
		os.makedirs(tune_path)
		paths_list.append(tune_path)
		logger_name = f"thread_{thread_id}" # we cannot serialize a logger effectively, so we pass around its name
		thread_info = {"thread_id": thread_id, "tune_path": tune_path, "logger_name": logger_name}
		package = (hyperparameter_slice, config, thread_info)
		package_list.append(package)

	with mp.Pool(processes=num_cpus) as pool:
		# Map f to the list of parameters
		all_results = pool.map(train_model_multiprocessing, package_list)
  
	#cleanup
	logger.debug("Cleaning up...")
	hyper_keys = list(hyper_config.keys())
	header_keys = get_header().split(",")
	columns = header_keys + hyper_keys

	data = []
	for i in range(len(all_results)):
		hyper_setting = [hyper_list[i].get(k) for k in hyper_keys]
		best_result = [all_results[i].get(k) for k in header_keys]
		data.append(best_result + hyper_setting)
	df = pd.DataFrame(data, columns=columns)
	df.to_csv(os.path.join(tune_directory, f"{config.model_type}_{config.dataset}_results.csv"), index=False)

	config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
	for k in hyper_keys:                                                                                     
		config_dict[k] = None
	with open(os.path.join(tune_directory, f"config.json"), "w") as f:
		f.write(json.dumps(config_dict))

	logger.debug("Tuning run completed successfully...")


def train_model_multiprocessing_sample_f(package):
	"""The function to train a model called by a specific thread, with on-the-fly data generation.
	
	This function is designed to be called from within a thread, and therefore must be 
	initialized from purely serializable args. 
	"""
	# Note: Function wrapping within `tune_hyperparameters_multiprocessing` is dangerous
	# because of serializability issues, e.g. pickling a function that is not defined at the module level
	(hyper_config, config, thread_info) = package

	# Initialize a logging system for this thread
	device = torch.device("cpu") # gpu is not implemented
	manager = ThreadManager(thread_info)
	manager.log_print(f"initializing multiprocessing on: {device}")
	manager.log_print(f"Process ID: {os.getpid()}, Process Name: {mp.current_process().name}")
	
	# merge the hyperparameter config into the ordinary config, giving priority to the hyperparameter config
	# Then dispatch to the training pipeline
	for k, v in hyper_config.items():
		setattr(config, k, v)
		manager.log_print(f"Hyperparameter {k} set to {v}")

	# Generate data on-the-fly instead of loading from files
	manager.log_print(f"Generating dataset with n_bool={config.n_bool}, bf_bool={config.bf_bool}, seed_bool={config.seed_bool}")
	voc, train_loader, val_loader, noiseless_val_loader, bool_func = generate_data(config=config, logger=manager.logger)
	model = build_model(config=config, voc=voc, device=device, logger=manager.logger)

	best_results = train_model(model, train_loader, val_loader, noiseless_val_loader, voc,
					device, config, manager.logger, 0, manager=manager)
	
	# now, update best_results with sens_f, sens_(f_N*), err(f), err(f_N*)
	f = 2 * np.array(bool_func) - 1
	fhat = boolean.boolean_fourier_transform(f)
	bf_bool = config.bf_bool
	rho = 1 - 2 * bf_bool
	n = config.n_bool
	# fourier transform f, apply noise operator in fourier space, fourier transform to get 
	# trhof, then take sign, then fourier transform to get ghat
	H = boolean.walsh_hadamard_matrix(n)
	trho_mask = boolean.build_trho_mask(n, rho)
	Trhof_hat = np.multiply(trho_mask, fhat)
	Trho_f = H @ Trhof_hat
	all_sgn_Trho_f = np.sign(Trho_f)
	ghat = H @ all_sgn_Trho_f
	sens_f = boolean.compute_influence_fourier(fhat.reshape(-1, 1)).flatten()[0]
	sens_g = boolean.compute_influence_fourier(ghat.reshape(-1, 1)).flatten()[0]
	dot_f = np.dot(Trhof_hat, fhat)
	dot_g = np.dot(Trhof_hat, ghat)
	best_results['sens_f'] = sens_f
	best_results['sens_g'] = sens_g
	best_results['err_f'] = 0.5 * (1 - dot_f)
	best_results['err_g'] = 0.5 * (1 - dot_g)

	# use bool_func to generate some larger sample for entropy estimation
	n_samples_entropy = 2 ** n * 500
	lookup = np.array(list(itertools.product([0, 1], repeat=n)))  # all possible inputs
	all_XY = np.concatenate((lookup, bool_func), axis=1)  # shape: (2^n, n+1)
	Xy_entropy = all_XY[np.random.choice(all_XY.shape[0], size=n_samples_entropy, replace=True)]
	flips_entropy = np.random.binomial(1, bf_bool, size=(n_samples_entropy, n))
	flips_entropy = np.concatenate((flips_entropy, np.zeros((n_samples_entropy, 1))), axis=1) # don'tflip the labels
	Zy_entropy = np.logical_xor(Xy_entropy, flips_entropy).astype(int)
	H_YgivenZ, _, _ = empirical_entropy_estimate(Zy_entropy)
	H_YgivenZ = H_YgivenZ[0]
	best_results['entropy'] = H_YgivenZ
	# print(f"H_YgivenZ: {H_YgivenZ}, bf_bool: {bf_bool}, n: {n}")

	# wrap-up operations: save config as json, save hyper_config as json
	manager.save_configs(hyper_config)
	return best_results


def tune_hyperparameters_multiprocessing_sample_f(hyper_config, hyper_settings, config, logger):
	"""
	This version generates datasets on-the-fly using n_bool and bf_bool parameters.
	"""
	num_cpus=hyper_settings.get("total_cpus")
	num_gpus=hyper_settings.get("total_gpus") 
	if num_gpus != 0:
		raise NotImplementedError("GPU support not yet implemented")

	# Number of CPUs to use
	tot_cpus = mp.cpu_count()  # Get the number of available CPUs
	logger.debug(f"Number of available CPUs: {tot_cpus}")
	logger.debug(f"Number of CPUs to use: {num_cpus}")

	# build the hyperparameter space by sampling (gridsearch not yet supported)
	# Each element of this list is a dictionary of hyperparameters with the same 
	# keys as `hyper_config`
	hyper_list = []
	for i in range(hyper_settings.get("num_samples")):
		hyperparameter_slice = {}
		for key, value in hyper_config.items():
			hyperparameter_slice[key] = np.random.choice(value).item() # keep serializable
			# NEW FUNCTIONALITY: we will impose a different seed for each child, but make it reproducible.
			seed = i + 3048597713
			hyperparameter_slice['seed_bool'] = seed
		hyper_list.append(hyperparameter_slice)

	# Assemble local variables to be called from within a thread
	tune_directory = config.tune_directory
	threads_directory = os.path.join(tune_directory, "threads/")
	package_list = []
	paths_list = []
	for thread_id, hyperparameter_slice in enumerate(hyper_list):
		# Configure tuning path for this specific thread
		# note that the thread id does not belong to a specific thread, but rather to a job
		tune_path = os.path.join(threads_directory, f"job_{thread_id}") # where this thread's results live: .../threads/job_<thread_id>
		os.makedirs(tune_path)
		paths_list.append(tune_path)
		logger_name = f"thread_{thread_id}" # we cannot serialize a logger effectively, so we pass around its name
		thread_info = {"thread_id": thread_id, "tune_path": tune_path, "logger_name": logger_name}
		package = (hyperparameter_slice, config, thread_info)
		package_list.append(package)

	with mp.Pool(processes=num_cpus) as pool:
		# Map f to the list of parameters
		all_results = pool.map(train_model_multiprocessing_sample_f, package_list)
  
	#cleanup
	logger.debug("Cleaning up...")
	hyper_keys = list(hyper_config.keys())
	header_keys = get_header_sample_f().split(",")
	columns = header_keys + hyper_keys

	data = []
	for i in range(len(all_results)):
		hyper_setting = [hyper_list[i].get(k) for k in hyper_keys]
		best_result = [all_results[i].get(k) for k in header_keys]
		data.append(best_result + hyper_setting)
	df = pd.DataFrame(data, columns=columns)
	df.to_csv(os.path.join(tune_directory, f"{config.model_type}_sample_f_results.csv"), index=False)

	config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
	for k in hyper_keys:                                                                                     
		config_dict[k] = None
	with open(os.path.join(tune_directory, f"config.json"), "w") as f:
		f.write(json.dumps(config_dict))

	logger.debug("Tuning run completed successfully...")

