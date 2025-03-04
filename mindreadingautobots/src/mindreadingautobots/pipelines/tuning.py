import os
import multiprocessing as mp
import numpy as np
import json
import datetime
import torch
import pandas as pd

from mindreadingautobots.pipelines.training import train_model, load_data, build_model
from mindreadingautobots.utils.logger import init_logger, get_logger


def make_tune_results_path(path):
    return os.path.join(path, "tune_results.csv")

def hyper_config_path(path):
    return os.path.join(path, f"hyper_config.json")

def get_header():
    return "epoch,train_loss,train_acc,val_acc,noiseless_val_acc,final_train_acc,final_val_acc,final_noiseless_val_acc,sensitivity"


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
			f.write(f"{epoch_results['epoch']},{epoch_results['train_loss']},{epoch_results['train_acc']},{epoch_results['val_acc']},{epoch_results['noiseless_val_acc']}\n")

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

	voc, train_loader, val_loader, noiseless_val_loader, noiseless_train_loader = load_data(config, manager.logger)
	model = build_model(config=config, voc=voc, device=device, logger=manager.logger)
	best_results = train_model(model, train_loader, val_loader, noiseless_val_loader, noiseless_train_loader, voc,
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

