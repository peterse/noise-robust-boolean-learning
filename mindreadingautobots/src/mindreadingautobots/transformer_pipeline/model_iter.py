import ipdb as pdb
from time import time
from typing import OrderedDict
import numpy as np
import wandb

from mindreadingautobots.transformer_pipeline.components.transformer import TransformerCLF

from mindreadingautobots.utils.logger import store_results, print_log
import torch
import torch.nn as nn
from torch import optim

import ray
from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray import tune, train
from ray.train import RunConfig
from ray.train.torch import TorchTrainer, get_device
from ray.tune.experiment.trial import Trial
# from ray.util.check_serialize import inspect_serializability
from mindreadingautobots.utils.training import EarlyStopping
from mindreadingautobots.utils.helper import save_checkpoint

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


def train_model(model, train_loader, val_loader, noiseless_val_loader, voc, device, 
				config, logger, epoch_offset=0):

	max_val_acc = 0
	best_epoch = 0
	if config.wandb:
		wandb.watch(model, log_freq= 1000)

	itr= 0

	num_batches = int(train_loader.num_batches)
	early_stopping = EarlyStopping(patience=200, delta=0.0, logger=logger)

	for epoch in range(1, config.epochs + 1):
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		val_acc_epoch = 0.0
		model.train()
		start_time = time()
		lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']

		for batch, i in enumerate(range(0, len(train_loader), config.batch_size)):
			source, targets, word_lens = train_loader.get_batch(i)			
			source, targets, word_lens= source.to(device), targets.to(device), word_lens.to(device)
			loss = model.trainer(source, targets, word_lens, config)
			train_loss_epoch += loss 
			itr += 1
		
		train_loss_epoch = train_loss_epoch/train_loader.num_batches
		time_taken = time() - start_time
		time_mins = int(time_taken/60)
		time_secs= time_taken%60

		logger.debug('Training for epoch {} completed...\nTime Taken: {} mins and {} secs'.format(epoch, time_mins, time_secs))
		logger.debug('Starting Validation')

		val_acc_epoch = run_validation(config, model, val_loader, voc, device, logger)
		train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)

		# If noiseless validation is not enabled, the model will report '0'
		noiseless_val_acc_epoch = 0
		if config.noiseless_validation is not None:
			noiseless_val_acc_epoch = run_validation(config, model, noiseless_val_loader, voc, device, logger)

		gen_gap = train_acc_epoch- val_acc_epoch
		# Early stopping is based on _loss_, so we negate the accuracy
		early_stopping( (-1) * val_acc_epoch, model)

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
				"val_acc": val_acc_epoch,
				"noiseless_val_acc": noiseless_val_acc_epoch,
				"epoch": epoch,
			})

		if val_acc_epoch > max_val_acc :

			max_val_acc = val_acc_epoch
			best_epoch= epoch
			curr_train_acc= train_acc_epoch
			state = {
				'epoch' : epoch+epoch_offset,
				'model_state_dict' : model.state_dict(),
				'voc' : model.voc,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss': train_loss_epoch,
				'val_acc': max_val_acc,
				"noiseless_val_acc": noiseless_val_acc_epoch,
				'lr': lr_epoch
			}

			# save_checkpoint(state, epoch, logger, config.model_path, config.ckpt)  # Only save best model

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

	if config.wandb:
		wandb.log({
			'max-val-acc': max_val_acc,
			# 'gen-success': gen_success,
			# 'conv-time': conv_time,
			})
	
	if config.results:
		store_results(config, max_val_acc, curr_train_acc, best_epoch, noiseless_val_acc_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))

	
class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated
	

def build_and_train_model_raytune(hyper_config, config, train_loader, val_loader, noiseless_val_loader, voc, 
				logger, epoch_offset= 0):
	"""Build and train a model with the given hyperparameters
	
	Reasons why this exists:
	 - ray (or pickle) cannot serialize torch models
	 - We need to distribute the model training/building pipeline to multiple workers
	"""
	device = get_device()
	for key, value in hyper_config.items():
		setattr(config, key, value)
	model = build_model(config, voc, device, logger)
	train_model(model, train_loader, val_loader, noiseless_val_loader, voc, device, config, logger, epoch_offset)

def trial_dirname_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


def tune_model(hyper_settings, hyper_config, train_loader, val_loader, noiseless_val_loader, voc, 
				config, logger, epoch_offset= 0):	
	
	ray.init(include_dashboard=False) # suppress dashboard resources

	# config should have tune=True
	scheduler = ASHAScheduler(
		metric="val_acc",
		mode="max",
		max_t=hyper_settings.get("max_iterations"), # I'm not sure what this kwarg does and neither is the documentation
		grace_period=hyper_settings.get("grace_period"),
		reduction_factor=2)

	# reporter = CLIReporter(
	# 	metric_columns=["loss", "training_iteration", "mean_accuracy"],
	# 	print_intermediate_tables=False,
		# )
	reporter = TrialTerminationReporter()

	tune_config = tune.TuneConfig(
		num_samples=hyper_settings.get("num_samples"),
		trial_dirname_creator=trial_dirname_creator,
		scheduler=scheduler,
		max_concurrent_trials=hyper_settings.get("max_concurrent_trials"),
		)
		
	run_config = RunConfig(
		progress_reporter=reporter,
		stop={"training_iteration": config.epochs, "val_acc": 0.8},
	)
	trainable = tune.with_parameters(
					build_and_train_model_raytune, 
					config=config,
					train_loader=train_loader,
					val_loader=val_loader,
					noiseless_val_loader=noiseless_val_loader,
					voc=voc,
					logger=logger,
					epoch_offset=epoch_offset,
					)
	
	resources = tune.with_resources(
		trainable,
		{
			"cpu": hyper_settings.get("cpus_per_worker"), 
   			"gpu": hyper_settings.get("gpus_per_worker")},
	)
	
	tuner = Tuner(
		resources,
		param_space=hyper_config,
		tune_config=tune_config,
		run_config=run_config,
	)
	result = tuner.fit()

	df = result.get_dataframe()
	print(df)
	with open(config.hyper_path, 'a') as f:
		f.write(df.to_string(header=True, index=False))
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
            try:
                # Fetch batch data
                source, targets, word_lens = data_loader.get_batch(i)
                source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)

                # Evaluate the batch
                acc = model.evaluator(source, targets, word_lens, config)

                # Log individual batch accuracy
                logger.debug(f"Batch {batch_num}: Accuracy={acc}")
                val_acc_epoch += acc
                batch_num += 1
            except Exception as e:
                logger.error(f"Error during validation at batch {batch_num}: {e}")
                raise

    # Ensure all batches were processed
    if batch_num != data_loader.num_batches:
        logger.warning(
            f"Number of processed batches ({batch_num}) does not match total batches ({data_loader.num_batches})"
        )

    # Compute average validation accuracy
    val_acc_epoch = val_acc_epoch / max(1, data_loader.num_batches)
    logger.info(f"Validation completed: Average Accuracy={val_acc_epoch:.4f}")
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





	