
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
	
	# https://github.com/ray-project/ray/issues/30012#issuecomment-1305006855 I guess
	ray.init(
		include_dashboard=False, 
		  num_cpus=hyper_settings.get("total_cpus"), 
		  num_gpus=hyper_settings.get("total_gpus"), 
		  _temp_dir=None, 
		  ignore_reinit_error=True)
	# config should have tune=True
	scheduler = ASHAScheduler(
		time_attr="training_iteration",
		metric="val_acc",
		mode="max",
		max_t=hyper_settings.get("max_iterations"), # I'm not sure what this kwarg does and neither is the documentation
		grace_period=hyper_settings.get("grace_period"), 
		reduction_factor=2)

	reporter = CLIReporter(
		metric_columns=["loss", "training_iteration", "mean_accuracy"],
		print_intermediate_tables=False,
		)

	tune_config = tune.TuneConfig(
		num_samples=hyper_settings.get("num_samples"),
		scheduler=scheduler,
		trial_dirname_creator=trial_dirname_creator,
		max_concurrent_trials=hyper_settings.get("max_concurrent_trials"),
		)
		
	run_config = RunConfig(
		progress_reporter=reporter,
		# stop={"training_iteration": config.epochs, "val_acc": 0.80},
	)
	# # pdb.set_trace()
	# for v in [model, train_loader, val_loader, voc, device, config, logger]:
	# 	print("checking object", v)
	# 	print(inspect_serializability(v))
	# 	print()

	resources = tune.with_resources(
				tune.with_parameters(
					build_and_train_model_raytune, 
					config=config,
					train_loader=train_loader,
					val_loader=val_loader,
					noiseless_val_loader=noiseless_val_loader,
					voc=voc,
					logger=logger,
					epoch_offset=epoch_offset,
					),
				resources={"cpu": hyper_settings.get("cpus_per_worker"), "gpu": hyper_settings.get("gpus_per_worker")}
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



def build_and_train_model_raytune(hyper_config, config, train_loader, val_loader, noiseless_val_loader, voc, 
				logger, epoch_offset= 0):
	"""Build and train a model with the given hyperparameters
	
	Reasons why this exists:
	 - ray (or pickle) cannot serialize torch models
	 - We need to distribute the model training/building pipeline to multiple workers
	"""
	device = get_device()
	print("raytune device:", device)
	for key, value in hyper_config.items():
		setattr(config, key, value)
	model = build_model(config, voc, device, logger)
	train_model(model, train_loader, val_loader, noiseless_val_loader, voc, device, config, logger, epoch_offset)

def trial_dirname_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


def tune_model(hyper_settings, hyper_config, train_loader, val_loader, noiseless_val_loader, voc, 
				config, logger, epoch_offset= 0):	
	
	ray.init(
		include_dashboard=False, 
		  num_cpus=hyper_settings.get("total_cpus"), 
		  num_gpus=hyper_settings.get("total_gpus"), 
		  _temp_dir=None, 
		  ignore_reinit_error=True)
	
	
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

