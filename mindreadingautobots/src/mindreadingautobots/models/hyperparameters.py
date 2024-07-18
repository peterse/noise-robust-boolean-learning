"""hyperparameters.py - Hyperparameter tuning for generic models"""
from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.train import RunConfig

def tune_hyperparameters(config, training_function, data, num_samples=10, n_cpus=1, gpus_per_trial=0):
    """Tune hyperparameters for a given model training routine.

    Several design decisions:
        - `data` will be passed directly as a serialized object. This incurs some bandwidth
            and some (de)serialization overhead per trial, but we don't know the scope
            of the HPC that we will be using, and our data will never be huge.
        - `training_function` should not refer to anything outside its function scope.
    
    Args:
        config: dictionary of ray[tune] hyperparameters to search over. Contains
            kwargs for the model (training_function).
        train_model: Signature of this: Input should be `config`, *args. We
            can pass any args by including them as kwargs in `tune.with_parameters`.
        num_samples: number of models to try for each hyperparameter configuration.
        gpus_per_trial: number of GPUs to use per trial
    """

    # Notes about this scheduler:
    # - It will terminate iterations depending on how promising a hyperparameter set looks
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=config["epochs"], # I'm not sure what this kwarg does and neither is the documentation
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration", "mean_accuracy"],
        print_intermediate_tables=False,
        )

    tune_config = tune.TuneConfig(
        num_samples=num_samples,
        scheduler=scheduler,
        )
        
    run_config = RunConfig(
        progress_reporter=reporter,
        # stop={"training_iteration": config["epocs"], "mean_accuracy": 0.8},
    )
    resources = tune.with_resources(
                tune.with_parameters(training_function, data=data),
                resources={"cpu": n_cpus, "gpu": gpus_per_trial}
    )
    
    tuner = Tuner(
        resources,
        param_space=config,
        tune_config=tune_config,
        run_config=run_config,
    )
    result = tuner.fit()

    return result
