"""hyperparameters.py - Hyperparameter tuning for generic models"""
from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.train import RunConfig

def tune_hyperparameters(config, training_function, data, num_samples=10, max_num_epochs=10, n_cpus=1, gpus_per_trial=0):
    """Tune hyperparameters for a given model training routine.

    Several design decisions:
        - `data` will be passed directly as a serialized object. This incurs some bandwidth
            and some (de)serialization overhead per trial, but we don't know the scope
            of the HPC that we will be using, and our data will never be huge.
        - `training_function` should not refer to anything outside its function scope.
    
    Args:
        config: dictionary of ray[tune] hyperparameters to search over
        train_model: Signature of this: Input should be `config`, *args. We
            can pass any args by including them as kwargs in `tune.with_parameters`.
        num_samples: number of models to try for each hyperparameter configuration.
        max_num_epochs: maximum number of epochs to train for
        gpus_per_trial: number of GPUs to use per trial
    """
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])

    tune_config = tune.TuneConfig(
        num_samples=num_samples,
        scheduler=scheduler,
        )
        
    run_config = RunConfig(progress_reporter=reporter)
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
