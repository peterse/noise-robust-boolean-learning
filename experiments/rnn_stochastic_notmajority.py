from ray import tune

from mindreadingautobots.sequence_generators import make_datasets
from mindreadingautobots.models import rnn, hyperparameters  

def main():
    """Run the hyperparameter search for the RNN with this dataset"""
    config = {
            "hidden_size": tune.choice([16, 32, 64]),
            "num_layers": tune.choice([1, 2, 3]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "epochs": 10,
        }
    
    
    # TODO
    stochastic_majority_data = None

    hyperparameters.tune_hyperparameters(
        config=config, 
        training_function=rnn.train_binary_rnn, 
        data=stochastic_majority_data, 
        num_samples=10, 
        max_num_epochs=10, 
        gpus_per_trial=0
    )

    # TODO: enable multiprocessing


if __name__ == "__main__":
    main()