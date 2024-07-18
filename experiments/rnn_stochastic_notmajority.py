import torch
from ray import tune

from mindreadingautobots.sequence_generators import make_datasets
from mindreadingautobots.models import rnn, hyperparameters  

def main():
    """Run the hyperparameter search for the RNN with this dataset"""
    # config = {
    #         "hidden_size": tune.choice([16, 32, 64]),
    #         "num_layers": tune.choice([1, 2, 3]),
    #         "lr": tune.loguniform(1e-4, 1e-1),
    #         "epochs": 100,
    #     }
    config = {
            "hidden_size": tune.choice([16, 32]),
            "num_layers": tune.choice([1 ]),
            "lr": [1e-4],
            "epochs": 100,
        }    
    
    # DATA LOADING
    seed = 334
    n_train = 200 
    n_data = int(n_train * 5/4) # downstream we have a 80/20 train/val split
    n_bits = 32 + 1
    k = 4
    noisy_not_majority_transition_matrix = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.95, 4: 0.95}
    X, _ = make_datasets.k_lookback_weight_dataset(noisy_not_majority_transition_matrix, k, n_data, n_bits, 0, seed)
    # Insert feature dimension (1 for scalar bits), and convert to tensor
    stochastic_majority_data = torch.tensor(X).float().unsqueeze(-1) 

    hyperparameters.tune_hyperparameters(
        config=config, 
        training_function=rnn.train_binary_rnn, 
        data=stochastic_majority_data, 
        num_samples=10, 
        gpus_per_trial=0
    )

    # TODO: enable multiprocessing


if __name__ == "__main__":
    main()