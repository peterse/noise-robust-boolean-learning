import torch
from ray import tune

from mindreadingautobots.sequence_generators import make_datasets
from mindreadingautobots.models import decoder_transformer, hyperparameters


RUN_HYPERPARAMETER_SEARCH = False

def main():
    """Run the hyperparameter search for the RNN with this dataset"""
    
    # DATA LOADING
    seed = 334
    n_train = 1000
    n_data = int(n_train * 5/4) # downstream we have a 80/20 train/val split
    n = 40
    k = 4
    p_bitflip = 0.0
    raw_data = make_datasets.sparse_parity_k_n(n, k, n_data, p_bitflip)


    config = {"epochs": 3,
            "batch_size": 32,
            "device": torch.device("mps" if torch.backends.mps.is_available() else "cpu"), # NOTE: this is only for mac. For windows use cuda instead of mps.
            "lr": 1e-3,
            "context_size": 500,
            "vocab_size": 2,
            "n_layer": 2,
            "n_head": 2,
            "d_model": 16,
            "dropout": 0.1,
            "d_ff": 128,
            "activation": "relu",
            "standard_positional_encoding": False,
            "loss_type": "cross_entropy",
            "bias": True,
            "tie_weights": False,
            "embedding": "embedding",
            "mode": "encoder",
            }


    decoder_transformer.train_loop(config, raw_data)

    return 
    # HYPERPARAMETER SEARCH
    config = {
            "hidden_size": tune.choice([16, 32]),
            "num_layers": tune.choice([1, 2]),
            "lr": tune.loguniform(1e-4, 1e-3),
            "epochs": 100,
        } # NOTE: you can't just replace these with a list with a single element, or tune.choice([1])...
    
    hyperparameters.tune_hyperparameters(
        config=config, 
        training_function=rnn.train_binary_rnn, 
        data=stochastic_majority_data, 
        num_samples=10, 
        gpus_per_trial=1
    )

    # TODO: enable multiprocessing


if __name__ == "__main__":
    main()