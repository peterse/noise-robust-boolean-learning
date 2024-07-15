from ray import tune

from mindreadingautobots import models    

def main():
    """Run the hyperparameter search for the RNN with this dataset"""
    config = {
            "hidden_size": tune.choice([16, 32, 64]),
            "num_layers": tune.choice([1, 2, 3]),
            "lr": tune.loguniform(1e-4, 1e-1)
        }


if __name__ == "__main__":
    main()