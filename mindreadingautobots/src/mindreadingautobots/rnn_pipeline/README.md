#### Usage for different experiments

The set of command line arguments available can be seen in the respective `args.py` file. Here, we illustrate running the experiment for training Transformers on sparse parities. Follow the same methodology for running any experiments with LSTMs.


#### Training a specific rnn on a specific dataset

From `rnn_pipeline`:

Run training for a vanilla RNN (or, switch cell_type to LSTM). 
```shell
$	python -m main -mode train -gpu 0 -dataset sparse_parity_k4_n5000_bf0_seed1234 -run_name 0 \
-cell_type RNN -depth 1 -lr 0.001 -emb_size 128 -hidden_size 128 \
```
gpu flags the gpu number, it is not a boolean. To use CPU, omit the gpu arg entirely.

#### Doing hyperparameter tuning with a specific dataset

Hyperparameters to sweep with raytune are set by modifiying `main.py` directly; it isn't worth the trouble to set up command line parsing to deal with all sorts of sweeps or grids or whatever.

To tune hyperparameters for an LSTM on sparse parity dataset (provide the hyperparameters that you want to remain fixed; anything that you have coded into main.py will overwrite anything that you pass as a command line argument)
```shell
$	python -m main -mode tune -dataset sparse_parity_k4_n5000_bf0_seed1234  \
-cell_type LSTM -lr 0.001 -emb_size 128 -hidden_size 128 
```

