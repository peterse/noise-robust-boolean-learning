#### Usage for different experiments

The set of command line arguments available can be seen in the respective `args.py` file.


#### Training a specific rnn on a specific dataset

From `rnn_pipeline`:

Run training for a vanilla RNN (or, switch cell_type to LSTM):

```shell
$	python -m main -mode train -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 -run_name 0 \
-model_type RNN -cell_type RNN_TANH -depth 1 -lr 0.001 -emb_size 128 -hidden_size 128 -noiseless_validation \
```
gpu flags the gpu number, it is not a boolean. To use CPU, omit the gpu arg entirely. Other arguments
 - cell_type determines which kind of RNN, choices= ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']
 - depth (number of rnn hidden layers)
 - emb_size (embedding dimension for inputs)
 - hidden size (hidden units per layer)

#### Doing hyperparameter tuning with a specific dataset

Hyperparameters to sweep with raytune are set by modifiying `main.py` directly; it isn't worth the trouble to set up command line parsing to deal with all sorts of sweeps or grids or whatever.

To tune hyperparameters for an LSTM on sparse parity dataset (provide the hyperparameters that you want to remain fixed; anything that you have coded into main.py will overwrite anything that you pass as a command line argument)

```shell
$	python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234  \
-cell_type LSTM -lr 0.001 -emb_size 128 -hidden_size 128 
```


# workspace, temp, trash
$ python -m main -mode train -gpu 0 -dataset hamilton_6_choose_4_nbits16_n2000_bf20_seed1234 -run_name 0 \
-model_type RNN -cell_type LSTM -depth 1 -lr 0.001 -emb_size 128 -hidden_size 128 


$	python -m main -mode tune -dataset hamilton_6_choose_4_nbits16_n2000_bf20_seed1234 -cell_type LSTM | hyperparameters.log