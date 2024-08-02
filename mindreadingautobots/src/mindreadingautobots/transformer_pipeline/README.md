#### Usage for different experiments

The set of command line arguments available can be seen in the respective `args.py` file. Here, we illustrate running the experiment for training Transformers on sparse parities. 


#### Training transformer on a specific dataset

From `rnn_pipeline`:

Run training for a transformer

```shell
$	python -m main -mode train -gpu 0 -dataset sparse_parity_k4_nbits20_n5000_bf0_seed1234 -run_name 0 \
-model_type SAN -depth 2 -heads 4 -lr 0.001 -d_model 32 -d_ffn 32 
```
gpu flags the gpu number, it is not a boolean. To use CPU, omit the gpu arg entirely. Other args:
 - depth (number of layers in decoder and encoder)
 - d_model (embedding size)
 - d_ffn (FFN hidden size)
 - heads (attention heads per layer)

#### Doing hyperparameter tuning with a specific dataset

WORK IN PROGRESS
Hyperparameters to sweep with raytune are set by modifiying `main.py` directly; it isn't worth the trouble to set up command line parsing to deal with all sorts of sweeps or grids or whatever.

To tune hyperparameters for an LSTM on sparse parity dataset (provide the hyperparameters that you want to remain fixed; anything that you have coded into main.py will overwrite anything that you pass as a command line argument)

```shell
$	python -m main -mode tune -dataset sparse_parity_k4_n5000_bf0_seed1234  \
-cell_type LSTM -lr 0.001 -emb_size 128 -hidden_size 128 
```

