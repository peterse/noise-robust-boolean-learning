#### Usage for different experiments

The set of command line arguments available can be seen in the respective `args.py` file. Here, we illustrate running the experiment for training Transformers on sparse parities. 


#### Training transformer on a specific dataset

From `transformer_pipeline`:

Run training for a transformer

```shell
$	python -m main -mode train -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 -run_name 0 \
-model_type SAN -depth 2 -heads 4 -lr 0.001 -d_model 32 -d_ffn 32 -noiseless_validation
```
gpu flags the gpu number, it is not a boolean. To use CPU, omit the gpu arg entirely. Other args:
 - depth (number of layers in decoder and encoder)
 - d_model (embedding size)
 - d_ffn (FFN hidden size)
 - heads (attention heads per layer)

#### Doing hyperparameter tuning with a specific dataset

WORK IN PROGRESS
Hyperparameters to sweep with raytune are set by modifiying `main.py` directly; it isn't worth the trouble to set up command line parsing to deal with all sorts of sweeps or grids or whatever.

To tune hyperparameters for an xformer do the following:
 - at COMMANDLINE: provide the hyperparameters that you want to remain fixed
 - in `main.py`: modify `hyper_config` with the sweep hyperparameters. ANYTHING HARDCODED IN `main.py` WILL OVERWRITE COMMANDLINE

```shell
<<<<<<< HEAD
$	python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234  \
=======
$	python -m main -mode tune -dataset hamilton_6_choose_4_nbits16_n2000_bf20_seed1234  \
>>>>>>> e8b97bc166dd0fd6f535c41241e4b8e04601ffc8
-model_type SAN 
```
 _if I have a d_ffn kwarg in `hyper_config`, the above "32" will be ignored!!




# # WORKSPACE, TEMPORARY TRASH STUFF

python -m main -mode train -gpu 0 -dataset hamilton_6_choose_4_nbits16_n2000_bf20_seed1234 -run_name 0 \
-model_type SAN -depth 2 -heads 4 -lr 0.001 -d_model 32 -d_ffn 32 