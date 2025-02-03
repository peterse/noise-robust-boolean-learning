# python -m main -mode train -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 -run_name 0 \
# -model_type RNN -cell_type RNN_TANH -depth 1 -lr 0.001 -noiseless_validation -epochs 2\

# python -m main -mode train -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 -run_name 0 \
# -model_type SAN -depth 1 -lr 0.001 -noiseless_validation -epochs 2\

# python -m main -mode tune -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 -run_name 0 \
# -model_type RNN -cell_type RNN_TANH -noiseless_validation -epochs 2\

python -m main -mode tune -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 -run_name 0 \
-model_type SAN -lr 0.003 -noiseless_validation -epochs 2\

# TODO:
# - get save destinations and directories working
# - multiprocessing from mldec
# - remove pipeline folders
# some kind of debug?
# clean up readmes