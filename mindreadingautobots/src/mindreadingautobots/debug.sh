# python -m main -mode train -gpu 0 -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 \
# -model_type RNN -cell_type RNN_TANH -depth 1 -lr 0.001 -noiseless_validation -epochs 2\

# python -m main -mode train -gpu 0 -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234  \
# -model_type SAN -depth 3 -d_model 32 -d_ffn 32 -heads 4 -lr 0.003 -noiseless_validation -epochs 400\


# python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 \
# -model_type RNN -noiseless_validation -epochs 2\

python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf0_seed1234 \
-model_type SAN -lr 0.003 -noiseless_validation -epochs 2\

# TODO:
# clean up readmes