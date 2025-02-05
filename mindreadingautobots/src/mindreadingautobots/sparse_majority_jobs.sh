# python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 \
# -model_type SAN -noiseless_validation -epochs 200\

# python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf10_seed1234 \
# -model_type SAN -noiseless_validation -epochs 400\

# python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf20_seed1234 \
# -model_type SAN -noiseless_validation -epochs 400\

# python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 \
# -model_type RNN -noiseless_validation -epochs 100\

# python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf10_seed1234 \
# -model_type RNN -noiseless_validation -epochs 300\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf20_seed1234 \
-model_type RNN -noiseless_validation -dropout 0.05 -epochs 400\