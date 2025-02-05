cd mindreadingautobots/src/mindreadingautobots
conda activate autobots 

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 \
-model_type SAN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf10_seed1234 \
-model_type SAN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf20_seed1234 \
-model_type SAN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 \
-model_type RNN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf10_seed1234 \
-model_type RNN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf20_seed1234 \
-model_type RNN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf20_seed1234 \
-model_type RNN -noiseless_validation -epochs 1000\

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf35_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\ 

python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf20_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\ 

