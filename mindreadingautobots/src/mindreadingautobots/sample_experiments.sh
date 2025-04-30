# This script has a list of sample commands to launch experiments of each dataset

# Sparse Majority: 

# LSTM: 
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 -hyper_config_path ../../hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000

# Transformer:
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 -hyper_config_path ../../hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000

# To Report Sensitivity:
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf0_seed1234 -hyper_config_path ../../hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000 -sensitivity True

# Sparse Parity: 
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf0_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\  

# Hamiltonian: 
python -m main -mode tune -dataset hamilton_6_choose_6_nbits11_n5000_bf5_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000 
