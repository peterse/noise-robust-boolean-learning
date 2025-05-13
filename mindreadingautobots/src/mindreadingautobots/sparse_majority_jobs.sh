cd mindreadingautobots/src/mindreadingautobots
conda activate autobots  

# if you just ssh into another CPU, here is the directory: (at least for ando) 
cd ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots
conda activate autobots  
python -m main -mode tune -dataset multitask_sparse_majority_ntasks20_ncontrol4_k3_ndata2000_bf5_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\ 
python -m main -mode tune -dataset sparse_majority_k5_nbits41_n2000_bf45_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\  

python -m main -mode tune -dataset counterexample100110_nbits20_n2000_bf20_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000 -sensitivity True 

python -m main -mode tune -dataset hamilton_6_choose_6_nbits11_n5000_bf5_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000 -sensitivity True 

python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf30_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\ 

python -m main -mode tune -dataset sparse_parity_k4_nbits10_n5000_bf20_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\ 



# lstm_sparse_parity_20_4_bf0, on CPU 154 
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf0_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 5\  

# lstm_sparse_parity_20_4_bf10, on CPU 154 
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf10_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\   

# lstm_sparse_parity_20_4_bf20, on CPU 158
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf20_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\  

# xformer_sparse_parity_20_4_bf0, on CPU 158
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf0_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\  


# xformer_sparse_parity_20_4_bf10, on CPU 149
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf10_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\  

# xformer_sparse_parity_20_4_bf20, on CPU 149
python -m main -mode tune -dataset sparse_parity_k4_nbits21_n5000_bf20_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\   

# ========================================
## FIXME retrain these guys, there were set to n = 5000 by mistake before 
# xformer_sparse_majority_20_5_bf5, on CPU 152 
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf5_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\

# xformer_sparse_majority_20_5_bf15, on CPU 155
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf15_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\

# xformer_sparse_majority_20_5_bf40, on CPU 150
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf40_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\

# xformer_sparse_majority_20_5_bf45, on CPU 147
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf45_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
-model_type SAN -noiseless_validation -epochs 1000\

# lstm_sparse_majority_20_5_bf5, on CPU 151
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf5_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\ 

# lstm_sparse_majority_20_5_bf15, on CPU 155
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf15_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\

# ========================================

# python -m main -mode tune -dataset sparse_majority_k5_nbits10_n200_bf10_seed1234 -hyper_config_path /u/mhzambia/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
# -model_type RNN -noiseless_validation -epochs 2 -epoch_report True -sensitivity True


############# XFORMERS #############

python -m main -mode tune -dataset counterexample000110000_nbits10_n10000_bf20_seed1234 -hyper_config_path /u/mhzambia/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
 -model_type SAN -noiseless_validation -epochs 1000 -epoch_report True -sensitivity True

################ RNNs ###############

python -m main -mode tune -dataset counterexample000110000_nbits14_n10000_bf20_seed1234 -hyper_config_path /u/mhzambia/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
 -model_type RNN -noiseless_validation -epochs 1000 -epoch_report True -sensitivity True

