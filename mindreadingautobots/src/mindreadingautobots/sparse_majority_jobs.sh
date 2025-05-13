cd mindreadingautobots/src/mindreadingautobots
conda activate autobots  

# if you just ssh into another CPU, here is the directory: (at least for ando) 
cd ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots
conda activate autobots

# python -m main -mode tune -dataset sparse_majority_k5_nbits10_n200_bf10_seed1234 -hyper_config_path /u/mhzambia/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
# -model_type RNN -noiseless_validation -epochs 2 -epoch_report True -sensitivity True


############# XFORMERS #############

python -m main -mode tune -dataset counterexample000110000_nbits10_n10000_bf20_seed1234 -hyper_config_path /u/mhzambia/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/xformer_hyper_config.yaml \
 -model_type SAN -noiseless_validation -epochs 1000 -epoch_report True -sensitivity True

################ RNNs ###############

python -m main -mode tune -dataset counterexample000110000_nbits14_n10000_bf20_seed1234 -hyper_config_path /u/mhzambia/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
 -model_type RNN -noiseless_validation -epochs 1000 -epoch_report True -sensitivity True

