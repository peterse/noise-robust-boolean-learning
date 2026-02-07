import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Classifier')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test', 'tune'], help='Modes: train, test, tune')
	# parser.add_argument('-debug', action='store_true', help='Operate on debug mode')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-noiseless_validation', dest='noiseless_validation', action='store_true', help='Compute validation score for noiseless data')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)
	parser.add_argument('-savei', dest='savei', action='store_true', help='save models in intermediate epochs')
	parser.add_argument('-no-savei', dest='savei', action='store_false', help='Do not save models in intermediate epochs')
	parser.set_defaults(savei=False)

	# parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='sparity40_5k', help='Dataset')

	# parser.add_argument('-itr', dest='itr', action='store_true', help='Iteratively train')
	parser.add_argument('-no-itr', dest='itr', action='store_false', help='Train epochwise on fixed dataset')
	parser.set_defaults(itr=False)
	# Device Configuration
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=1729, help='Default seed to set')
	
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	
	# Dont modify ckpt_file
	# If you really want to then assign it a name like abc_0.pth.tar (You may only modify the abc part and don't fill in any special symbol. Only alphabets allowed
	# parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')

	# Model parameters
	parser.add_argument('-model_type', type=str, choices= ['RNN', 'SAN'],  help='Model Type')

	# SHARED PARAMETERS
	parser.add_argument('-depth', type=int, help='Number of layers (encoder and decoder layers for SAN, just depth for RNN)')
	parser.add_argument('-dropout', type=float, help= 'Dropout probability for input/output/state units (0.0: no dropout)')

	# RNN PARAMETERS: 
	parser.add_argument('-cell_type', type=str, choices= ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'],  help='RNN cell type, default: lstm')
	parser.add_argument('-emb_size', type=int, help='Embedding dimensions of inputs')
	parser.add_argument('-hidden_size', type=int, help='Number of hidden units in each layer')
	parser.add_argument('-tied', dest='tied', action='store_true', help='Tied Weights in input and output embeddings')
	parser.add_argument('-no-tied', dest='tied', action='store_false', help='Without Tied Weights in input and output embeddings')
	parser.set_defaults(tied=False)

	# TRANSFORMER PARAMETERS
	parser.add_argument('-d_model', type=int, help='Embedding size in Transformer')
	parser.add_argument('-d_ffn', type=int, help='Hidden size of FFN in Transformer')
	parser.add_argument('-heads', type=int, help='Number of Attention heads in each layer')
	parser.add_argument('-pos_encode', default='learnable', choices= ['absolute','learnable'], help='Type of position encodings')
	parser.add_argument('-mask', dest='mask', action='store_true', help='Pos Mask')
	parser.add_argument('-no-mask', dest='mask', action='store_false', help='Do not Pos Mask')
	parser.set_defaults(mask=False)


	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')

	# OPTIMIZATION
	parser.add_argument('-lr', type=float, help='Learning rate')
	parser.add_argument('-decay_patience', type=int, default=3, help='Wait before decaying learning rate')
	parser.add_argument('-decay_rate', type=float, default=0.2, help='Amount by which to decay learning rate on plateu')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('-epochs', type=int, default=1000, help='Maximum # of training epochs')
	parser.add_argument('-iters', type=int, default=40000, help='Maximum # of training iterations in iter mode')
	parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
	
	# REGULARIZATION
	parser.add_argument('-lambda_sens', type=float, default=0, help='Regularization strength')
	parser.add_argument('-loss', type=str, default='default', choices=['default', 'sensitivity_reg'], help='Regularization scheme function')
	parser.add_argument('-reg_samples_per_batch', type=int, default=32, help='Number of samples to use for sensitivity computation')


	# Wandb
	parser.add_argument('-project', type=str, default='Bool', help='wandb project name')
	parser.add_argument('-entity', type=str, default='your_entity', help='wandb entity name') 

	# YAML configuration 
	parser.add_argument('-hyper_config_path', type=str, help='Path to hyperparameter configuration yaml file', default='mindreadingautobots/hyper_config/hyper_config.yaml')
  
	# Do you want sensitivity to be computed ? 
	parser.add_argument('-sensitivity', type=bool, default=False, help='if true, the sensitivity will be computed whenever the validation score is improved')
	parser.add_argument('-epoch_report', type=bool, default=False, help='if true, the epoch results will be reported every epoch, and will be saved in job_results.csv for each job, the sensitivity will be computed every epoch as well')
	
	# Add sample_f specific arguments
	parser.add_argument('-n_bool', type=int, required=False, help='Number of boolean variables used to compute function')
	parser.add_argument('-n_bits', type=int, required=False, help='Number of bits in the dataset')
	parser.add_argument('-bf_bool', type=float, required=False, help='Boolean function parameter (float value)')
	parser.add_argument('-seed_bool', type=int, required=False, help='Seed for boolean function generation')
	parser.add_argument('-n_samples_train', type=int, required=False, help='Number of samples to generate for training')
	parser.add_argument('-n_samples_val', type=int, required=False, help='Number of samples to generate for validation')
	parser.add_argument('-n_samples_test', type=int, required=False, help='Number of samples to generate for testing')
	return parser