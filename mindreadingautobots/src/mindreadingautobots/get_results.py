import os
import json
import pandas as pd

def check_valid_run(result_path): 
  return os.path.exists(os.path.join(result_path, "config.json")) and any(fname.endswith('.csv') for fname in os.listdir(result_path))

tune_results_path = "/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots/tune_results/"  
# result_path = "/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots/tune_results/RNN_sparse_majority_k5_nbits41_n2000_bf10_seed1234/run_2025-02-12-15-17-10-398381" 

# threads_path = result_path + "/threads" 
data = []

def create_agg_results_csv(result_path, threads_path):
  num_set_hyperparameters = len(os.listdir(threads_path))
  for i in range(num_set_hyperparameters):
    job_folder = os.path.join(threads_path, f"job_{i}")
    hyper_config_path = os.path.join(job_folder, "hyper_config.json")
    job_results_path = os.path.join(job_folder, "job_results.csv")s
    
    if os.path.exists(hyper_config_path) and os.path.exists(job_results_path):
      with open(hyper_config_path, 'r') as f:
        hyper_params = json.load(f)
      
      job_results = pd.read_csv(job_results_path)
      
      # Find the row with the maximum val_acc
      max_valid_acc_row = job_results.loc[job_results['val_acc'].idxmax()]
      max_valid_acc_epoch = job_results['val_acc'].idxmax() + 1  # Adding 1 to convert zero-based index to epoch number

      
      # Get the final row (last epoch)
      final_row = job_results.iloc[-1]
      
      result = {
        'best_epoch': max_valid_acc_row['epoch'],
        **hyper_params, 
        'train_acc': max_valid_acc_row['train_acc'],
        'valid_acc': max_valid_acc_row['val_acc'],
        'train_loss': max_valid_acc_row['train_loss'],
        'final_train_acc': final_row['train_acc'],
        'final_valid_acc': final_row['val_acc'],
        'final_train_loss': final_row['train_loss']
      }
      
      data.append(result)

  df = pd.DataFrame(data)
  output_csv_path = os.path.join(result_path, "aggregated_results.csv")
  df.to_csv(output_csv_path, index=False) 

for model_dataset in os.listdir(tune_results_path):
  for run in os.listdir(os.path.join(tune_results_path, model_dataset)):
    run_path = os.path.join(tune_results_path, model_dataset, run)
    if check_valid_run(run_path):
      threads_path = os.path.join(run_path, "threads")
      create_agg_results_csv(run_path, threads_path)
      print(f"Created aggregated results for {run_path}")
    else:
      print(f"Skipping {run_path} as it is not a valid run")
