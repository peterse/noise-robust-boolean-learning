# FIXME: Broken, i don't know what is going on !!!!!!
#  import os 
# import csv

# tune_results_path = "/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/src/mindreadingautobots/tune_results"

# # for each `run` folder in a hyperparameter tuning result, check whether it is actually 
# # a finished run (not terminated with errors), to have this, this folder should contain 
# # a config.json, and a .csv file 
# def check_valid_run(result_path): 
#   return os.path.exists(os.path.join(result_path, "config.json")) and any(fname.endswith('.csv') for fname in os.listdir(result_path))

# # check if the csv already has three accuracy scores for the final epoch 
# # def check_csv_header(run_path): 
# #   csv_file = next((fname for fname in os.listdir(run_path) if fname.endswith('.csv')), None)
# #   if not csv_file:
# #     return False

#   # csv_path = os.path.join(run_path, csv_file)
#   # with open(csv_path, 'r') as f:
#   #   header = f.readline().strip().split(',')
#   #   required_columns = {'final_train_acc', 'final_val_acc', 'final_noiseless_val_acc'}
#   #   return required_columns.issubset(header)

# for result in os.listdir(tune_results_path):
#   model_dataset_path = result 
#   result_path = os.path.join(tune_results_path, result) 
#   print(f"Looking at {result_path}") 
#   for run in os.listdir(result_path): 
#     run_name = model_dataset_path + "/" + run 
#     run_path = os.path.join(result_path, run)
#     print("======================================")
#     print(f"Checking {run_name}")
#     if not check_valid_run(run_path): 
#       print(f"Skipping {run_path} as it is not a valid run")
#       continue
#     threads_path = os.path.join(run_path, "threads")
#     if not os.path.exists(threads_path):
#       print(f"Skipping {run_path} as it does not contain a threads folder")
#       continue

#     output_csv_path = os.path.join(run_path, f"{model_dataset_path.replace}_add_final.csv")
#     with open(output_csv_path, 'w', newline='') as output_csv_file:
#       csv_writer = csv.writer(output_csv_file)
#       header_written = False

#       for job_folder in os.listdir(threads_path):
#         job_folder_path = os.path.join(threads_path, job_folder)
#         job_results_path = os.path.join(job_folder_path, "job_results.csv")
        
#         if not os.path.exists(job_results_path):
#           print(f"Skipping {job_folder_path} as it does not contain job_results.csv")
#           continue

#         with open(job_results_path, 'r') as job_csv_file:
#           csv_reader = csv.DictReader(job_csv_file)
#           rows = list(csv_reader)
          
#           if not rows:
#             print(f"Skipping {job_results_path} as it is empty")
#             continue

#           max_val_acc_row = max(rows, key=lambda row: float(row['val_acc']))
#           final_row = rows[-1]

#           if not header_written:
#             header = list(max_val_acc_row.keys()) + ['final_val_acc', 'final_noiseless_val_acc', 'final_train_acc']
#             csv_writer.writerow(header)
#             header_written = True

#           max_val_acc_row['final_val_acc'] = final_row['val_acc']
#           max_val_acc_row['final_noiseless_val_acc'] = final_row['noiseless_val_acc']
#           max_val_acc_row['final_train_acc'] = final_row['train_acc']
#           csv_writer.writerow(max_val_acc_row.values())


    
    

  