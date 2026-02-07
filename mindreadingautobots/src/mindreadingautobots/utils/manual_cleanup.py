from mindreadingautobots.pipelines import tuning
import pandas as pd
import numpy as np
import json
import os

n_jobs = 30 # SET THIS NUMBER
data = []
header_keys = tuning.get_header().split(",")

targets = []
for i in range(n_jobs):
	fname = f"./threads/threads/job_{str(i)}/job_results.csv"
	hyper_path = f"./threads/threads/job_{str(i)}/hyper_config.json"
	# load the hyper config
	if not os.path.exists(hyper_path):
		continue
	targets.append((fname, hyper_path))
print("Found the following files to clean up:")
print("\n".join([x[0] for x in targets]))

for i, (fname, hyper_path) in enumerate(targets):
	with open(hyper_path, "r") as f:
		hyper_config = json.load(f)
	hyper_keys = list(hyper_config.keys())
	columns = header_keys + hyper_keys
	# load the results into pandas 
	df = pd.read_csv(fname)
	# get the row with the best validation accuracy
	best_row = df[df['val_acc'] == df['val_acc'].max()]
	# get the earliest epoch with the best validation accuracy
	best_row = best_row[best_row['epoch'] == best_row['epoch'].min()]
	# get a list of values "[epoch,train_loss,train_acc,val_acc,noiseless_val_acc]"
	best_result = best_row.values[0]
	# get the hyperparameters
	hyper_setting = [hyper_config.get(k) for k in hyper_keys]
	x = list(best_result) + list(hyper_setting)
	data.append(x)
	

df = pd.DataFrame(data, columns=columns)
abs_path = os.path.dirname(os.path.abspath(__file__))
print("saving to", os.path.join(abs_path, "tune_results.csv"))
df.to_csv(os.path.join(abs_path, "tune_results.csv"), index=False)
print("Done!")
	