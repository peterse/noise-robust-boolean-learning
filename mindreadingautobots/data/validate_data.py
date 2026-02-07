# This file checks whether noiseless data is different from noisy data. 
import os 
import pickle

data_dir = '/u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/data'
def compare_pickles(file1, file2):
  with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)
    return data1 == data2

for folder in os.listdir(data_dir):
  folder_path = os.path.join(data_dir, folder)
  print(folder_path)
  if os.path.isdir(folder_path):
    train_file = os.path.join(folder_path, 'train.pkl')
    noiseless_train_file = os.path.join(folder_path, 'noiseless_train.pkl')
    if os.path.exists(train_file) and os.path.exists(noiseless_train_file):
      result = compare_pickles(train_file, noiseless_train_file)
      print(f'{folder}: {"Same" if result else "Different"}')