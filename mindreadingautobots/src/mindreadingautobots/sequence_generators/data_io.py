"""data_io.py - utilities for loadding/saving data."""
import os
import numpy as np
import pickle


def save_numpy_as_dict(data, data_path):
    """Save the output of a data generator as a set of dictionaries.
    
    Args:
        data (np.array): The data (n_data, n_bits) array
    """
    data_dict = {'line':[], 'label':[]}
    for i in range(data.shape[0]):
        data_dict['line'].append(''.join([str(x) for x in data[i,:-1]]))
        data_dict['label'].append(str(data[i][-1]))
    with open(data_path, 'wb') as handle:
        pickle.dump(data_dict, handle)

def load_dict_as_numpy(data_path):
    with open(data_path, 'rb') as handle:
        data_dct = pickle.load(handle)
    data = []
    for i in range(len(data_dct['line'])):
        x = [int(x) for x in data_dct['line'][i]]
        y = int(data_dct['label'][i])
        data.append(x + [y])
    data = np.array(data)
    return data