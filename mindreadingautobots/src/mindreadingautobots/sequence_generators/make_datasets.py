"""make_datasets_final.py - Generate the datasets used for Part 1 of the analysis"""
import itertools
import numpy as np


def k_lookback_weight_dataset(transition_matrix, k, n_data, n_bits, p_bitflip, seed):
    """Abstract function for _specific_ k-lookback boolean function.
    
    Args:
        transition_matrix: dict with (k+1) entries, corresponding to probability of 1
                           given the weight of the previous k bits. keys are integers
                           values are probabilities

    returns:
        X: (n_data, n_bits) array of noiseless data
        Z: (n_data, n_bits) array of noisy data, or None if p_bitflip is 0
    """
    assert len(transition_matrix) == k + 1
    assert np.all([0 <= v <= 1 for v in transition_matrix.values()])
    assert n_bits > k
    
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(n_data, n_bits))
    Z = None
    for i in range(n_data):
        for j in range(k, n_bits):
            weight = np.sum(X[i, j-k:j])
            X[i, j] = np.random.binomial(1, transition_matrix[weight])
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        Z = np.logical_xor(X, flips)

    return X, Z


def k_lookback_weight_dataset_mixed_genfuncs(k, n_data, n_bits, p_bitflip, seed):
    """Create a dataset generated from a uniform mixture of generating funcs

    For simplicity, we will not deal with stochastic generating functions.
    """
    # Compute how much of each function to use, dealing with remainders
    # TODO:
    n_by_func = None
    # Iterate over all generating funcs uniformly:
    for signature in itertools.combinations([0, 1], k+1):
        transition_matrix = dict(zip(range(k+1), signature))
        new = k_lookback_weight_dataset(transition_matrix, k, n_by_func, n_bits, p_bitflip, seed)
        out = np.vstack((out, new))

    return out

def not_majority_4lookback(n_data, n_bits, p_bitflip, seed):
    """Generate NOT-MAJORITY forecasting data with n_bits bits.

    We break ties by rounding up: majority of 1100 is 1.
    
    """
    transition_matrix = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0}
    return k_lookback_weight_dataset(transition_matrix, 4, n_data, n_bits, p_bitflip, seed) 


def parity_4lookback(n_data, n_bits, p_bitflip, seed):
    """Generate PARITY forecasting data with n_bits bits.
    
    """
    transition_matrix = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    return k_lookback_weight_dataset(transition_matrix, 4, n_data, n_bits, p_bitflip, seed) 





def sparse_classification_k_n(n, k, n_data, p_bitflip):
    """Generate a dataset where the final bit is a function of a subset k of the n bits.
    
    Args:
        n: number of bits
        k: number of bits in the subset, to be chosen randomly.
        n_data: number of data points
        p_bitflip: probability of flipping a bit
    """

def sparse_parity_4_40(n_data, p_bitflip):
    """Generate SPARSE PARITY classification data with 40 bits.
    
    """
    pass


def sparse_not_majority_4_40(n_data, p_bitflip):
    """Generate SPARSE NOT-MAJORITY classification data with 40 bits.
    
    """
    pass