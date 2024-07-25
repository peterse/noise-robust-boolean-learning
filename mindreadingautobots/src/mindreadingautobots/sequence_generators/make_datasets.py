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
        Z = np.logical_xor(X, flips).astype(int)

    return X, Z


def k_lookback_weight_dataset_mixed_genfuncs(k, n_data, n_bits, p_bitflip, seed):
    """Create a dataset generated from a uniform mixture of generating funcs

    For simplicity, we will not deal with stochastic generating functions.
    """
    # Compute how much of each function to use, dealing with remainders
    # TODO:
    raise NotImplementedError
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


def not_majority_4lookback_nondeterministic(n_data, n_bits, nondeterm, seed):
    """Generate nondeterministic NOT-MAJORITY forecasting data with n_bits bits.

    the `nondeterm` parameter is the probability of bitflipping a bit during 
    sequence generation. This is not equivalent to the `p_bitflip` parameter.

    We break ties by rounding up: majority of 1100 is 1.
    
    """
    transition_matrix = {0: 1 - nondeterm, 1: 1 - nondeterm, 2: nondeterm, 3: nondeterm, 4: nondeterm}
    return k_lookback_weight_dataset(transition_matrix, 4, n_data, n_bits, 0, seed) 


def parity_4lookback(n_data, n_bits, p_bitflip, seed):
    """Generate PARITY forecasting data with n_bits bits.
    
    """
    transition_matrix = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    return k_lookback_weight_dataset(transition_matrix, 4, n_data, n_bits, p_bitflip, seed) 


def parity_4lookback_nondeterministic(n_data, n_bits, nondeterm, seed):
    """Generate PARITY forecasting data with n_bits bits.
    
    the `nondeterm` parameter is the probability of bitflipping a bit during 
    sequence generation. This is not equivalent to the `p_bitflip` parameter.

    """
    transition_matrix = {0: nondeterm, 1: 1 - nondeterm, 2: nondeterm, 3: 1 - nondeterm, 4: nondeterm}
    return k_lookback_weight_dataset(transition_matrix, 4, n_data, n_bits, 0, seed) 



def sparse_parity_k_n(n_bits, k, n_data, p_bitflip=0.0, seed=0):

    """Generate a dataset where the final bit is the parity of a subset k of the n bits.
    
    Args:
        n_bits: TOTAL number of bits (including final bit)
        k: number of bits in the subset, to be chosen randomly.
        n_data: number of data points
        p_bitflip: probability of flipping a bit of INPUT data - 
            We do not apply label noise with this bitflip!

    returns:
        X: (n_data, n_bits) array of noiseless data
        Z: (n_data, n_bits) array of noisy data, or None if p_bitflip is 0

    """
    np.random.seed(seed)
    # TODO: add p_bitflip
    subseq_idx = np.random.choice(np.arange(n_bits - 1), k, replace=False)
    
    in_subset = np.zeros(n_bits-1, dtype=np.bool_)
    in_subset[subseq_idx] = 1
    X = np.random.randint(0, 2, size=(n_data, n_bits))

    for i in range(n_data):
        seq = X[i]
        if np.sum(seq[subseq_idx]) % 2 == 0:
            X[i, -1] = 0
        else:
            X[i, -1] = 1

    Z = None
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        flips[:,-1] = 0
        Z = np.logical_xor(X, flips).astype(int)

    return X, Z, subseq_idx


def sparity_k4(n_data, n_bits, p_bitflip, seed):
    return sparse_parity_k_n(n_bits, 4, n_data, p_bitflip, seed)


def sparse_not_majority_k_n(n, k, n_data, p_bitflip=0.0):
    """Generate a dataset where the final bit is a function of a subset k of the n bits.
    
    Args:
        n: number of bits
        k: number of bits in the subset, to be chosen randomly.
        n_data: number of data points
        p_bitflip: probability of flipping a bit
    """
    pass