import numpy as np
import itertools

def maj(b, n):
    return 1 if np.sum(b) > n/2 else 0

def parity(b):
    return np.sum(b) % 2

def bool_func_from_signature(n, signature):
    """
    A signature of a boolean function is a length 2^n array that assigns
    a 0 or 1 to each input bitstring. The array is shape (2, )^n and the 
    value at each position is 0 or 1.
    """
    pass

def s(f, x):
    """Compute the sensitivity of f at x"""
    n = len(x)
    out = 0
    for i in range(n):
        x_ = x.copy()
        x_[i] = 1 - x_[i]
        if f(x) != f(x_):
            out += 1
    return out

def average_sensitivity(f, X_arr):
    """scaling of n2^n"""
    n = len(X_arr[0])
    out = 0
    for x in X_arr:
        out += s(f, x)
    return out / len(X_arr)

def s_i(f, x, i):
    """Compute the sensitivity of f at x with respect to i"""
    x_ = x.copy()
    x_[i] = 1 - x_[i]
    return 1 if f(x) != f(x_) else 0

def average_s_i(f, i, X_arr, verbose=False):
    """Compute the average sensitivity of f with respect to i"""
    out = 0
    for x in X_arr:
        out += s_i(f, x, i)
        if verbose:
            print(f"i: {i}, x: {x}, s_i: {s_i(f, x, i)}")
    return out / len(X_arr) if len(X_arr) > 0 else 0


def bits_to_idx(b):
    out = 0
    for bit in b:
        out = (out << 1) | bit
    return out

def idx_to_bits(x, n):
    if x == 0: 
        return [0] * n
    bit = []
    while x:
        bit.append(x % 2)
        x >>= 1
    return (bit + [0] * (n-len(bit)))[::-1]


def generate_noisy_distr(n, p, f):
    """For a true label function f, generate a distribution p_{Z, Y}
    where Y = f(X) and Z is bitflipped X with prob p.

    ALWAYS ASSUMES UNIFORM DISTR FOR X

    we will return P_{Z, Y}(z, y) with shape (2**n, 2)
    """
    out = np.zeros((2 ** n, 2))
    for i, x in enumerate(itertools.product([0, 1], repeat=n)):
        p_x = 1 / 2 ** n
        y = int(f(x))
        for e in itertools.product([0, 1], repeat=n):
            p_z_given_x = p ** np.sum(e) * (1 - p) ** (n - np.sum(e))
            z = np.array(x) ^ np.array(e)
            z_idx = bits_to_idx(z)
            out[z_idx, y] += p_z_given_x * p_x

    return out

def compute_acc_noisytest(p_zy, func, n):
    """Compute the theoretical accuracy of `func` on p_ZY, i.e.

        Pr_{ZY}(func(Z) = Y

    So this evaluates the noisy validation accuracy of func.
    As before, assumes uniform distribution for X.
    """
    
    acc = 0
    for i, p_zy in enumerate(p_zy):
        # p_z = np.sum(p_zy)
        bitstring = idx_to_bits(i, n)
        pred = func(bitstring)
        acc += p_zy[pred]

    return acc

def compute_acc_test(func, true_func, n):
    """Compute the accuracy of `func` on the true distribution of X, i.e.
        Pr_{X}(func(X) = Y)

    So this evaluates the validation accuracy of func.
    As before, assumes uniform distribution for X.    
    """
    acc = 0
    for i, x in enumerate(itertools.product([0, 1], repeat=n)):
        truth = true_func(x)
        pred = func(x)
        if truth == pred:
            acc += 1 / 2 ** n
    return acc

def boolean_function_from_signature(f):
    """Given a length 2^n binary array, return the function with that boolean signature."""
    X_arr = list(itertools.product([0, 1], repeat=n))
    X_arr = [tuple(x) for x in X_arr]
    lookup = dict(zip(X_arr, f))
    def func(x):
        return lookup[tuple(x)]
    return lookup, func

def random_boolean_function(n):
    X_arr = list(itertools.product([0, 1], repeat=n))
    X_arr = [tuple(x) for x in X_arr]
    f = np.random.randint(2, size=2**n, dtype=int)
    f = [x.item() for x in f]
    lookup = dict(zip(X_arr, f))
    def func(x):
        return lookup[tuple(x)]
    return lookup, func


def dataset_lookup_table(dataset):
    """
    Given a dataset X with N rows and the last column being labels, all bit-valued, create a lookup table:
    The lookup table is a dictionary where each key is a unique row in X, and the value
    is the most common label associated with that. This means that for every unique row, 
    we must calculate which label was more likely.
    
    Args:
        dataset (np.array): The dataset with shape (N, n+1) where the last column is the label
        
    Returns:
        function: A function that takes in features and returns the most common label
    """
    # Extract features (all columns except the last one) and labels (last column)
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    
    # Create a dictionary to store counts of each label for each unique feature row
    label_counts = {}
    
    # Iterate through the dataset
    for i in range(len(dataset)):
        # Convert feature row to tuple so it can be used as dictionary key
        feature_tuple = tuple(features[i])
        label = labels[i]
        
        # Initialize counter for this feature if it doesn't exist
        if feature_tuple not in label_counts:
            label_counts[feature_tuple] = {}
            
        # Increment count for this label
        if label not in label_counts[feature_tuple]:
            label_counts[feature_tuple][label] = 0
        label_counts[feature_tuple][label] += 1
    
    # Create the final lookup table with the most common label for each feature
    lookup_table = {}
    for feature, counts in label_counts.items():
        # Find the label with the highest count
        most_common_label = max(counts.items(), key=lambda x: x[1])[0]
        lookup_table[feature] = most_common_label
    
    # now, convert the lookup table to a function
    def lookup_func(x):
        return lookup_table[tuple(x)]
    return lookup_func



