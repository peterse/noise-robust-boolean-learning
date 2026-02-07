"""make_datasets_final.py - Generate the datasets used for Part 1 of the analysis"""
import itertools
import numpy as np
import networkx as nx
import itertools

def k_choose_m_transition_forecast_dataset(transition_func, k, m, n_data, n_bits, p_bitflip, seed, subseq_idx):
    """Generate a dataset of forecast data with a k-choose-m-hamilton scheme.

    The scheme is as follows:
    - The first k bits are randomly generated
    - A subset of m bits is chosen randomly
    - Each subsequent bit is determined by transition_func acting on a length-m prefix;
        the length-m prefix is the size-m subsequence of the k previous bits
    - If bitflip is nonzero, all the bits in the range [k:-1] are flipped with probability p_bitflip.
        The final bit is not flipped and the seed bits are not flipped.


    Note: We do not remove nor noise up the seed bits, as these serve as a kind of 'function label' for 
        the rest of the sequence
    
    Args:
        transition_func: A function with inputs as int-sequences of length m and return value of 0 or 1
        n_bits: TOTAL number of bits (including final bit)
        k: amount of lookback
        m: number of bits in the subset of lookback, to be chosen randomly.
        n_data: number of data points
        p_bitflip: probability of flipping a bit of INPUT data - 
            We do not apply label noise with this bitflip!

    returns:
        X: (n_data, n_bits) array of noiseless data
        Z: (n_data, n_bits) array of noisy data, or None if p_bitflip is 0

    """
    np.random.seed(seed)

    assert len(subseq_idx) == m
    assert n_bits > k
    assert n_data > 2 ** k

    subseq_idx = np.sort(np.array(subseq_idx))

    # in_subset = np.zeros(n_bits-1, dtype=np.bool_)
    # in_subset[subseq_idx] = 1

    # attempt to initially generate the data efficiently
    unq_entries = np.zeros((2**k, n_bits), dtype=int)
    # iterate over all unique bitstrings of length k
    for i, bits in enumerate(itertools.product('01', repeat=k)):
        unq_entries[i, :k] = np.array([int(x) for x in bits])
        for j in range(k, n_bits):
            prefix = unq_entries[i, j-k:j]
            targets = prefix[subseq_idx]
            unq_entries[i, j] = transition_func(targets)
    
    # now generate more data by randomly selecting from the unique entries
    resample_idx = np.random.choice(np.arange(2**k), size=n_data, replace=True)
    X = unq_entries[resample_idx]
    # shuffle X
    np.random.shuffle(X)

    Z = X
    # X = np.random.randint(0, 2, size=(n_data, n_bits))
    # for i in range(n_data):
    #     for j in range(k, n_bits):
    #         prefix = X[i, j-k:j]
    #         targets = prefix[subseq_idx]
    #         X[i, j] = transition_func(targets)
    if p_bitflip > 0:
        # flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits - 1 - k))
        # Z = np.copy(X)
        # Z[:,k:-1] = np.logical_xor(X[:,k:-1], flips).astype(int)
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits - 1))
        Z = np.copy(X)
        Z[:,:-1] = np.logical_xor(X[:,:-1], flips).astype(int)
    return X, Z, subseq_idx


def construct_graph(k):
    G = nx.DiGraph()
    
    # Generate all k-bit bitstrings
    vertices = [''.join(bits) for bits in itertools.product('01', repeat=k)]
    G.add_nodes_from(vertices)
    
    # Add edges based on the given rule
    for u in vertices:
        suffix = u[1:]
        for b in '01':
            v = suffix + b
            if v in G:
                G.add_edge(u, v)
    return G


def find_hamiltonian_cycle(G):
    n = len(G.nodes)
    path = []
    visited = set()
    
    def backtrack(current_node):
        if len(path) == n:
            # Check if there is an edge from the last node to the first node to form a cycle
            if path[0] in G.successors(current_node):
                path.append(path[0])
                return True
            else:
                return False

        for neighbor in G.successors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack(neighbor):
                    return True
                visited.remove(neighbor)
                path.pop()
        return False
    
    for start_node in G.nodes:
        path = [start_node]
        visited = {start_node}
        if backtrack(start_node):
            return path
    return None


def k_choose_m_hamilton_forecast_dataset(k, m, n_data, n_bits, p_bitflip, seed, subseq_idx=None):
    """
    Generate the next bit according to a De Bruijn sequence on a random size-m 
    subset of the previous k bits. This is the longest-period bitstring generating function
    with k-lookback.
    
    """
    # indices are relative to the bit being determined, not absolute
    if subseq_idx is None:
        subseq_idx = np.random.choice(np.arange(k), m, replace=False)
    # Construct a Hamilton cycle on m-bit prefixes
    G = construct_graph(m)
    hamiltonian_cycle = find_hamiltonian_cycle(G)
    mapping = {}
    for i in range(len(hamiltonian_cycle) - 1):
        u, v = hamiltonian_cycle[i], hamiltonian_cycle[i + 1]
        mapping[u] = v[-1]

    # construct a transition function
    def func(arr):
        s = ''.join([str(x) for x in arr])
        out = int(mapping.get(s))
        return out

    return k_choose_m_transition_forecast_dataset(func, k, m, n_data, n_bits, p_bitflip, seed, subseq_idx)


def k_lookback_weight_dataset(transition_matrix, k, n_data, n_bits, p_bitflip, seed):
    """Abstract function for _specific_ k-lookback boolean function of bitstring _weight_.
    
    The data generation works in 4 steps:
     1. Generate a length-k uniformly random seed string
     2. Recursively generate n+k additional bits
     3. Remove the seed bits, leaving just an (n_data, n_bits) array

    Using this scheme, we can generate all of the 

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
    determ_transition_matrix = {k: np.round(v) for k, v in transition_matrix.items()}

    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(n_data, n_bits + k))
    for i in range(n_data):
        for j in range(k, n_bits + k):
            if j == n_bits + k - 1:
                # The final bit is generated in a ''noiseless' manner, i.e. deterministically
                X[i, j] = determ_transition_matrix[np.sum(X[i, j-k:j])]
            else:
                weight = np.sum(X[i, j-k:j])
                X[i, j] = np.random.binomial(1, transition_matrix[weight])
    X = X[:,k:]
    Z = X.copy()
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        Z = np.logical_xor(X, flips).astype(int)
        
    return X, Z, None


# def k_choose_m_forecast_dataset(transition_matrix, k, n_data, n_bits, p_bitflip, seed):
#     assert len(transition_matrix) == k + 1
#     assert np.all([0 <= v <= 1 for v in transition_matrix.values()])
#     assert n_bits > k


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


def not_majority_5lookback_nondeterministic(n_data, n_bits, nondeterm, seed):
    """Generate nondeterministic NOT-MAJORITY forecasting data with n_bits bits.

    the `nondeterm` parameter is the probability of bitflipping a bit during 
    sequence generation. This is not equivalent to the `p_bitflip` parameter.

    We break ties by rounding up: majority of 1100 is 1.
    
    """
    transition_matrix = {0: 1 - nondeterm, 1: 1 - nondeterm, 2: 1 - nondeterm, 3: nondeterm, 4: nondeterm, 5: nondeterm}
    return k_lookback_weight_dataset(transition_matrix, 5, n_data, n_bits, 0, seed) 


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


def sparse_parity_k_n(n_bits, k, n_data, p_bitflip=0.0, seed=0, subseq_idx=None):
    """Generate a dataset where the final bit is the parity of a subset k of the n bits.
    
    The first n_bits-1 bits are randomly generated, and the final bit is the parity of 
    some size-k subset of the previous bits.

    Args:
        n_bits: TOTAL number of bits (including final bit)
        k: number of bits in the subset, to be chosen randomly.
        n_data: number of data points
        p_bitflip: probability of flipping a bit of INPUT data

    returns:
        X: (n_data, n_bits) array of noiseless data
        Z: (n_data, n_bits) array of noisy data, or None if p_bitflip is 0
    """
    np.random.seed(seed)
    if subseq_idx is None:
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

    Z = X
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        flips[:,-1] = 0 # we do not flip the last 'label' bit.
        Z = np.logical_xor(X, flips).astype(int)

    return X, Z, subseq_idx


def sparse_majority_k_n(n_bits, k, n_data, p_bitflip=0.0, seed=0, subseq_idx=None):
    """Generate a dataset where the final bit is a function of a subset k of the n bits.
    
    Args:
        n_bits: number of bits
        k: number of bits in the subset, to be chosen randomly.
        n_data: number of data points
        p_bitflip: probability of flipping an input bit
    """
    np.random.seed(seed)
    if subseq_idx is None:
        subseq_idx = np.random.choice(np.arange(n_bits - 1), k, replace=False)
    
    in_subset = np.zeros(n_bits-1, dtype=np.bool_)
    in_subset[subseq_idx] = 1
    X = np.random.randint(0, 2, size=(n_data, n_bits))

    for i in range(n_data):
        seq = X[i]
        if np.sum(seq[subseq_idx]) <= k // 2:
            X[i, -1] = 0
        else:
            X[i, -1] = 1

    Z = X
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        flips[:,-1] = 0 # we do not flip the last 'label' bit.
        Z = np.logical_xor(X, flips).astype(int)

    return X, Z, subseq_idx

# New functions with label noise (commented out for now)
def sparse_parity_k_n_with_label_noise(n_bits, k, n_data, p_bitflip=0.0, p_label_noise=0.0, seed=0, subseq_idx=None):
    # Generate a dataset where the final bit is the parity of a subset k of the n bits.
    # The first n_bits-1 bits are randomly generated, and the final bit is the parity of 
    # some size-k subset of the previous bits.
    #
    # Args:
    #     n_bits: TOTAL number of bits (including final bit)
    #     k: number of bits in the subset, to be chosen randomly.
    #     n_data: number of data points
    #     p_bitflip: probability of flipping a bit of INPUT data
    #     p_label_noise: probability of flipping the label bit
    #
    # returns:
    #     X: (n_data, n_bits) array of noiseless data
    #     Z: (n_data, n_bits) array of noisy data, or None if p_bitflip is 0
    np.random.seed(seed)
    if subseq_idx is None:
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

    Z = X
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        flips[:,-1] = 0 # we do not flip the last 'label' bit.
        Z = np.logical_xor(X, flips).astype(int)
        
    if p_label_noise > 0:
        label_flips = np.random.binomial(1, p_label_noise, size=(n_data, 1))
        Z[:,-1] = np.logical_xor(Z[:,-1], label_flips).astype(int)

    return X, Z, subseq_idx


def sparse_majority_k_n_with_label_noise(n_bits, k, n_data, p_bitflip=0.0, p_label_noise=0.0, seed=0, subseq_idx=None):
    # Generate a dataset where the final bit is a function of a subset k of the n bits.
    #
    # Args:
    #     n_bits: number of bits
    #     k: number of bits in the subset, to be chosen randomly.
    #     n_data: number of data points
    #     p_bitflip: probability of flipping an input bit
    #     p_label_noise: probability of flipping the label bit
    np.random.seed(seed)
    if subseq_idx is None:
        subseq_idx = np.random.choice(np.arange(n_bits - 1), k, replace=False)
    
    in_subset = np.zeros(n_bits-1, dtype=np.bool_)
    in_subset[subseq_idx] = 1
    X = np.random.randint(0, 2, size=(n_data, n_bits))

    for i in range(n_data):
        seq = X[i]
        if np.sum(seq[subseq_idx]) <= k // 2:
            X[i, -1] = 0
        else:
            X[i, -1] = 1

    Z = X
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        flips[:,-1] = 0 # we do not flip the last 'label' bit.
        Z = np.logical_xor(X, flips).astype(int)
        
    if p_label_noise > 0:
        label_flips = np.random.binomial(1, p_label_noise, size=(n_data, 1))
        Z[:,-1] = np.logical_xor(Z[:,-1], label_flips).astype(int)

    return X, Z, subseq_idx


def sparse_boolean_weightbased_k_n(n_bits, k, n_data, signature, p_bitflip=0.0, seed=0, subseq_idx=None):
    """Generate a dataset where the final bit is a function of a subset k of the n bits.
    
    Args:
        n_bits: number of bits
        k: number of bits in the subset, to be chosen randomly.
        n_data: number of data points
        signature: dict with (k+1) entries, corresponding to whether we assign 1 or 0 to that weight
        p_bitflip: probability of flipping a bit
    """
    np.random.seed(seed)
    assert len(signature) == k + 1
    assert np.all([0 <= v <= 1 for v in signature.values()])

    if subseq_idx is None:
        subseq_idx = np.random.choice(np.arange(n_bits - 1), k, replace=False)
    
    in_subset = np.zeros(n_bits-1, dtype=np.bool_)
    in_subset[subseq_idx] = 1
    X = np.random.randint(0, 2, size=(n_data, n_bits))

    for i in range(n_data):
        seq = X[i]
        weight = np.sum(seq[subseq_idx])
        X[i, -1] = signature[weight]

    Z = X
    if p_bitflip > 0:
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        flips[:,-1] = 0 # we do not flip the last 'label' bit.
        Z = np.logical_xor(X, flips).astype(int)

    return X, Z, subseq_idx


### Examine the patterns of the data
def k_choose_m_transition_cycle_dataset(transition_func, k, m, subseq_idx, n_bits=None):
    """Generate a dataset of forecast data with a k-choose-m scheme that continues until cycles are formed.

    The scheme is as follows:
    - All possible k-bit prefixes are used (2^k total)
    - A subset of m bits is chosen according to subseq_idx
    - Each subsequent bit is determined by transition_func acting on a length-m prefix;
        the length-m prefix is the size-m subsequence of the k previous bits
    - For each prefix, bits are generated until the original k-bit prefix is seen again
      (or until n_bits total bits are generated, if n_bits is specified)

    Args:
        transition_func: A function with inputs as int-sequences of length m and return value of 0 or 1
        k: amount of lookback
        m: number of bits in the subset of lookback
        subseq_idx: indices to select m bits from the k-bit lookback window (must be provided)
        n_bits: if provided, generate exactly this many bits (including prefix) for each sequence

    returns:
        sequences: list of numpy arrays, each representing a complete cycle starting from a unique k-bit prefix
        subseq_idx: the indices used for selecting m bits from the k-bit window
    """
    assert len(subseq_idx) == m
    subseq_idx = np.sort(np.array(subseq_idx))

    sequences = []
    # iterate over all unique bitstrings of length k
    for bits in itertools.product('01', repeat=k):
        prefix = np.array([int(x) for x in bits])
        
        # Initialize sequence with the k-bit prefix
        sequence = list(prefix)
        
        # Generate new bits until condition is met (cycle found or n_bits reached)
        while True:
            # Get the last k bits
            last_k_bits = np.array(sequence[-k:])
            
            # Select the m-bit subsequence
            targets = last_k_bits[subseq_idx]
            
            # Generate the next bit
            next_bit = transition_func(targets)
            sequence.append(next_bit)
            
            # Check termination condition
            if n_bits is not None:
                # Stop if we've generated n_bits total
                if len(sequence) >= n_bits:
                    # Truncate to exactly n_bits
                    sequence = sequence[:n_bits]
                    break
            else:
                # Original logic: stop if we see the original prefix again
                if len(sequence) > k and np.array_equal(sequence[-k:], prefix):
                    break
        
        sequences.append(np.array(sequence))
    
    return sequences

def k_choose_m_hamilton_cycle_dataset(k, m, subseq_idx, n_bits=None):
    """
    Generate datasets based on a Hamilton cycle where each sequence continues until a cycle is formed.
    
    Uses a De Bruijn sequence on a specified size-m subset of the previous k bits.
    
    Args:
        k: amount of lookback
        m: number of bits in the subset of lookback
        subseq_idx: indices to select m bits from the k-bit lookback window (must be provided)
        n_bits: if provided, generate exactly this many bits (including prefix) for each sequence
        
    Returns:
        X: numpy array of noiseless data with shape matching original implementation (2^k, cycle_length)
        None: placeholder for noisy data (for compatibility with original function)
        subseq_idx: the indices used for selecting m bits from the k-bit window
    """
    assert len(subseq_idx) == m, "subseq_idx must contain exactly m indices"
    assert all(0 <= idx < k for idx in subseq_idx), "all indices in subseq_idx must be between 0 and k-1"
    
    # Construct a Hamilton cycle on m-bit prefixes
    G = construct_graph(m)
    hamiltonian_cycle = find_hamiltonian_cycle(G)
    
    # Create mapping from m-bit sequences to next bit
    mapping = {}
    for i in range(len(hamiltonian_cycle) - 1):
        u, v = hamiltonian_cycle[i], hamiltonian_cycle[i + 1]
        mapping[u] = v[-1]

    # Construct transition function
    def transition_func(arr):
        s = ''.join([str(x) for x in arr])
        out = int(mapping.get(s))
        return out

    # Use the cycle detection function to generate sequences
    X = k_choose_m_transition_cycle_dataset(transition_func, k, m, subseq_idx, n_bits)
    
    # Return in the same format as the original function (X, Z, subseq_idx)
    return X