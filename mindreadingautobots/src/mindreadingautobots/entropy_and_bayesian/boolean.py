import numpy as np
import itertools
from itertools import combinations

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
    """a.k.a. Inf_i[f]. Compute the average sensitivity of f with respect to i"""
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
        try:
            # sometimes when you use lookup tables on training data, we don't actually see all the data
            pred = func(x)
        except KeyError:
            continue
        if truth == pred:
            acc += 1 / 2 ** n
    return acc

def compute_acc_on_dataset(func, dataset):
    """Compute the accuracy of `func` on the dataset."""
    acc = 0
    for x in dataset:
        truth = x[-1]
        try:
            # sometimes when you use lookup tables on training data, we don't actually see all the data
            pred = func(x[:-1])
        except KeyError:
            continue
        if truth == pred:
            acc += 1 / len(dataset)
    return acc

def boolean_function_from_signature(f, n):
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
        lookup_table[tuple([int(x) for x in feature])] = int(most_common_label)
    
    # now, convert the lookup table to a function
    def lookup_func(feature):
        # gave_warning = False
        key = tuple([int(x) for x in feature])
        # if key not in lookup_table:
        #     if not gave_warning:
        #         print("Warning: This feature is not in the lookup table, outputting random {0,1}. You will not be warned again.")
        #         gave_warning = True
        #     return np.random.randint(2)
            # return 0
        return lookup_table[key]
    return lookup_func, lookup_table


def did_you_forget_your_subset_idx(signature_tup, X):
    """
    Suppose you have a dataset X but you didn't write down the subset used to compute
    the boolean function with a weight-based signature `signature_tup`. This code will
    do an efficient search to find the subset consistent with signature_tup.
    """
    # This is code for recovering the hidden subset in relatively fast time
    positions = list(combinations(range(13), 8))
    bitstrings = np.zeros((len(positions), 13), dtype=int)
    for i, pos in enumerate(positions):
        bitstrings[i, pos] = 1
    # X = data_io.load_dict_as_numpy("counterexample000110000_nbits14_n50000_bf20_seed1234/noiseless_train.pkl")
    for row in X:
        x = row[:-1]
        y = row[-1]
        testers = np.multiply(x, bitstrings)
        tester_sums = testers.sum(axis=1)
        labels = [signature_tup[w] for w in tester_sums]
        bitstrings = bitstrings[labels == y]
        print(len(bitstrings))
        if len(bitstrings) == 1:
            break

    ranran = list(range(13))
    subset_idx = []
    for i, f in enumerate(bitstrings[0]):
        if f == 1:
            subset_idx.append(ranran[i])
    for i, x in enumerate(X):
        assert signature_tup[sum(x[:-1][subset_idx])] == x[-1]
    return bitstrings[0]


def compute_dataset_optimal_sens_and_err(X, Z1, Z2):
    """
    Train by building a lookup table for Z1, then evaluate that table on Z1 and Z2
    Args:
        X: (N, n) array of noiseless data, with each row containing feature x and label y
        Z1: (N, n) array of noisy data
    
    Returns: A sequence of pairs:
    [
        (error of Z1_lookup on Z1, sensivitiy of Z1_lookup),
        (err of X_tr_lookup on Z1, sens of X_tr_lookup)
    ]
    """
    n_bits = Z1.shape[1] - 1
    all_bitstrings = np.array(list(itertools.product([0, 1], repeat=n_bits)))

    f_lookup_on_Z1, lookup_table_on_Z = dataset_lookup_table(Z1)
    # f_lookup_on_X, lookup_table_on_X = dataset_lookup_table(X)
    err_memorize_Z1_evaluate_Z1 = 1 - compute_acc_on_dataset(f_lookup_on_Z1, Z1)
    err_memorize_Z1_evaluate_Z2 = None
    if Z2 is not None:
        err_memorize_Z1_evaluate_Z2 = 1 - compute_acc_on_dataset(f_lookup_on_Z1, Z2)
    sens_memorize_Ztrain = average_sensitivity(f_lookup_on_Z1, all_bitstrings)

    return err_memorize_Z1_evaluate_Z1, err_memorize_Z1_evaluate_Z2, sens_memorize_Ztrain


def compute_fnstar_err_sens(signature, p_bitflip, signature_type="weight", signature_signed=False):
    """Compute the error and sensitivity of the (noisy) Bayes-optimal predictor.
    
    Args:
        signature: a tuple of 0s and 1s, representing the weight-signature of the noiseless function
            (i.e. f(x) = signature[sum(x)])
        p_bitflip: the probability of a BSC bitflip in the noisy dataset
        signature_weight: if True, the given signature is a weight-based signature with length n+1
        signature_signed: if True, the given signature is {-1, 1}, and we return the signed Bayes-optimal predictor

    """

    if signature_type == "weight":
        if signature_signed:
            raise NotImplementedError("Signed weight-based signatures are not implemented")
        n = len(signature) - 1
        hash = dict(zip(range(n+1), signature))
        func = lambda b: hash[sum(b)]
    elif signature_type == "all":
        n = int(np.log2(len(signature)))
        bools = [tuple(x) for x in itertools.product([0, 1], repeat=n)]
        hash = dict(zip(bools, signature))
        if signature_signed:
            func = lambda b: (hash[tuple(int(x) for x in b)] + 1) // 2
        else:
            func = lambda b: hash[tuple(int(x) for x in b)]

    X_arr = np.array(list(itertools.product([0, 1], repeat=n)))

    # noisy_lookup[row,col] is the JOINT probability Pr(f(z)=row| x=col)
    noisy_lookup = np.zeros((2, 2**n))
    true_lookup = np.zeros((2, 2**n))

    # simulate a noisy dataset to compute the noisy lookup table
    for i, x in enumerate(X_arr):
        func_value = func(x) # compute y=f(x)
        # true lookup is an array with 2 rows; there is a p_x at [row, column] if 
        # f[column] = row]. so, true_lookup[i, j] = pr(f(x) = i| x=j)
        true_lookup[func(x), i] = 1
        # iterate over all of the z values that contribute to 
        for e in itertools.product([0, 1], repeat=n):
            z = np.array(x) ^ np.array(e)
            p_x_given_z = p_bitflip ** sum(e) * (1-p_bitflip)**(n - sum(e))
            # increment noisy_lookup at the binary index of z
            # noisy_lookup[i, j] = pr(f(z) = i,  x=j) 
            noisy_lookup[func_value, int(''.join(map(str, z)), 2)] += p_x_given_z 

    # round up to get argmax 
    # noisy_mle = np.round(noisy_lookup)  
    # fnstar_err_arr = np.multiply(noisy_lookup, true_lookup) / 2 ** n # "inner product" of the functions
    # fnstar_err_v2 = 1 - fnstar_err_arr.sum()
    # print(fnstar_err_v2) # fixme: why is this different than fnstar_err? montecarlo confirms that fnstar_err below is right.
    
    # build the function f_N* and compute its sensitivity
    fnstar_dct = {}
    for i, x in enumerate(X_arr):
        fnstar_dct[tuple(x)] = np.argmax(noisy_lookup[:, i])
    def fnstar(x):
        return fnstar_dct[tuple(x)]
    p_zy = generate_noisy_distr(n, p_bitflip, func)
    fnstar_acc = compute_acc_noisytest(p_zy, fnstar, n) # accuracy of fN* MLE on noisy data
    fnstar_err = 1 - fnstar_acc
    # print(fnstar_err)
    sensitivity_fnstar = average_sensitivity(fnstar, X_arr)

    if signature_signed:
        def fnstar_signed(x):
            return fnstar_dct[tuple(x)] * 2 - 1
        return fnstar_err, sensitivity_fnstar, fnstar_signed
    else:   
        return fnstar_err, sensitivity_fnstar, fnstar


def boolean_fourier_transform(f):
    """
    Compute the boolean Fourier transform of a boolean function f specified by (-1, 1)^{2^n} array
    
    Args:
        f: A length 2^n array of {1, -1} representing the boolean function
        
    Returns:
        A length 2^n array of {1, -1} representing the Fourier components
    """
    n = int(np.log2(len(f)))
    assert len(f) == 2**n, "Input array must have length 2^n"
    f = np.array(f)
    fourier = np.zeros(2**n)
    
    for S, S_bin in enumerate(itertools.product([0, 1], repeat=n)):
        for x, x_bin in enumerate(itertools.product([-1, 1], repeat=n)):
            masked = np.multiply(x_bin, S_bin)
            masked[masked == 0] = 1
            chi_S_x = np.prod(masked)
            fourier[S] += f[x] * chi_S_x

    fourier /= 2**n
    
    return fourier


def chi(S_bin,n):
    """generate a parity function for a subset S
    
    Args:
        S: the bitstring {0,1}^n indicating affected subset of bits
        n: the number of bits
    """
    out = np.zeros(2**n)
    for x, x_bin in enumerate(itertools.product([-1, 1], repeat=n)):
        masked = np.multiply(x_bin, S_bin)
        masked[masked == 0] = 1
        out[x] = np.prod(masked)

    return out


def inverse_boolean_fourier_transform(fhat):
    """
    Compute the inverse boolean Fourier transform of a real array fhat of length 2^n
    
    Args:
        fhat: A length 2^n array of real numbers representing the Fourier components
        
    Returns:
        A length 2^n array of real numbers representing the boolean function
    """
    n = int(np.log2(len(fhat)))
    assert len(fhat) == 2**n, "Input array must have length 2^n"
    fhat = np.array(fhat)
    f = np.zeros(2**n)
    
    for S, S_bin in enumerate(itertools.product([0, 1], repeat=n)):
        # create a one-hot vector with a 1 at S
        f += fhat[S] * chi(S_bin, n)
    
    return f


def noise_operator_on(f, rho, input_fourier=False, return_fourier=False):
    """
    Compute the noise operator on a boolean function f with respect to the noise parameter rho.
    """
    n = int(np.log2(len(f)))
    assert len(f) == 2**n, "Input array must have length 2^n"
    f = np.array(f)
    if not input_fourier:
        f = boolean_fourier_transform(f)
    trhof_hat = np.zeros(2**n)
    for S, S_bin in enumerate(itertools.product([0, 1], repeat=n)):
        trhof_hat[S] = rho ** sum(S_bin) * f[S]
    if return_fourier:
        return trhof_hat
    else:
        return boolean_fourier_transform(trhof_hat)
    

def inf_i(f, i):
    """compute Inf_i[f], assuming a {-1, 1} signature input"""
    n = int(np.log2(len(f)))
    X_arr = [tuple(a) for a in list(itertools.product([-1, 1], repeat=n))]
    f_hash = dict(zip(X_arr, f))
    out = 0
    for x in X_arr:
        x_ = [a for a in x]
        x_[i] = -x_[i]
        x_ = tuple(x_)
        if f_hash[x] != f_hash[x_]:
            out += 1

    return out / (2**n)

def total_inf(f):
    """compute I[f], assuming a {-1, 1} signature input"""
    n = int(np.log2(len(f)))
    out = 0
    for i in range(n):
        out += inf_i(f, i)
    return out

def s_v1(f, x):
    """Compute the sensitivity of f at x"""
    n = len(x)
    out = 0
    for i in range(n):
        x_ = list(x)
        x_[i] = - x_[i]
        x_ = tuple(x_)
        if f(x) != f(x_):
            out += 1
    return out


def walsh_hadamard_matrix(n):
    """
    Construct a Walsh-Hadamard matrix iteratively.
    
    Args:
        n: The dimension parameter, matrix will be 2^n x 2^n
        
    Returns:
        A 2^n x 2^n numpy array representing the Walsh-Hadamard matrix
    """
    size = 2**n
    H = np.ones((size, size))
    
    for i in range(n):
        step = 2**i
        for row in range(0, size, 2*step):
            for col in range(0, size, 2*step):
                # Set the bottom-right quadrant to -1
                H[row+step:row+2*step, col+step:col+2*step] *= -1
    
    return H / (2**n)


def build_szS_mask(n):
    """
    build a vector of |S| in fourier space
    elementwise multiplication with $\hat{f}^2$ with this mask and then summing
    the result gives total influence
    """
    out = np.zeros(2**n)
    for S, S_bin in enumerate(itertools.product([0, 1], repeat=n)):
        out[S] = sum(S_bin)
    return out


def build_trho_mask(n, rho):
    """
    Build a vector of rho^|S| in fourier space
    elementwise multiplication with $\hat{f}$ gives the spectrum of $T_\rho f$
    """
    out = np.zeros(2**n)
    for S, S_bin in enumerate(itertools.product([0, 1], repeat=n)):
        out[S] = rho ** sum(S_bin)
    return out


def compute_influence_fourier(fhat):
    """
    args: fhat is (2**n, N) array of fourier coefficients, one function per column
    """
    n = int(np.log2(len(fhat)))
    szS_mask = build_szS_mask(n)
    H_fhat_squared = np.multiply(fhat, fhat)
    H_fhat_squared_szS = np.multiply(szS_mask[:,None], H_fhat_squared)
    return np.sum(H_fhat_squared_szS, axis=0)


def compute_sgn_Trho_f_influence(fhat, rho, H=None):
    n = int(np.log2(len(fhat)))
    if H is None:
        H = walsh_hadamard_matrix(n)
    trho_mask = build_trho_mask(n, rho)
    Trho_H_sampled_f = np.multiply(trho_mask[:,None], fhat)
    Trho_f = H @ Trho_H_sampled_f
    all_sgn_Trho_f = np.sign(Trho_f)
    ghat = H @ all_sgn_Trho_f
    inf_g = compute_influence_fourier(ghat)
    return inf_g


def test_parity_functions():
    # Test fourier transform of parity functions for 3 bits; by linearty we are good for the rest.
    n = 3
    for S in itertools.product([0, 1], repeat=n):
        # Create parity function for subset S
        f = []
        for x in itertools.product([-1, 1], repeat=n):
            # Compute parity: product of bits in S
            parity = 1
            for i, bit in enumerate(S):
                if bit == 1:
                    parity *= x[i]
            f.append(parity)
        
        f = tuple(f)
        fhat = boolean_fourier_transform(f)
        
        # Convert S to index in Fourier domain
        S_idx = int(''.join(map(str, S)), 2)
        
        # Check that fhat is one-hot at S_idx
        expected = np.zeros(2**n)
        expected[S_idx] = 1
        
        print(f"Testing parity function for subset S={S}")
        print(f"Fourier transform: {fhat}")
        print(f"Expected one-hot at index {S_idx}")
        print(f"Max absolute error: {np.max(np.abs(fhat - expected))}")
        print()

if __name__ == "__main__":
    # Run the test
    test_parity_functions()

