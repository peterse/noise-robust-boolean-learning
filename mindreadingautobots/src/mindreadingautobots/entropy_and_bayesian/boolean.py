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
    n = len(X_arr[0])
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