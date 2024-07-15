"""ucan.py - methods for gerneating training data involving UCAN."""

import numpy as np
import numbers 
def bitwise_ucan_v1(n, n_data, p0_delta, p_diff, seed=0):
    """Generate a sample of (Gamma, Delta) UCAN pairs, in Gamma|Delta mode.

    The CAN is SEPARABLE and SYMMETRIC. The inputs assume that all bits are independent,
    so that we can specify the distribution with (2^2)*n parameters.
    
    This will return an (n_data, n, 2) array of UCAN pairs. The last axis corresponds to
    a noise bitstring (Gamma) and a CAN bitstring (Delta). 

    NOTE: fixing p0_delta and p_diff and enforcing symmetry FIXES the joint distribution.
    e.g., p_00 = p0_delta - p_diff
    p_diff and p_00 must describe a valid join distribution, i.e. p_00 + 2*p_diff <=1

    Args:
        p0_delta: (float) or (array). Probability for correlated noise to be 0. 
                If float, it will be promoted to a length-n array containing Pr(Delta_i=0) at location i
        p_diff: (float) or (array) probability that gamma, delta differ. 
                Containing Pr(Gamma_i=1, Delta_i=0) for i=1...n

    Returns:
        (n_data, n, 2) array of (gamma, delta)

    """
    if isinstance(p0_delta, numbers.Number):
        p0_delta = np.full(n, p0_delta)
    if isinstance(p_diff, numbers.Number):
        p_diff = np.full(n, p_diff)

    assert len(p0_delta) == n
    assert len(p_diff) == n

    # The joint distribution (Gamma, Delta) is [p_00, p_diff, p_diff, 1-p_00-2*p_diff]
    p_00 = p0_delta - p_diff
    p_11 = 1 - p_00 - 2*p_diff
    assert (np.all(p_11 >= 0) and np.all(p_11 <= 1))


    # Sample our Delta bits according to p0_delta
    np.random.seed(seed)
     # shape (n_data, n); the second axis probabilities of 0 are given by p0_delta
    delta = np.random.binomial(1, 1 - p0_delta, size=(n_data, n))

    # Now compute conditionals for Gamma, and sample from anouther Bernoulli using these
    p_gd_10 = np.divide(p_diff, p0_delta) #size n array, pr(Gamma_i=1|Delta_i=0)
    p1_delta = 1 - p0_delta
     # size n array, pr(Gamma_i=1|Delta_i=1)
    p_gd_11 = np.divide(p_11, p1_delta, out=np.zeros_like(p_11), where=p1_delta!=0)

    # We'll just mask for two separate bernouli sampling experiments
    mask_11 = p_gd_11 * delta 
    gammas_11 = np.random.binomial(1, mask_11)
    mask_10 = p_gd_10 * (1 - delta)
    gammas_10 = np.random.binomial(1, mask_10)
    gammas = gammas_11 + gammas_10

    # Now stack the two datasets of bitstrings
    return np.stack([gammas, delta], axis=-1)