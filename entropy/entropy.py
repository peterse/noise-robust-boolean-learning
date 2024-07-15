import numpy as np


def xlogx(x):
    """Safely compute x*logx in base 2"""
    temp = np.log2(x, out=np.zeros_like(x), where=(x!=0))
    return np.multiply(x, temp)


def shannon_entropy(p):
    """Compute the shannon entropy of an arbitrary distribution p"""
    return -np.sum(xlogx(p))

def binary_entropy(p):
    """Compute the binary entropy function H(p) = -p*log(p) - (1-p)*log(1-p)"""
    return - xlogx(p) - xlogx(1 - p)


def one_prob_to_conditional(transition_matrix, k):
    """Convert a weight-based transition matrix to a cond. distr.
    
    Args:
        transition_matrix (np.array): a dictionary with (k+1) keys and values in [0, 1]
        k (int): the lookback parameter

    Returns:
        p_x_conditional (np.array): a (2,) * (k + 1) array 
    """
    p_x_conditional = np.zeros((2,) * (k + 1))
    # we iterate over all indices of p_x_conditional and assign values based on the weight
    # of the binary representation of the index
    for idx in np.ndindex(p_x_conditional.shape):
        conditioner = idx[1:]
        weight = sum(conditioner)
        # the probability of a 1 given the previous `lookback` bits
        # each of these is a stochastic matrix. I'm fucking lazy and i'm 
        # not going to deal with sparse arrays this is python ffs
        p_x_conditional[tuple([1] + list(conditioner))] = transition_matrix[weight]
        p_x_conditional[tuple([0] + list(conditioner))] = 1 - transition_matrix[weight]
    assert p_x_conditional.shape == (2,)*(k+1)
    
    return p_x_conditional


def conditional_H_of_xn_given_kbits_klookback(n, k, p_S, p_x_conditional):
    """Compute the conditional entropy 
    
        H(X_n | X_{n-1}, ..., X_{n-k})
    
    where the conditional distribution has k-lookback, i.e.

        p(x_i | x_{i-1}, ..., x_1) = p(x_i | x_{i-1}, ..., x_{i-k})

    Args:
        n (int): the length of the bit sequence
        k (int): the number of bits to condition on
        p_S (np.array, shape (2,)*k): the distribution of the first k bits (seed bits).
                                    The axes of `p_S` correspond to [s_n, s_{n-1}, ..., s_1].
        p_x_conditional (np.array, shape (2,)*(k+1)): the conditional distribution of 
            the next bit given the last k bits. The axes of `p_x_conditional` correspond to
            [x_i, x_{i-1}, ..., x_{i-k}]. 
    """
    assert p_S.shape == (2,)*k
    assert p_x_conditional.shape == (2,)*(k + 1)
    # confirm that p_x_conditional is a valid conditional distribution
    assert np.allclose(p_x_conditional.sum(axis=0), 1)
    assert n >= k

    # compute p(x_{n-1}, ..., x_{n-k}) := running_prob
    p_S = p_S.astype(float)
    p_x_conditional = p_x_conditional.astype(float)
    running_prob = p_S
    for _ in range(n - k - 1):
        running_prob = p_x_conditional * running_prob[None, :]
        running_prob = running_prob.sum(axis=-1)


    # Compute H(X_n | X_{n-1}, ..., X_{n-k})
    H = - xlogx(p_x_conditional) * running_prob[None, :]
    H = H.sum()
    return H


def conditional_H_of_xnplusm_given_kbits_klookback(n, m, k, p_S, p_x_conditional):
    """Compute the conditional entropy 
    
        H(X_{n+m-1}, ..., X_n | X_{n-1}, ..., X_{n-k})
    
    where the conditional distribution has k-lookback, i.e.

        p(x_i | x_{i-1}, ..., x_1) = p(x_i | x_{i-1}, ..., x_{i-k}).

    Note that this implies that 
    
        p(x_{i+m-1}, ..., x_i|x_{i-1}, ..., x_1) = p(x_{i+m}, ..., x_i|x_{i-1}, ..., x_{i-k})

    The total number of variables runs from 1,...,n-1, n, ..., n+m-1. Examples

    (n, m, k)       computes:
    (3, 2, 2)       H(X4X3|X2X1)
    (4, 3, 2)       H(X6X5X4|X3X2)
    (4, 3, 3)       H(X6X5X4|X3X2X1)

    This function has a tricky slicing operation that reduces complexity to O(2^(k+m)):
    The slice in Step 2 re-inserts variables that the cond. distr. is otherwise
    cond. indep. from. Example: Say m=3, k=2. We need to compute p(X5X4X3|X2X1).
    We start with 
        p_x_conditional := p(X3|X2X1)
        conditional_m_given_k := p(X4|X3X2)
    with this slice, we now have 
        p_x_conditional := p(X4|X3X2X1)
    Elementwise multiplication gives us conditional_m_given_k = p(X4X3|X2X1).
    With the next iteration slice, we have
        p_x_conditional := p(X5|X4X3X2X1)
        conditional_m_given_k := p(X4X3|X2X1)
        elementwise multiplication gives us p(X5X4X3|X2X1).

    Args:
        n (int): This is n such that the entropy is conditioned on (n-1) bits; the nth
            bit is the first that appears to the left of th `|` conditioning bar.
        m (int): the _total_ number of bits to the left of the conditioning bar
        k (int): the number of bits to condition on
        p_S (np.array, shape (2,)*k): the distribution of the first k bits (seed bits).
                                    The axes of `p_S` correspond to [s_n, s_{n-1}, ..., s_1].
        p_x_conditional (np.array, shape (2,)*(k+1)): the conditional distribution of 
            the next bit given the last k bits. The axes of `p_x_conditional` correspond to
            [x_i, x_{i-1}, ..., x_{i-k}]. 
    """
    assert p_S.shape == (2,)*k
    assert p_x_conditional.shape == (2,)*(k + 1)
    # confirm that p_x_conditional is a valid conditional distribution
    assert np.allclose(p_x_conditional.sum(axis=0), 1)
    assert n > k
    assert m >= 1

    # compute p(x_{i-1}, ..., x_{i-k}) := running_prob
    p_S = p_S.astype(float)
    p_x_conditional = p_x_conditional.astype(float) # shape (2,)*(k+1)
    running_prob = np.copy(p_S) # shape (2,)*k


    # Step 1: stepforward - the recursion step transforms 
    # p(x_k, ..., x_1) into p(x_{n-1}, ..., x_{n-k})
    # which is the thing we are conditioning on. it has len k
    for i in range(n - k - 1):
        running_prob = p_x_conditional * running_prob[np.newaxis, :]
        running_prob = running_prob.sum(axis=-1)

    # Step 2: stepforward - compute p(x_{n+m-1}, ..., x_n | x_{n-1}, ..., x_{n-k}) 
    conditional_m_given_k = np.copy(p_x_conditional)
    for i in range(m - 1):
        # See docstring.
        cond_slice = (...,) + (np.newaxis,) * (i + 1)
        conditional_m_given_k =   p_x_conditional[cond_slice] * conditional_m_given_k

    # Compute H(X_n | X_{n-1}, ..., X_{n-k}) - Careful: we're computing xlogx, not logx
    p_slice = (np.newaxis,)*m + (slice(None),)
    H = - xlogx(conditional_m_given_k) * running_prob[p_slice]
    # print(conditional_m_given_k * running_prob[p_slice], "p_x3x2")
    H = H.sum()
    return H