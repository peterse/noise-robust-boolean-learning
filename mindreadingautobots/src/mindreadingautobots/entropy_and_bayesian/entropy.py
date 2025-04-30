import numpy as np
import matplotlib.pyplot as plt


def empirical_entropy_estimate(X, intermediate_idx=[None], noiseless_X=None):
    """Construct an empirical estimator for the entropy H(X_n|X^{n-1})
        and maximum likelihood estimator.
    
    This is based purely on an empirical estimate for p_{X_n|X^{n-1}}.
    Thus, at the same time we can approximate the optimal success
    probability for the next bit.

    When intermediate_idx is provided, we do this for a dataset of size
    `n` for each n in intermediate_idx. This allows us to see convergence
    of the entropy estimate.

    Returns:
        - entropy_out: entropy estimate for the next bit
        - mle_out: maximum likelihood success probability for the next bit
        - mle_lookup_out: A lookup table for the maximum likelihood estimator
    """
    # n_data, n_bits = X.shape

    entropy_out = [] # entropy estimate for the next bit
    mle_out = [] # maximum likelihood estimate for the next bit
    mle_lookup_out = [] # maximum likelihood estimator for the next bit, in the form of {tuple(X^{n-1}): X_n}
    lb = 0
    current_total = 0
    # These dicts represent histograms for empirical distribution of X^n and X^{n-1}
    dct_n = {}
    dct_nminus1 = {}
    keys_n = []
    keys_nminus1 = []

    # This will iterate over smaller subsets of data.
    for i in intermediate_idx:
        current_bounds = [lb, i]
        keys_n_new, counts_n = np.unique(X[lb:i,:], axis=0, return_counts=True)
        keys_nminus1_new = np.unique(keys_n_new[:, :-1], axis=0)
        # cast to normal ints.
        keys_n_new = [list([int(x) for x in k]) for k in keys_n_new]
        keys_nminus1_new = [list([int(x) for x in k]) for k in keys_nminus1_new]
        # keep track of the new total data
        slc_n = sum(counts_n)
        current_total += slc_n
        keys_n = list(set(keys_n + [tuple(k) for k in keys_n_new]))
        keys_nminus1 = list(set(keys_nminus1 + [tuple(k) for k in keys_nminus1_new]))

        for k, v in zip(keys_n_new, counts_n):
            if tuple(k) not in dct_n:
                dct_n[tuple(k)] = 0
            dct_n[tuple(k)] += v

            if tuple(k[:-1]) not in dct_nminus1:
                dct_nminus1[tuple(k[:-1])] = 0
            dct_nminus1[tuple(k[:-1])] += v

        # at this point, we have a histogram for p(X^n) and p(X^{n-1}) where 
        # X describes a (possibly noisy) bit sequence. 

        # estimate entropy from histograms
        H_Xn = 0
        for k, v in dct_n.items():
            p = v / current_total
            H_Xn -= xlogx(p)
        
        H_Xnminus1 = 0
        mle = 0
        mle_lookup = {}
        for k in keys_nminus1:
            # entropy calculation: H(X^{n-1})
            p = dct_nminus1[k] / current_total # = p_{X^{n-1}}(x^{n-1})
            H_Xnminus1 -= xlogx(p)
            # MLE calculation: Find out whether p_{X_n|X^{n-1}}(0|x^{n-1}) > p_{X_n|X^{n-1}}(1|x^{n-1})
            pr_xnminus1_and_0 = dct_n.get(tuple(list(k)+[0])) 
            pr_xnminus1_and_1 = dct_n.get(tuple(list(k)+[1]))

            mle_lookup[tuple(list(k))] = 0
            
            if pr_xnminus1_and_0 is not None:
                pr_xnminus1_and_0 = pr_xnminus1_and_0 / current_total
                # This condition is equivalent to the conditional distirbution being > 0.5
                if pr_xnminus1_and_0 > 0.5 * p:
                    mle += pr_xnminus1_and_0
                else:
                    mle += p - pr_xnminus1_and_0
                    mle_lookup[tuple(list(k))] = 1
            else: # logically, this other entry must be present
                mle += pr_xnminus1_and_1  / current_total
                mle_lookup[tuple(list(k))] = 1

        mle_out.append(mle)
        entropy_out.append(H_Xn - H_Xnminus1)
        mle_lookup_out.append(mle_lookup)
        lb = i

    return entropy_out, mle_out, mle_lookup_out


def compute_mle_with_lookup(X, lookup):
    """Given a trained lookup table of {tuple(bitstring): next bit}, compute MLE accuracy"""
    n_data, n_bits = X.shape
    keys, counts = np.unique(X, axis=0, return_counts=True)
    counts = counts / n_data
    out = 0
    for k, c in zip(keys, counts):
        prefix = tuple([int(s) for s in k[:-1]])
        next_bit = lookup.get(prefix)
        if next_bit is not None:
            # print(prefix, lookup[prefix], k[-1], out)
            out += c * (1 - k[-1] ^ lookup[prefix])
        else:
            print("prefix not found: ", prefix)
    return out


def plot_entropy_and_convergence(all_H, all_mle, all_mle_noiseless, benchmarks, p_bitflips, intermediate_idx, fano=False, xcoord='p_bitflips'):

    fig, ax = plt.subplots()
    final_H = [H[-1] for H in all_H]
    final_mle = [mle[-1] for mle in all_mle]
    final_mle_noiseless = [mle for mle in all_mle_noiseless]
    if fano:
        ax.plot(p_bitflips, 1 - binary_entropy_inverse(np.array(final_H)), label='Fano bound')
    if xcoord == 'p_bitflips':
        x = p_bitflips
        xlabel = "Pr(bitflip)"
    elif xcoord == 'entropy':
        x = final_H
        xlabel = r"$H(X_n|Z^{n-1})$"
    ax.plot(x, np.array(final_mle), label='MLE[noisy] noisy val acc', linestyle='-', marker='o')
    ax.plot(x, np.array(final_mle_noiseless), label='MLE[noisy] noiseless val acc', linestyle='-', marker='o')
    if not np.allclose(benchmarks, 0):
        ax.plot(x, benchmarks, label='memorizing on training data', linestyle='-', marker='o')
    # ax.set_ylim([.5, 1.05])
    ax.grid()
    ax.legend()
    ax.set_xlabel(xlabel)
    print(final_mle)

    # plot convergences
    fig, axes = plt.subplots(1, len(p_bitflips), figsize=(10, 4))
    if len(p_bitflips) == 1:
        axes = [axes]
    for i, (H_results, mle_results) in enumerate(zip(all_H, all_mle)):
        ax = axes[i]
        print(len(H_results), len(mle_results), len(intermediate_idx))
        ax.plot(intermediate_idx,  np.array(H_results), label='entropy convergence')
        ax.plot(intermediate_idx, np.array(mle_results), label='MLE convergence')
        # ax.set_ylim([.5, 1.05])
        ax.grid()
        ax.legend()
        ax.set_title(f'p={p_bitflips[i]}')
        ax.set_xlabel('n_data')
        ax.semilogx()


def xlogx(x):
    """Safely compute x*logx in base 2"""
    temp = np.log2(x, out=np.zeros_like(x), where=(x!=0))
    return np.multiply(x, temp)


def shannon_entropy(p):
    """Compute the shannon entropy of an arbitrary distribution p"""
    return -np.sum(xlogx(p))


def binary_entropy_inverse(y):
    return 0.5 * (1 - np.sqrt(1 - y ** (4/3)))


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