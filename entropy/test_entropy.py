import entropy
import numpy as np


def test_conditional_H_of_xn_given_kbits_klookback():
    k = 2
    p_S = np.ones(4).reshape(2, 2) / 4
    p_x_conditional = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(2, 2, 2)
    for n in [3, 4, 5, 6]:
        # the last bit is uniformly random
        assert np.allclose(entropy.conditional_H_of_xn_given_kbits_klookback(n, k, p_S, p_x_conditional), 1) 

    # the only entropy here is the seeds
    p_x_conditional = np.array([1, 1, 1, 1, 0, 0, 0, 0]).reshape(2, 2, 2)
    assert np.allclose(entropy.conditional_H_of_xn_given_kbits_klookback(7, k, p_S, p_x_conditional), 0) 

    # p_x_conditional = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 2, 2)
    # p_x_conditional = p_x_conditional / p_x_conditional.sum(axis=0)
    # print(entropy.conditional_H_of_xn_given_kbits_klookback(3, k, p_S, p_x_conditional))


def test_lookback_math():
    # This is also kind of a tutorial for working with conditional probabilities 
    # Note: Please prefer `np.newaxis` over `None` for clarity. `None` is 
    # how slice denotes ':', which differs from how numpy convers `None` to `np.newaxis`
    # 1. Create # p_{X2|X1X0}: this only works for a single bit left of the conditional '|'
    p_x_conditional = np.random.rand(2, 2, 2) 
    p_x_conditional = p_x_conditional / p_x_conditional.sum(axis=0) 
    assert p_x_conditional.sum(axis=0).all() == 1 # check for valid conditional along axis0

    # 2. Create p_{X4X3X2|X1X0} iteratively: We pad the conditional
    # distribution with enough newaxis (or 'None') to align p_x_conditional
    # with the first three axes of p
    p = np.random.rand(4).reshape(2, 2) # p_{X1X0}
    for i in range(3):
        cond_slice = (slice(None),) + (np.newaxis,) * i
        # Here's the math: we add one axis to p.shape with the slice [None,:] that 
        # aligns the axes of p to the right of the axes of p_x_conditional. But
        # we also keep one axis of p_x_conditional to the left of p...
        assert len(cond_slice) == (len(p.shape) + 1) - (len(p_x_conditional.shape) - 1)
        p = p_x_conditional[cond_slice] * p[np.newaxis, :]
    
    # p = p_{X4X3X2X1X0}
    p_marginal = p.sum(axis=(0,1,2)) # p_{X1X0}
    # Now we slice to align the marginal with p to get a conditional
    marginal_slice = (np.newaxis,)*3 + (slice(None),)
    p_cond_truth = p / p_marginal[marginal_slice] # p_{X4X3X2|X1X0}

    # 3. Compute p_{X4X3X2|X1X0} by propagation: p_cond represents
    # p_{Xi|X{i-1}...X{i-k}}, so we pad this with enough newaxes at
    # its end to align with the axes of p_x_conditional. 
    p_cond = np.copy(p_x_conditional)
    for i in range(2):
        forward_slice = (slice(None),) + (np.newaxis,)*(i+1) 
        p_cond = p_x_conditional[forward_slice] * p_cond[np.newaxis,:]
    # now we have propagated p_{X2|X1X0} to p_{X4X3X2|X1X0}
    # print(p_cond.shape)

    assert np.allclose(p_cond, p_cond_truth)