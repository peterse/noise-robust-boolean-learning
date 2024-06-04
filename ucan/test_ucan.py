import numpy as np
import ucan 


def test_bitwise_ucan_v1():
    n = 6
    n_data = 100000
    p0_delta = 0.35
    p_diff = 0.2

    x = ucan.bitwise_ucan_v1(n, n_data, p0_delta, p_diff, seed=0)
    gammas = x[:,:,0]
    deltas = x[:,:,1]

    # Check that the marginal probabilities are correct
    assert np.allclose(np.mean(deltas, axis=0), 1 - p0_delta, atol=0.05)

    # Check that the joint probabilities are correct
    p_00 = p0_delta - p_diff
    p_11 = 1 - p_00 - 2*p_diff
    assert np.allclose(np.mean(gammas * deltas), p_11, atol=0.05)
    assert np.allclose(np.mean(gammas ^ deltas), 2 * p_diff, atol=0.05)


def test_edge_cases_bitwise_ucan_v1():
    """TODO: for when we have probabilities exactly equal to zero or one."""