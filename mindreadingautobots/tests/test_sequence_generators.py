import os

import numpy as np
from mindreadingautobots.sequence_generators import deterministic, make_datasets
from mindreadingautobots.sequence_generators.data_io import save_numpy_as_dict, load_dict_as_numpy

def test_deterministic():

    # Test that number_of_generating_methods is working for single generating method
    gen = deterministic.SequenceGen(lookback=4, seed=228, number_of_generating_methods=1)
    x, funcs = gen.deterministically_generate_sequences(15, 10, save=False)
    for f in funcs:
        assert f == funcs[0]

def test_deterministic_number_of_generating_methods():
    # make sure we get the right number of generating methods and sequences
    for (length, num_gen) in [(7, 4), (77, 6), (23, 7)]:

        gen = deterministic.SequenceGen(lookback=4, seed=228, number_of_generating_methods=num_gen)
        x, funcs = gen.deterministically_generate_sequences(4, length, save=False)

        # hash the funcs because of course we have to do that
        funcs = [tuple([(k, v) for k, v in f.items()]) for f in funcs]
        assert len(x) == length
        assert len(set(funcs)) == num_gen


def test_sparse_parity_k_n():
    # Test that sparse_parity_k_n is working
    for p_bitflip in [0, 0.5]:
        for (n_bits, k, n_data) in [(8, 2, 20), (16, 4, 30), (40, 4, 100)]:
            X, Z, idx = make_datasets.sparse_parity_k_n(n_bits, k, n_data, p_bitflip=p_bitflip, seed=0)
            assert X.shape == (n_data, n_bits)
            if p_bitflip == 0:
                assert Z is None
            else:
                assert Z.shape == (n_data, n_bits)
            
            assert len(idx) == k
            for i, row in enumerate(X):
                assert np.sum(row[idx]) % 2 == X[i][-1] # enforce parity
                if Z is not None:
                    assert X[i,-1] == Z[i,-1] # we do not apply label noise.


def test_not_majority_4lookback():

    n_data = 100
    n_bits = 16
    p_bitflip = 0.0
    seed = 1234
    X, _ = make_datasets.not_majority_4lookback(n_data, n_bits, p_bitflip, seed)
    for row in X:
        for i in range(5, 9):
            if sum(row[i:i+4]) >= 2:
                assert row[i+4] == 0
            else:
                assert row[i+4] == 1

    # test seed working.
    np.random.seed(4434)
    Xnew, _ = make_datasets.not_majority_4lookback(n_data, n_bits, p_bitflip, seed)
    np.testing.assert_array_equal(X, Xnew) 



def test_parity_4lookback():

    n_data = 100
    n_bits = 16
    p_bitflip = 0.0
    seed = 1234
    X, _ = make_datasets.parity_4lookback(n_data, n_bits, p_bitflip, seed)
    for row in X:
        for i in range(5, 9):
            assert sum(row[i:i+4]) % 2 == row[i+4]

    # test seed working.
    np.random.seed(22922)
    Xnew, _ = make_datasets.parity_4lookback(n_data, n_bits, p_bitflip, seed)
    np.testing.assert_array_equal(X, Xnew) 


def test_data_io():
    # Test that save_numpy_as_dict and load_dict_as_numpy pass round-trip test
    data = np.random.randint(0, 2, size=(10, 5))
    data_path = 'test_data.pkl'
    save_numpy_as_dict(data, data_path)
    data_loaded = load_dict_as_numpy(data_path)
    assert np.all(data == data_loaded)
    # cleanup
    os.remove(data_path)

if __name__ == "__main__":
    test_deterministic()
    test_deterministic_number_of_generating_methods()
    test_sparse_parity_k_n()
    test_data_io()
    test_not_majority_4lookback()
    test_parity_4lookback()
    print("All tests passed!")