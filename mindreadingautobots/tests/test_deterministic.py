from mindreadingautobots.sequence_generators import deterministic

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