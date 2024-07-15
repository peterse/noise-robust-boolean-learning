"""This is a rewrite of SequenceGen_Deterministic.py from before."""

import numpy as np
import itertools

class SequenceGen:
    def __init__(self, lookback=4, seed=228, number_of_generating_methods=1) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) #Seed rng

        self.number_of_generating_methods = number_of_generating_methods
        self.k = lookback

        # all possible binary strings in array form
        self.source_seqs = [''.join(i) for i in itertools.product(['0', '1'], repeat=self.k)]
        self.source_weights = np.arange(0, self.k + 1)


    def generate_sequence(self, length, one_prob):
        seq = self.rng.choice(self.source_seqs, shuffle=False)
        for i in range(self.k, length):
            if one_prob.get(np.sum([int(x) for x in seq[i-self.k:i]])) == 0:
                seq += '0'
            else:
                seq += '1'
        return seq
    
    def get_prob(self):
        return self.rng.choice([0, 1])
    
    def deterministically_generate_sequences(self, length, num_seq, save=True):
        all_seq = np.empty((num_seq, length), dtype=str)
        one_probs = [] # cache the underlying pr distribution per bitstring

        # stake out how many strings will be generated according to which method
        # Divide num_seq into self_number_of_generating_methods parts as evenly as possible
        sub_len, r = divmod(num_seq, self.number_of_generating_methods)
        loc = 0
        for _ in range(self.number_of_generating_methods):
            # Fix a generator method for this chunk of sequences
            one_prob = {i:self.get_prob() for i in self.source_weights}
            
            # fucky remainder stuff. This is so we get exactly num_seq rows
            sub_len_ragged = sub_len
            if r > 0:
                sub_len_ragged += 1
                r -= 1

            for s in range(sub_len_ragged):               
                all_seq[loc + s] = np.array(list(self.generate_sequence(length, one_prob)))
                one_probs.append(one_prob)
            loc += sub_len_ragged

        if save:
            np.savetxt(f"datasets/testdata_determ_lookback{self.k}_length{length}_seed{self.seed}_numberMethods{self.number_of_generating_methods}.txt", all_seq, fmt='%s', delimiter=' ')
        return all_seq, one_probs