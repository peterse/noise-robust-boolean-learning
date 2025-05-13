from itertools import product
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mindreadingautobots.entropy_and_bayesian import boolean


for n in [4, 5, 6, 7, 8]:
    for p in [0.2, 0.22, 0.24, 0.26, 0.28, 0.30]:
    
        X_arr = np.array(list(itertools.product([0, 1], repeat=n)))

        # p_x = 1 / (2 ** n) # uniform distribution over x # chaos distribution and the thing with chaos distribution is its fair
        # WEIGHT-BASED FUNCTIONS

        signatures = itertools.product([0, 1], repeat=n+1)
        f_accs = []
        fn_accs = []
        fn_noiseless_accs = []
        imbal_list = []
        sentitivity_f_list = []
        sensitivity_fnstar_list = []
        sensitivity_diff_list = []

        for signature in signatures:

            hash = dict(zip(range(n+1), signature))
            func = lambda b: hash[sum(b)]


            
            noisy_lookup = np.zeros((2, 2**n)) # noisy_lookup[row,col] is the JOINT probability Pr(f(z)=row| x=col)
            true_lookup = np.zeros((2, 2**n)) # true lookup is an array with 2 rows; there is a p_x at [row, column] if  
                                            # f[column] = row]. so, true_lookup[i, j] = pr(f(x) = i| x=j)

            for i, x in enumerate(X_arr):

                func_value = func(x) 
                true_lookup[func(x), i] = 1

                # Iterate over all possible noisy strings
                for e in product([0, 1], repeat=n):

                    z = np.array(x) ^ np.array(e)
                    p_x_given_z = p ** sum(e) * (1-p)**(n - sum(e)) 

                    noisy_lookup[func_value, int(''.join(map(str, z)), 2)] += p_x_given_z 

            imbal = abs(true_lookup[0,:].sum() - true_lookup[1,:].sum())  / 2 ** n
            imbal_list.append(imbal)

            noisy_mle = np.round(noisy_lookup)  
            out = np.multiply(noisy_mle, true_lookup) / 2 ** n # "inner product" of the functions
            diff = out.sum()


            fnstar_dct = {}

            for i, x in enumerate(X_arr):
                fnstar_dct[tuple(x)] = np.argmax(noisy_lookup[:, i])

            def fnstar(x):
                return fnstar_dct[tuple(x)]

            sensitivity_f = boolean.average_sensitivity(func, X_arr)
            sensitivity_fnstar = boolean.average_sensitivity(fnstar, X_arr)
            sensitivity_diff = sensitivity_f - sensitivity_fnstar

            # accuracies on dataset
            p_zy = boolean.generate_noisy_distr(n, p, func)
            noisy_f_acc = boolean.compute_acc_noisytest(p_zy, func, n) # accuracy of f on noisy data
            noiseless_fnstar_acc = boolean.compute_acc_test(fnstar, func, n) # accuracy of fN* on noiseless data
            noisy_fnstar_acc = boolean.compute_acc_noisytest(p_zy, fnstar, n) # accuracy of fN* MLE on noisy data

            f_accs.append(noisy_f_acc)
            fn_accs.append(noisy_fnstar_acc)
            fn_noiseless_accs.append(noiseless_fnstar_acc)
            sentitivity_f_list.append(sensitivity_f)
            sensitivity_fnstar_list.append(sensitivity_fnstar)
            sensitivity_diff_list.append(sensitivity_diff)


        signatures_list = list(itertools.product([0, 1], repeat=n+1))
        df = pd.DataFrame({'signature': signatures_list, 'f_acc': f_accs, 'fn_acc': fn_accs, 'fn_noiseless_acc': fn_noiseless_accs,
                        'imbalance': imbal_list, 'sensitivity_f': sentitivity_f_list, 'sensitivity_fnstar': sensitivity_fnstar_list, 'sensitivity_diff': sensitivity_diff_list,
                        'bitflip': len(signatures_list)*[p]})
        df['acc_diff'] = df['f_acc'] - df['fn_acc']

        df_filtered = df[
            (df['imbalance'] < 1) &
            (df['sensitivity_f'] > 0) &
            (df['sensitivity_fnstar'] > 0) &
            (df['sensitivity_diff'] > 0) &
            (df['fn_acc'] > 0.6) &
            (df['f_acc'] > 0.6) &
            (df['acc_diff'] != 0)
        ]

        if len(df_filtered) > 0:

            df_filtered.to_csv(f'dentsets/weight_functions_n={n}_p={p}.csv', index=False)