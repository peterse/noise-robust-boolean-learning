import numpy as np
from mindreadingautobots.analysis.analysis_config import DATASET_NAMES


def append_last(x, y, mode='entropy'):
    if mode == 'bf':
        return np.concatenate((x, [0.5])).flatten(), np.concatenate((y, [0.5])).flatten()
    elif mode == 'entropy':
        return np.concatenate((x, [1])).flatten(), np.concatenate((y, [0.5])).flatten()
    else:
        raise ValueError(f"Invalid mode: {mode}")

def make_san_vs_rnn_plot(df_all, df_ana, bf_vals, ax, mode='entropy', apply_labels=True):
    rnn_stats = np.zeros((len(bf_vals), 8)) # val_acc median, q1, q3, max_val_acc, noiseless_val_acc median, q1, q3, max_noiseless_val_acc
    san_stats = np.zeros((len(bf_vals), 8))

    stats_dct = {
        "RNN": rnn_stats,
        "SAN": san_stats
    }

    for k in stats_dct.keys():
        for i, bf in enumerate(bf_vals):
            df = df_all.loc[df_all["bf"] == bf].loc[df_all["model"] == k]
            stats_dct[k][i, 0] = df['val_acc'].median()
            stats_dct[k][i, 1] = df['val_acc'].quantile(0.35)
            stats_dct[k][i, 2] = df['val_acc'].quantile(0.85)
            stats_dct[k][i, 3] = df['val_acc'].max()
            stats_dct[k][i, 4] = df['noiseless_val_acc'].median()
            stats_dct[k][i, 5] = df['noiseless_val_acc'].quantile(0.33)
            stats_dct[k][i, 6] = df['noiseless_val_acc'].quantile(0.67)
            stats_dct[k][i, 7] = df['noiseless_val_acc'].max()


    
    xvals = df_ana[mode].values.flatten() # entropy x-axis
    mle_noisy_final = df_ana["mle_noisy"].values.flatten()

    rnn_noisy_best = np.array(stats_dct["RNN"][:,3]).flatten()
    rnn_noiseless_best = np.array(stats_dct["RNN"][:,7]).flatten()
    rnn_noisy_median = np.array(stats_dct["RNN"][:,0]).flatten()
    rnn_noiseless_median = np.array(stats_dct["RNN"][:,4]).flatten()
    rnn_q1 = np.array(stats_dct["RNN"][:,1]).flatten()
    rnn_q3 = np.array(stats_dct["RNN"][:,2]).flatten()
    rnn_q1_noiseless = np.array(stats_dct["RNN"][:,5]).flatten()
    rnn_q3_noiseless = np.array(stats_dct["RNN"][:,6]).flatten()

    san_noisy_median = np.array(stats_dct["SAN"][:,0]).flatten()
    san_noiseless_median = np.array(stats_dct["SAN"][:,4]).flatten()
    san_q1 = np.array(stats_dct["SAN"][:,1]).flatten()
    san_q3 = np.array(stats_dct["SAN"][:,2]).flatten()
    san_q1_noiseless = np.array(stats_dct["SAN"][:,5]).flatten()
    san_q3_noiseless = np.array(stats_dct["SAN"][:,6]).flatten()

    if apply_labels:
        label_best = 'best LSTM'
        label_noiseless_best = 'best LSTM [noiseless]'
        label_median = 'median SAN'
        label_noiseless_median = 'median SAN [noiseless]'
        label_opt = 'opt'
        label_noiseless_opt = 'opt [noiseless]'
    else:
        label_best = None
        label_noiseless_best = None
        label_median = None
        label_noiseless_median = None
        label_opt = None
        label_noiseless_opt = None

    marker = '.'
    ax.plot(*append_last(xvals, rnn_noisy_best, mode=mode), label=label_best, c='r', ls='-', marker=marker)
    ax.plot(*append_last(xvals, rnn_noiseless_best, mode=mode), label=label_noiseless_best, c='r', ls='--', marker=marker)

    # ax.plot(*append_last(xvals, rnn_noisy_median), label='median LSTM', c='r', ls='-')
    # ax.plot(*append_last(xvals, rnn_noiseless_median), label='median LSTM [noiseless]', c='r', ls='--')
    # ax.fill_between(xvals, rnn_q1, rnn_q3, color='r', alpha=0.2)
    # ax.fill_between(xvals, rnn_q1_noiseless, rnn_q3_noiseless, color='r', alpha=0.2)

    ax.plot(*append_last(xvals, san_noisy_median, mode=mode), label=label_median, c='b', ls='-', marker=marker)
    ax.plot(*append_last(xvals, san_noiseless_median, mode=mode), label=label_noiseless_median, c='b', ls='--', marker=marker)
    ax.fill_between(xvals, san_q1, san_q3, color='b', alpha=0.2)
    ax.fill_between(xvals, san_q1_noiseless, san_q3_noiseless, color='b', alpha=0.2)

    ax.plot(*append_last(xvals, mle_noisy_final, mode=mode), label=label_opt, c='k', ls='-')

    # get all_mle_noiseless values from df_ana
    # all_mle_noiseless = df_ana["mle_noiseless"].values.flatten()
    # ax.plot(*append_last(xvals, all_mle_noiseless, mode=mode), label='opt [noiseless]', c='k', ls='--')
    if mode == 'entropy':
        ax.plot((0, 1, 1), (1, 1, 0.5), label=label_noiseless_opt, c='k', ls='--')
    elif mode == 'bf':
        ax.plot((0, 0.5, 0.5), (1, 1, 0.5), label=label_noiseless_opt, c='k', ls='--')

    ax.set_xlabel('Next bit entropy')
    return ax