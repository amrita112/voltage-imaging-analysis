import numpy as np
import matplotlib.pyplot as plt

def latency_single_cell(psth, tvec, threshold = 0.05, make_plot = False):

    """ Get latency of go cue response on left and right correct and incorrect trials for a single cell, and plot spike rasters with onset latency indicated.
        Inputs:
        psth: dict, output of spike_rasters.get_psth() with the fields: 'left_corr', 'right_corr', 'left_inc', 'right_inc', each of which is a dictionary
              with fields 'all_trials', 'mean', 'sem'. The field 'all_trials' is a n_binsXn_trials float array containing the spike psth.
        tvec: 1Xn_bins arary (float), time of each bin, tvec = 0 at go cue
        threshold: float, default 0.05, probability threshold for determining onset of go cue response
        Outputs:
        output: dict, with fields
            - 'probability': dict with fields 'left_corr', 'right_corr', 'left_inc', 'right_inc', each of which is a n_binsX1 float array
            - 'latency': dict with fields 'left_corr', 'right_corr', 'left_inc', 'right_inc', each of which is a int, bin # of onset of go cue response
    """
    output = {'latency': {}, 'probability': {}}
    trial_types = list(psth.keys())
    n_trial_types = len(trial_types)

    go_cue_frame = np.argmin(np.abs(tvec)) # tvec = 0 at go cue time
    n_frames = psth[trial_types[0]]['all_trials'].shape[0]

    for trial_type in trial_types:
        spike_train = np.reshape(np.sum(psth[trial_type]['all_trials'], axis = 1), [1, -1])
        assert(spike_train.shape[1] == n_frames)

        (output['latency'][trial_type], output['probability'][trial_type]) = get_latency(spike_train, go_cue_frame, threshold = threshold)

    if make_plot:

        fig, ax = plt.subplots(nrows = 3, ncols = n_trial_types, sharex = True, constrained_layout = True, figsize = (15, 10))
        for col in range(n_trial_types):

            trial_type = trial_types[col]
            ax[0, col].set_title(trial_type, fontsize = 20)

            n_trials = psth[trial_type]['all_trials'].shape[1]

            for trial in range(n_trials):
                spike_times = tvec[np.where(psth[trial_type]['all_trials'][:, trial])[0].astype(int)]
                ax[0, col].scatter(spike_times, np.ones(len(spike_times))*(trial + 1), color = 'k', marker = '.')

            spike_train = np.sum(psth[trial_type]['all_trials'], axis = 1)
            ax[1, col].plot(tvec, spike_train, color = 'k', linewidth = 0.4)

            ax[2, col].plot(tvec, output['probability'][trial_type], color = 'k')

            if output['latency'][trial_type] > 0:
                latency_time = tvec[output['latency'][trial_type]]
                for row in range(3):
                    ylim = ax[row, col].get_ylim()
                    ax[row, col].plot([latency_time, latency_time], ylim, linestyle = '--', color = 'k')
                fig.suptitle('Latency = {0} ms'.format(latency_time*1000), fontsize = 25)
            else:
                fig.suptitle('No go cue response', fontsize = 25)

            ax[2, col].set_xlabel('Time from go cue (s)', fontsize = 15)

        ax[0, 0].set_ylabel('Trial #', fontsize = 15)
        ax[1, 0].set_ylabel('Spike train', fontsize = 15)
        ax[2, 0].set_ylabel('Probability', fontsize = 15)

    return output

def get_latency(spike_frames, go_cue_frame, threshold = 0.05):

    """ Calculate the latency of go cue response assuming a poisson spike train.
        Inputs:
        spike_frames: 1Xn_frames array (int), 0 for all frames and 1 on frames with spikes
        go_cue_frame: int, frame # of go cue
        threshold: float, default 0.05, probability threshold for determining onset of go cue response
        Outputs:
        latency: int, frame # of onset of go cue response
        probability: 1Xn_frames array (float), probability of onset of go cue response (only defined after go_cue_frame)
    """

    assert(spike_frames.shape[0] == 1)
    n_frames = spike_frames.shape[1]

    probability = np.zeros(n_frames)

    baseline_spike_rate = np.mean(spike_frames[:go_cue_frame])/go_cue_frame # Spike rate before go cue (spikes/frame)
    n_spikes = [int(np.sum(spike_frames[go_cue_frame:frame])) for frame in range(go_cue_frame, n_frames)]
    probability[go_cue_frame:] = [1 - poisson_cdf(baseline_spike_rate, n_spikes[frame - go_cue_frame], frame - go_cue_frame) for frame in range(go_cue_frame, n_frames)]
    sig_frames = np.where(probability[go_cue_frame:] < threshold)[0]
    if len(sig_frames) > 0:
        latency = sig_frames[0]
    else:
        latency = 0

    return (latency, probability)

def poisson_cdf(r, n, t):
    """ Return cdf of poisson distribution with rate 'r' """
    return np.sum([poisson_probability(r, x, t) for x in range(n - 1)])

def poisson_probability(r, x, t):
    """ Return proability of 'x' events occuring in t timepoints from poisson process with rate 'r'.
    """
    return ((r*t)**x)*np.exp(- r*t) / np.math.factorial(x)
