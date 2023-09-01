import numpy as np
import matplotlib.pyplot as plt
from os.path import sep
import pickle as pkl

from population import load_data_session
from behavior_responses import process_bpod_data
from population import population_psth
from behavior_responses import spike_rasters

def latency_session(data_path, metadata_file, good_cells, good_blocks, go_cue_time, threshold = 0.05, bin_ms = 10, psth_bin_size_ms = 2.5, baseline_period_ms = 100, min_latency_ms = 100, period_show_s = 0.5, filename = 'go_cue_latency.pkl', overwrite = False,  make_plots = False, save_plots = False, save_path = None):

    """ Get latency of go cue response on left and right correct and incorrect trials for all cells in a session,
        and plot spike rasters with onset latency indicated.

        Inputs:
        data_path: string, location of all data for session
        metadata_file: string, name of file where metadata is stored
        good_cells: array (int), indices of cells passing QC (starting from 0)
        good_blocks: array (int), of same length as good_cells, last block # (starting from 0) that passes QC for each cell
        go_cue_time: float, time of go cue
        threshold: float, default 0.05, probability threshold for determining onset of go cue response
        bin_ms: int, default 10, bin for calculating spike rate after go cue
        psth_bin_size_ms: float, default 2.5 (1/frame rate): bin for single spike
        baseline_period_ms: int, default 100, period before go cue to calculate baseline spike rate
        min_latency_ms: int, default 100, plots for neurons with latency lower than this value will be shown upto period_show_s
        period_show_s: float, default 0.5, see min_latency_ms

        Outputs:

    """
    try:
        with open('{0}{1}{2}'.format(data_path, sep, filename), 'rb') as f:
            output = pkl.load(f)
    except:
        overwrite = True

    if overwrite:

        output = {}

        # Load data for session
        print('Loading data')
        session_data = load_data_session.load_data_session(data_path, metadata_file)

        # Load spike counts
        print('Loading spike counts')
        trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_path, metadata_file)
        spike_times_trials = population_psth.get_spike_times_trials(data_path, metadata_file,
                                            good_cells, good_blocks,
                                            go_cue_time, trial_types_left_right_cor_inc,
                                            overwrite = False)
        psth = spike_rasters.get_psth(trial_types_left_right_cor_inc, spike_times_trials, bin_size_ms = psth_bin_size_ms)
        tvec_trial = psth['tvec'] - go_cue_time

        # Calculate latency
        for cell in good_cells:
            output[cell] = latency_single_cell(psth[cell], tvec_trial,
                                               threshold = threshold, bin_ms = bin_ms, baseline_period_ms = baseline_period_ms,
                                               min_latency_ms = min_latency_ms, period_show_s = period_show_s,
                                               make_plot = make_plots, save_plot = save_plots,
                                               save_path = '{0}{1}Cell_{2}.png'.format(save_path, sep, cell + 1))

        # Save results
        with open('{0}{1}{2}'.format(data_path, sep, filename), 'wb') as f:
            pkl.dump(output, f)

    return output

def latency_single_cell(psth, tvec, threshold = 0.05, bin_ms = 10, baseline_period_ms = 100, min_latency_ms = 100, period_show_s = 0.5, make_plot = False, save_plot = False, save_path = None):

    """ Get latency of go cue response on left and right correct and incorrect trials for a single cell, and plot spike rasters with onset latency indicated.

        Inputs:
        psth: dict, output of spike_rasters.get_psth() with the fields: 'left_corr', 'right_corr', 'left_inc', 'right_inc', each of which is a dictionary
              with fields 'all_trials', 'mean', 'sem'. The field 'all_trials' is a n_binsXn_trials float array containing the spike psth.
        tvec: 1Xn_bins arary (float), time of each bin, tvec = 0 at go cue
        threshold: float, default 0.05, probability threshold for determining onset of go cue response
        bin_ms: int, default 10, bin for calculating spike rate after go cue
        baseline_period_ms: int, default 100, period before go cue to calculate baseline spike rate
        min_latency_ms: int, default 100, plots for neurons with latency lower than this value will be shown upto period_show_s
        period_show_s: float, default 0.5, see min_latency_ms

        Outputs:
        output: dict, with fields
            - 'probability': dict with fields 'left_corr', 'right_corr', 'left_inc', 'right_inc', each of which is a n_binsX1 float array
            - 'latency': dict with fields 'left_corr', 'right_corr', 'left_inc', 'right_inc', each of which is a int, bin # of onset of go cue response (0 = at go cue response)
    """
    output = {'latency': {}, 'probability': {}, 'baseline': {}, 'spike_rate_hz': {}}
    trial_types = list(psth.keys())
    n_trial_types = len(trial_types)

    go_cue_frame = np.argmin(np.abs(tvec)) # tvec = 0 at go cue time
    n_frames = psth[trial_types[0]]['all_trials'].shape[0]
    frame_rate = 1/np.mean(np.diff(tvec))

    for trial_type in trial_types:
        spike_train = np.sum(psth[trial_type]['all_trials'], axis = 1)
        assert(len(spike_train) == n_frames)

        (output['latency'][trial_type], output['probability'][trial_type],
            output['baseline'][trial_type], output['spike_rate_hz'][trial_type]) = get_latency(spike_train, go_cue_frame, frame_rate,
                                                                                            threshold = threshold, bin_ms = bin_ms, baseline_period_ms = baseline_period_ms)

    if make_plot:

        fig, ax = plt.subplots(nrows = 3, ncols = n_trial_types, sharex = True, constrained_layout = True, figsize = (15, 10))
        for col in range(n_trial_types):

            trial_type = trial_types[col]
            #ax[0, col].set_title(trial_type, fontsize = 20)

            n_trials = psth[trial_type]['all_trials'].shape[1]

            for trial in range(n_trials):
                spike_times = tvec[np.where(psth[trial_type]['all_trials'][:, trial])[0].astype(int)]
                ax[0, col].scatter(spike_times, np.ones(len(spike_times))*(trial + 1), color = 'k', marker = '.')

            spike_train = np.sum(psth[trial_type]['all_trials'], axis = 1)
            ax[1, col].plot(tvec, spike_train, color = 'k', linewidth = 0.4)
            #ax[1, col].plot(tvec[:go_cue_frame], np.ones(go_cue_frame)*output['baseline'][trial_type], color = 'b', linewidth = 3, )
            #ax[1, col].plot(tvec[go_cue_frame + 1:], [output['spike_rate_hz'][trial_type][frame - go_cue_frame] for frame in range(go_cue_frame + 1, n_frames)],
            #                color = 'b', linewidth = 3)

            ax[2, col].plot(tvec, output['probability'][trial_type], color = 'k')

            for row in range(3):
                ylim = ax[row, col].get_ylim()
                ax[row, col].plot([0, 0], ylim, linestyle = '--', color = 'k')

                if output['latency'][trial_type] > 0:

                    latency_time = np.round(tvec[output['latency'][trial_type] + go_cue_frame], 2)
                    ax[row, col].plot([latency_time, latency_time], ylim, linestyle = '--', color = 'r')
                    ax[0, col].set_title('{0}\nLatency = {1} ms'.format(trial_type, latency_time*1000), fontsize = 15)

                    if latency_time <= min_latency_ms/1000:
                        ax[row, col].set_xlim([-period_show_s, period_show_s])
                else:
                    ax[0, col].set_title('{0}\nNo go cue response'.format(trial_type), fontsize = 15)

            ax[2, col].set_xlabel('Time from go cue (s)', fontsize = 15)

        ax[0, 0].set_ylabel('Trial #', fontsize = 15)
        ax[1, 0].set_ylabel('Spike train', fontsize = 15)
        ax[2, 0].set_ylabel('Probability', fontsize = 15)

        if save_plot:
            fig.savefig('{0}'.format(save_path))

    return output

def get_latency(spike_frames, go_cue_frame, frame_rate, threshold = 0.05, bin_ms = 10, baseline_period_ms = 100):

    """ Calculate the latency of go cue response assuming a poisson spike train.
        Inputs:
        spike_frames: 1Xn_frames array (int), 0 for all frames and 1 on frames with spikes
        go_cue_frame: int, frame # of go cue
        frame_rate: float, frames/s
        threshold: float, default 0.05, probability threshold for determining onset of go cue response
        bin_ms: int, default 10, bin for calculating spike rate after go cue
        baseline_period_ms: int, default 100, period before go cue to calculate baseline spike rate
        Outputs:
        latency: int, frame # of onset of go cue response
        probability: 1Xn_frames array (float), probability of onset of go cue response (only defined after go_cue_frame)
    """

    n_frames = len(spike_frames)
    bin_frames = int(bin_ms*frame_rate/1000)
    baseline_period_frames = int(baseline_period_ms*frame_rate/1000)

    probability = np.zeros(n_frames)

    baseline_spike_rate = np.sum(spike_frames[go_cue_frame - baseline_period_frames:go_cue_frame])/baseline_period_frames # Spike rate before go cue (spikes/frame)

    n_spikes = []
    spike_rate_hz = []
    for frame in range(go_cue_frame, n_frames):

        first_frame = np.max([go_cue_frame, frame - bin_frames])
        n_spikes.append(int(np.sum(spike_frames[first_frame:frame])))
        spike_rate_hz.append(np.sum(spike_frames[first_frame:frame])*frame_rate/(frame - first_frame))

        probability[frame] = 1 - poisson_cdf(baseline_spike_rate, n_spikes[frame - go_cue_frame], frame - first_frame)

    probability = np.array(probability)
    sig_frames = np.where(probability[go_cue_frame:] < threshold)[0]
    if len(sig_frames) > 0:
        latency = sig_frames[0]
    else:
        latency = 0

    return (latency, probability, baseline_spike_rate, spike_rate_hz)

def poisson_cdf(r, n, t):
    """ Return cdf of poisson distribution with rate 'r' """
    return np.sum([poisson_probability(r, m, t) for m in range(n)])

def poisson_probability(r, m, t):
    """ Return proability of 'x' events occuring in t timepoints from poisson process with rate 'r'.
    """
    return ((r*t)**m)*np.exp(-(r*t))/np.math.factorial(m)
