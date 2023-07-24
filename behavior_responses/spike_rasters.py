from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from behavior_responses import process_bpod_data
from segmentation import get_roi_arrays
from pre_processing import trial_tiff_stacks
from volpy import quality_control

def plot_spike_rasters(data_path, metadata_file, bin_size_ms = 10, snr_thresh = 5, suffix = ''):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']

    plots_path = metadata['plots_path']
    spike_rasters_path = '{0}{1}Spike rasters'.format(plots_path, sep)
    if not os.path.isdir(spike_rasters_path):
        os.mkdir(spike_rasters_path)

    # Load data
    trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_path, metadata_file)
    spike_times_trials = get_spike_times_trials(data_path, metadata_file, snr_thresh)
    go_cue_time = process_bpod_data.get_go_cue_time(data_path, metadata_file)
    sample_end_time = process_bpod_data.get_sample_end_time(data_path, metadata_file)
    sample_start_time = process_bpod_data.get_sample_start_time(data_path, metadata_file)
    psth = get_psth(trial_types_left_right_cor_inc, spike_times_trials, bin_size_ms = bin_size_ms)
    #tvec_trial = psth['tvec'] - go_cue_time + sample_start_time
    tvec_trial = psth['tvec'] - go_cue_time

    roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)
    n_cells = roi_arrays[sessions_to_process[0]].shape[0]

    for cell in range(n_cells):

        print('Cell {0}'.format(cell + 1))

        fig, ax = plt.subplots(nrows = 2, ncols = 1, constrained_layout = True, sharex = True, figsize = [4, 5])
        ax[1].set_xlabel('Time from go cue (s)', fontsize = 20)
        #ax[0].set_ylabel('Trial # (excl. EL trials and low SNR trials)')
        ax[0].set_ylabel('Trial #', fontsize = 20)
        ax[1].set_ylabel('Spike rate (Hz)', fontsize = 20)

        # Plot spike raster for each cell
        level = 0
        for session in sessions_to_process:
            n_trials = len(trial_types_left_right_cor_inc[session])
            for trial in range(n_trials):
                type = trial_types_left_right_cor_inc[session][trial]
                if trial in spike_times_trials[cell][session].keys():
                    if type > 0:
                        color_spikes = {1: 'b', 2: 'r', 3: 'cornflowerblue', 4: 'lightcoral'}.get(type)
                        ax[0].scatter(spike_times_trials[cell][session][trial] - go_cue_time,
                                        level*np.ones(len(spike_times_trials[cell][session][trial])),
                                        marker = '.', color = color_spikes)
                        level += 1

        # Plot PSTH for each cell
        ax[1].plot(tvec_trial, psth[cell]['left_corr']['mean'], color = 'r', linewidth = 1.2)
        ax[1].fill_between(tvec_trial, psth[cell]['left_corr']['mean'] - psth[cell]['left_corr']['sem'], psth[cell]['left_corr']['mean'] + psth[cell]['left_corr']['sem'],
                            color = 'r', alpha = 0.2, linewidth = 0)
        ax[1].plot(tvec_trial, psth[cell]['left_inc']['mean'], color = 'lightcoral', linewidth = 0.8)
        ax[1].fill_between(tvec_trial, psth[cell]['left_inc']['mean'] - psth[cell]['left_inc']['sem'], psth[cell]['left_inc']['mean'] + psth[cell]['left_inc']['sem'],
                            color = 'lightcoral', alpha = 0.2, linewidth = 0)
        ax[1].plot(tvec_trial, psth[cell]['right_corr']['mean'], color = 'b', linewidth = 1.2)
        ax[1].fill_between(tvec_trial, psth[cell]['right_corr']['mean'] - psth[cell]['right_corr']['sem'], psth[cell]['right_corr']['mean'] + psth[cell]['right_corr']['sem'],
                            color = 'b', alpha = 0.2, linewidth = 0)
        ax[1].plot(tvec_trial, psth[cell]['right_inc']['mean'], color = 'cornflowerblue', linewidth = 0.8)
        ax[1].fill_between(tvec_trial, psth[cell]['right_inc']['mean'] - psth[cell]['right_inc']['sem'], psth[cell]['right_inc']['mean'] + psth[cell]['right_inc']['sem'],
                            color = 'cornflowerblue', alpha = 0.2, linewidth = 0)

        # Plot dashed line to show sample end time and go cue time
        [y0, y1] = ax[0].get_ylim()
        ax[0].plot(np.ones(10)*(sample_end_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 1, color = 'k')
        ax[0].plot(np.ones(10)*(sample_start_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 1, color = 'k')
        ax[0].plot(np.zeros(10), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 1, color = 'k')
        ax[0].tick_params(axis = 'both', labelsize = 18)

        [y0, y1] = ax[1].get_ylim()
        ax[1].plot(np.ones(10)*(sample_end_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 1, color = 'k')
        ax[1].plot(np.ones(10)*(sample_start_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 1, color = 'k')
        ax[1].plot(np.zeros(10), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 1, color = 'k')
        ax[1].tick_params(axis = 'both', labelsize = 18)
        plt.savefig('{0}{1}Cell_{2}_{3}.png'.format(spike_rasters_path, sep, cell + 1, suffix))


def get_spike_times_trials(data_path, metadata_file, snr_thresh):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    batch_data = metadata['batch_data']

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_times = output['frame_and_trial_times']['frame_times']

    n_frames_per_trial = trial_tiff_stacks.get_n_frames_per_trial(data_path, metadata_file)

    volpy_results_file = metadata['volpy_results_file']
    with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
        volpy_results = pkl.load(f)

    good_cells = quality_control.perform_quality_control(data_path, metadata_file, volpy_results)

    roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)
    n_cells = roi_arrays[sessions_to_process[0]].shape[0]

    spike_times_trials = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}

    max_spike_time = 0
    for session in sessions_to_process:

        n_batches = batch_data[session]['n_batches']
        first_trials = np.array(batch_data[session]['first_trials']).astype(int)
        last_trials = np.array(batch_data[session]['last_trials']).astype(int)

        cum_frames_per_trial = np.cumsum(n_frames_per_trial[session]).astype(int)
        cum_frames_per_trial = np.insert(cum_frames_per_trial, 0, 0)

        for batch in range(n_batches):

            estimates = volpy_results[session][batch]['vpy']
            first_frame_batch = cum_frames_per_trial[first_trials[batch]]

            for trial in range(first_trials[batch], last_trials[batch]):

                frames = list(range(cum_frames_per_trial[trial], cum_frames_per_trial[trial + 1])) # 0 is the first frame in the session
                frames_batch = frames - first_frame_batch                                         # 0 is the first frame in the batch
                frame_times_trial = frame_times[session][frames]
                frame_times_trial = frame_times_trial - frame_times_trial[0]

                for cell in range(n_cells):
                    if good_cells[session][cell, batch] == 0:
                        continue
                    else:
                        if estimates['snr'][cell] < snr_thresh:
                            continue
                        else:
                            spike_frames_cell = estimates['spikes'][cell]
                            spike_frames_trial = [frame for frame in spike_frames_cell if frame in frames_batch]
                            spike_frames_trial = spike_frames_trial - frames_batch[0]
                            spike_times_trials[cell][session][trial] = frame_times_trial[spike_frames_trial.astype(int)]
                            if len(spike_times_trials[cell][session][trial]) > 0:
                                max_time = np.max(spike_times_trials[cell][session][trial])
                                if max_time > max_spike_time:
                                    max_spike_time = max_time

    spike_times_trials['max_spike_time'] = max_spike_time
    return spike_times_trials


def get_psth(trial_types_left_right_cor_inc, spike_times_trial, bin_size_ms = 10):

    max_spike_time = spike_times_trial['max_spike_time']
    bin_edges_ms = np.arange(0, max_spike_time*1000 + 3*bin_size_ms, bin_size_ms)
    n_bins = len(bin_edges_ms) - 1

    n_cells = len([key for key in spike_times_trial.keys() if np.issubdtype(key, np.integer)]) - 1
    psth = {cell: {'left_corr':    {'all_trials': np.zeros([n_bins, 1]), 'mean': np.zeros(n_bins), 'sem': np.zeros(n_bins)},
                   'right_corr':   {'all_trials': np.zeros([n_bins, 1]), 'mean': np.zeros(n_bins), 'sem': np.zeros(n_bins)},
                   'left_inc':     {'all_trials': np.zeros([n_bins, 1]), 'mean': np.zeros(n_bins), 'sem': np.zeros(n_bins)},
                   'right_inc':    {'all_trials': np.zeros([n_bins, 1]), 'mean': np.zeros(n_bins), 'sem': np.zeros(n_bins)}
              } for cell in range(n_cells)}

    for session in list(trial_types_left_right_cor_inc.keys()):

        types = trial_types_left_right_cor_inc[session]
        n_trials = len(types)
        for trial in range(n_trials):
            type_string = {1: 'right_corr', 2: 'left_corr', 3: 'right_inc', 4: 'left_inc'}.get(types[trial])
            if types[trial] > 0:
                for cell in range(n_cells):
                    if trial in spike_times_trial[cell][session].keys():
                        psth_cell_trial = np.histogram(spike_times_trial[cell][session][trial]*1000, bin_edges_ms)
                        psth[cell][type_string]['all_trials'] = np.append(psth[cell][type_string]['all_trials'],
                                                                          np.reshape(psth_cell_trial[0], [n_bins, 1]),
                                                                         axis = 1)

    for trial_type in ['left_corr', 'right_corr', 'left_inc', 'right_inc']:
        for cell in range(n_cells):

            psth[cell][trial_type]['all_trials'] = psth[cell][trial_type]['all_trials'][:, 1:]
            psth[cell][trial_type]['mean'] = np.mean(psth[cell][trial_type]['all_trials'], axis = 1)*1000/bin_size_ms
            n_trials = psth[cell][trial_type]['all_trials'].shape[1]
            psth[cell][trial_type]['sem'] = np.std(psth[cell][trial_type]['all_trials'], axis = 1)*1000/bin_size_ms/np.sqrt(n_trials)

    psth['tvec'] = (bin_edges_ms[1:] + bin_edges_ms[:-1])/2000

    return psth
