from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from behavior_responses import process_bpod_data
from behavior_responses import spike_rasters
from pre_processing import trial_tiff_stacks

def main(data_path, metadata_file, bin_size_psth_ms = 50, snr_thresh = 5, suffix = ''):


    # Check trial types vs number of frames per trial
    check_trial_types(data_path, metadata_file)

    # Plot spike rasters for all cells - correct and incorrect, left and right trials
    spike_rasters.plot_spike_rasters(data_path, metadata_file, bin_size_ms = bin_size_psth_ms, snr_thresh = snr_thresh, suffix = suffix)

def check_trial_types(data_path, metadata_file):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    batch_data = metadata['batch_data']

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_rate = output['frame_and_trial_times']['frame_rate'][sessions_to_process[0]]
    go_cue_frames =  process_bpod_data.get_go_cue_time(data_path, metadata_file)*frame_rate

    n_frames_per_trial = []
    n_frames_per_trial_dict = trial_tiff_stacks.get_n_frames_per_trial(data_path, metadata_file)
    trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_path, metadata_file)
    correct_incorrect = []
    early_lick = []

    for session in sessions_to_process:

        n_batches = batch_data[session]['n_batches']
        first_trials = np.array(batch_data[session]['first_trials']).astype(int)
        last_trials = np.array(batch_data[session]['last_trials']).astype(int)

        for batch in range(n_batches):
            n_frames_per_trial = np.append(n_frames_per_trial,
                                        n_frames_per_trial_dict[session][first_trials[batch]:last_trials[batch]])

        types = trial_types_left_right_cor_inc[session]
        early_lick = np.append(early_lick, types == 0)
        correct_incorrect = np.append(correct_incorrect, np.logical_or(types == 1, types == 2))

    fig, ax1 = plt.subplots()
    ax1.plot(n_frames_per_trial, color = 'k')
    ax1.set_ylabel('Number of frames per trial')
    ax1.set_xlabel('Trial #')
    ax2 = ax1.twinx()
    ax2.plot(correct_incorrect, color = 'g', label = 'Correct trials')
    ax2.plot(early_lick, color = 'r', label = 'Early lick trials')
    ax2.legend()
    ax1.plot(list(range(len(n_frames_per_trial))), np.ones(len(n_frames_per_trial))*go_cue_frames,
                color = 'k', linestyle = '--')
    ax1.text(0, go_cue_frames, 'Go cue')
