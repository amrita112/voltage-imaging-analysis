from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect

from behavior_responses import process_bpod_data
from behavior_responses import spike_rasters
from pre_processing import trial_tiff_stacks

def plot(data_path, metadata_file, cell_no, n_trials_plot = 5, trials_plot = [], trials_zoom = [], time_zoom = [], plot_zoom = False, scalebar_width_s = 0.3, scalebar_height = 0.05, color_spikes = 'gray', save_fig = False, save_path = None):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    session = sessions_to_process[0]

    trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_path, metadata_file)
    go_cue_time = process_bpod_data.get_go_cue_time(data_path, metadata_file)
    sample_end_time = process_bpod_data.get_sample_end_time(data_path, metadata_file)
    sample_start_time = process_bpod_data.get_sample_start_time(data_path, metadata_file)

    with open('{0}{1}dFF_trials.pkl'.format(data_path, sep), 'rb') as f:
        dFF_trials = pkl.load(f)
    frame_rate = dFF_trials['frame_rate']
    dFF_cell = dFF_trials[cell_no][session]

    spike_times_trials = spike_rasters.get_spike_times_trials(data_path, metadata_file, snr_thresh = 3)
    cell_ids = list(spike_times_trials.keys())
    spikes_cell = spike_times_trials[cell_ids[cell_no]][session]

    plt.figure(figsize = [4, 4], constrained_layout = True)

    max_val_prev = 0
    max_tvec = 0
    levels = []

    if len(trials_plot) == 0:
        trials_plot = list(range(n_trials_plot))

    for trial_no in range(n_trials_plot):

        dFF_plot = dFF_cell[trials_plot[trial_no]]
        min_val = np.min(dFF_plot)
        levels = np.append(levels, max_val_prev)
        tvec = np.linspace(0, len(dFF_plot)/frame_rate, len(dFF_plot))
        if np.max(tvec) > max_tvec:
            max_tvec = np.max(tvec)
        plt.plot(tvec, dFF_plot - min_val + max_val_prev, color = 'k', linewidth = 0.5)
        max_val_prev = np.max(dFF_plot - min_val + max_val_prev)

    levels = np.append(levels, max_val_prev)

    ylim = plt.ylim()
    plt.plot([sample_start_time, sample_start_time], ylim, color = 'k', linewidth = 1, linestyle = '--')
    plt.plot([sample_end_time, sample_end_time], ylim, color = 'k', linewidth = 1, linestyle = '--')
    plt.plot([go_cue_time, go_cue_time], ylim, color = 'k', linewidth = 1, linestyle = '--')
    plt.xticks(ticks = go_cue_time + np.array([-4, -2, 0, 2, 4]), labels = [-4, -2, 0, 2, 4])
    plt.yticks([])
    plt.xlabel('Time from go cue (s)', fontsize = 20)

    # dF/F scalebar
    left = max_tvec + scalebar_width_s
    bottom = scalebar_height
    dFF_scalebar = rect((left, bottom), scalebar_width_s, scalebar_height, color = 'k')
    plt.gca().add_patch(dFF_scalebar)
    plt.gca().tick_params(labelsize = 18)
    #plt.text(left + scalebar_width_s*2, bottom - 0.2*scalebar_height, '-{0} %\ndF/F'.format(np.round(scalebar_height*100, 2)), fontsize = 18)
    plt.xlim((go_cue_time - 4.5, left + scalebar_width_s*15))
    #plt.gca().axis('off')

    if plot_zoom:

        # Box to indicate location of zoom in main plot
        left = time_zoom[0]
        right = time_zoom[1]
        bottom = levels[trials_plot.index(trials_zoom[0])]
        top = levels[trials_plot.index(trials_zoom[1]) + 1]
        zoom_box = rect((left, bottom), right - left, top - bottom, color = 'm', fill = None, linewidth = 2)
        plt.gca().add_patch(zoom_box)

    if save_fig:
        if save_path == None:
            save_path = '{0}{1}Plots{1}'.format(data_path, sep)
        plt.savefig('{0}{1}Cell{2}_dFF_example_trials.png'.format(save_path, sep, cell_ids[cell_no]))

    if plot_zoom:
        plt.figure(figsize = [2, 2], constrained_layout = True)
        n_trials_plot = len(trials_zoom)
        max_val_prev = 0
        for trial_no in range(n_trials_plot):

            dFF_plot = dFF_cell[trials_zoom[trial_no]]
            tvec = np.linspace(0, len(dFF_plot)/frame_rate, len(dFF_plot))
            frame0 = int(np.argmin(np.abs(tvec - time_zoom[0])))
            frame1 = int(np.argmin(np.abs(tvec - time_zoom[1])))
            dFF_plot = dFF_plot[frame0:frame1]

            spikes_plot = spikes_cell[trials_zoom[trial_no]]
            spikes_plot = np.array([np.argmin(np.abs(tvec - spike)) for spike in spikes_plot])
            spikes_plot = spikes_plot[spikes_plot > frame0]
            spikes_plot = spikes_plot[spikes_plot < frame1]
            spikes_plot = spikes_plot - frame0
            spikes_plot = spikes_plot.astype(int)
            print(spikes_plot)

            min_val = np.min(dFF_plot)
            plt.plot(dFF_plot - min_val + max_val_prev, color = 'k', linewidth = 0.6)
            max_val_prev = np.max(dFF_plot - min_val + max_val_prev)
            plt.scatter(spikes_plot, max_val_prev*np.ones(len(spikes_plot)), color = color_spikes, marker = '.')

        # dF/F scalebar
        left = frame1 - frame0 + scalebar_width_s*frame_rate/10
        bottom = scalebar_height
        dFF_scalebar = rect((left, bottom), scalebar_width_s*frame_rate/10, scalebar_height, color = 'k')
        plt.gca().add_patch(dFF_scalebar)
        #plt.text(left + scalebar_width_s*frame_rate/2, bottom + scalebar_height/2, '-{0} %\ndF/F'.format(np.round(scalebar_height*100, 2)))
        plt.xlim((-10, left + 2*scalebar_width_s*frame_rate/10))
        plt.gca().axis('off')

        if save_fig:
            save_path = '{0}{1}Cell{2}_dFF_example_trials_zoom.png'.format(save_path, sep, cell_ids[cell_no])
            plt.savefig(save_path)
