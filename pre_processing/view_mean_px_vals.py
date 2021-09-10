import os
from os.path import sep
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def view_mean_px_vals(data_path, metadata_file, save_fig = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    n_sessions = len(sessions_to_process)

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_times = output['frame_and_trial_times']['frame_times']

    mean_px_val_file = metadata['mean_px_val_file']
    with open('{0}{1}{2}'.format(data_path, sep, mean_px_val_file), 'rb') as f:
        mean_px_vals = pkl.load(f)

    mean_photon_val_file = metadata['mean_photon_val_file']
    with open('{0}{1}{2}'.format(data_path, sep, mean_photon_val_file), 'rb') as f:
        mean_photon_vals = pkl.load(f)

    fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
    mean_px_vals_all = []
    mean_photon_vals_all = []
    tvec_all = []
    t_total = 0

    for session in sessions_to_process:

        n_trials = len(list(mean_px_vals[session].keys()))
        for trial in range(n_trials):
            mean_px_vals_all = np.append(mean_px_vals_all, mean_px_vals[session][trial])
            mean_photon_vals_all = np.append(mean_photon_vals_all, mean_photon_vals[session][trial])
            ax[1, 1].plot(mean_photon_vals[session][trial])

        tvec_all = np.append(tvec_all, frame_times[session] + t_total)
        t_total = t_total + frame_times[session][-1]

    ax[1, 1].set_xlabel('Frames')
    ax[1, 1].set_ylabel('Mean photon value')

    ax[0, 0].plot(tvec_all, mean_px_vals_all)
    ax[0, 0].set_ylabel('Mean pixel value in frame')
    ax[1, 0].plot(tvec_all, mean_photon_vals_all)
    ax[1, 0].set_ylabel('Mean photon value in frame')
    ax[1, 0].set_xlabel('Time (s)')

    ax[0, 1].scatter(mean_px_vals_all, mean_photon_vals_all, marker = '.', color = 'k', alpha = 0.8)
    ax[0, 1].set_xlabel('Mean pixel value in frame')
    ax[0, 1].set_xlabel('Mean photon value in frame')

    if save_fig:
        plt.savefig('{0}{1}mean_pixel_values.png'.format(data_path, sep))
