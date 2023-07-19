from scipy.signal import correlate
from numpy.linalg import norm
from numpy.random import permutation
from scipy.signal import find_peaks
from PIL import Image

from population import clustering
from segmentation import draw_rois

from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import davies_bouldin_score as db

def cross_correlograms_session(data_path, metadata_file, cell_ids, dFF, spike_frames, tvec, frame_rate, cc_file = 'cross_correlograms.pkl', range_ms = 50, spike_bin_ms = 20, spike_jitter_ms = 5, overwrite = False, make_plots = False, save_plots = False, save_path = None):

    try:
        with open('{0}{1}{2}'.format(data_path, sep, cc_file), 'rb') as f:
            cc_dict = pkl.load(f)
        dFF_cc = cc_dict['dFF']
        spike_cc = cc_dict['spike']
        spike_dFF_cc = cc_dict['spike_dFF']
        sta_mean = cc_dict['sta_mean']
        sta_sem = cc_dict['sta_sem']
        sta_peak_frame = cc_dict['sta_peak_frame']
        sta_peak = cc_dict['sta_peak']

    except:
        overwrite = True

    if frame_rate == 0:
        frame_rate = 1/np.mean(np.diff(tvec))
    range_frames = int(range_ms*frame_rate/1000)
    spike_bin_frames = int(spike_bin_ms*frame_rate/1000)
    spike_jitter_frames = int(spike_jitter_ms*frame_rate/1000)

    if overwrite:

        n_cells = len(cell_ids)
        pair_no = 0
        n_pairs = int(n_cells*(n_cells - 1)/2)

        # dFF cross correlations
        print('     Calculating dFF-dFF cross correlations')
        assert(len(cell_ids) == dFF.shape[0])
        assert(len(tvec) == dFF.shape[1])

        dFF_cc = np.zeros([n_pairs, 2*range_frames])
        mean_dFF = np.mean(dFF, axis = 1)
        assert(len(mean_dFF) == n_cells)
        mean_dFF = np.reshape(mean_dFF, [n_cells, 1])
        dFF = dFF - mean_dFF
        for i in range(n_cells):
            for j in range(i):
                dFF_cc[pair_no, :] = cc(dFF[i, :], dFF[j, :], range_frames)
                pair_no += 1

        # Spike cross correlations
        print('     Calculating spike-spike cross correlations')
        assert(len(list(spike_frames.keys())) == len(cell_ids))
        n_frames = len(tvec)
        spike_vectors = np.zeros([n_cells, n_frames])
        #print('           Binning spikes for {0} cells'.format(n_cells))
        for cell in range(n_cells):
            spike_frames_cell = spike_frames[cell_ids[cell]]
            spike_vec = np.zeros(n_frames)
            spike_vec[spike_frames_cell.astype(int)] = np.ones(len(spike_frames_cell))
            spike_vectors[cell, :] = get_spike_rate(spike_vec, spike_bin_frames)
            #for frame in range(n_frames):
            #    first_frame = np.max([0, frame - spike_bin_frames])
            #    spike_frames_causal = spike_frames_cell[spike_frames_cell < frame]
            #    spike_vectors[cell, frame] = np.sum(spike_frames_causal > first_frame)
            spike_vectors[cell, :] = spike_vectors[cell, :]*1000/spike_bin_ms
            spike_vectors[cell, :] -= np.mean(spike_vectors[cell, :])

        spike_cc = np.zeros([n_pairs, 2*range_frames])
        pair_no = 0
        for i in range(n_cells):
            for j in range(i):
                spike_cc[pair_no, :] = cc(spike_vectors[i], spike_vectors[j], range_frames)
                pair_no += 1

        # Spike dFF cross correlation
        print('     Calculating spike-dFF cross correlations')
        spike_dFF_cc = np.zeros([n_pairs*2, 2*range_frames])
        pair_no = 0
        for i in range(n_cells):
            for j in range(i):
                spike_dFF_cc[pair_no, :] = cc(spike_vectors[i], dFF[j], range_frames)
                pair_no += 1

        for i in range(n_cells):
            for j in range(i):
                spike_dFF_cc[pair_no, :] = cc(spike_vectors[j], dFF[i], range_frames)
                pair_no += 1

        # Spike triggered average dF/F
        print('     Calculating spike triggered average dFF')
        sta_mean = np.zeros([n_pairs*2, 2*range_frames])
        sta_sem = np.zeros([n_pairs*2, 2*range_frames])
        sta_peak_frame = np.zeros(n_pairs*2)
        sta_peak = np.zeros(n_pairs*2)
        pair_no = 0
        for i in range(n_cells):
            for j in range(i):
                sta_all = sta(spike_frames[cell_ids[i]], dFF[j], range_frames)
                sta_mean[pair_no, :] = np.mean(sta_all, axis = 0)
                sta_sem[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[i]]))
                sta_peak_frame[pair_no] = get_lag_cc(sta_mean[pair_no])
                sta_peak[pair_no] = sta_mean[pair_no, :][int(sta_peak_frame[pair_no])]
                pair_no += 1

        for i in range(n_cells):
            for j in range(i):
                sta_all = sta(spike_frames[cell_ids[j]], dFF[i], range_frames)
                sta_mean[pair_no, :] = np.mean(sta_all, axis = 0)
                sta_sem[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[j]]))
                sta_peak_frame[pair_no] = get_lag_cc(sta_mean[pair_no])
                sta_peak[pair_no] = sta_mean[pair_no, :][int(sta_peak_frame[pair_no])]
                pair_no += 1

        # Null distribution of spike triggered average
        print('     Calculating spike triggered average dFF')
        sta_mean_null = np.zeros([n_pairs*2, 2*range_frames])
        sta_sem_null = np.zeros([n_pairs*2, 2*range_frames])
        sta_peak_frame_null = np.zeros(n_pairs*2)
        sta_peak_null = np.zeros(n_pairs*2)
        pair_no = 0

        spike_vectors = np.zeros([n_cells, n_frames])
        for cell in range(n_cells):
            spike_frames_cell = spike_frames[cell_ids[cell]].astype(int)
            spike_vectors[cell, spike_frames_cell] = np.ones(len(spike_frames_cell))
        n_bins = int(np.ceil(n_frames/spike_jitter_frames))
        spike_vectors_jitter = get_shuffled_activity(spike_vectors, spike_jitter_frames, n_frames, n_bins)
        spike_frames_jitter = {}
        for cell in range(n_cells):
            spike_frames_jitter[cell_ids[cell]] = np.where(spike_vectors_jitter[cell, :])[0].astype(int)

        for i in range(n_cells):
            for j in range(i):
                sta_all = sta(spike_frames_jitter[cell_ids[i]], dFF[j], range_frames)
                sta_mean_null[pair_no, :] = np.mean(sta_all, axis = 0)
                sta_sem_null[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[i]]))
                sta_peak_frame_null[pair_no] = get_lag_cc(sta_mean_null[pair_no])
                sta_peak_null[pair_no] = sta_mean_null[pair_no, :][int(sta_peak_frame_null[pair_no])]
                pair_no += 1

        for i in range(n_cells):
            for j in range(i):
                sta_all = sta(spike_frames_jitter[cell_ids[i]], dFF[j], range_frames)
                sta_mean_null[pair_no, :] = np.mean(sta_all, axis = 0)
                sta_sem_null[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[i]]))
                sta_peak_frame_null[pair_no] = get_lag_cc(sta_mean_null[pair_no])
                sta_peak_null[pair_no] = sta_mean_null[pair_no, :][int(sta_peak_frame_null[pair_no])]
                pair_no += 1

        cc_dict = {'dFF': dFF_cc, 'spike': spike_cc, 'spike_dFF': spike_dFF_cc,
                    'sta_mean': sta_mean, 'sta_sem': sta_sem, 'sta_peak_frame': sta_peak_frame, 'sta_peak': sta_peak,
                    'sta_mean_null': sta_mean_null, 'sta_sem_null': sta_sem_null, 'sta_peak_frame_null': sta_peak_frame_null, 'sta_peak_null': sta_peak_null,
                    'range_frames': range_frames,
                    'spike_bin_frames': spike_bin_frames, 'spike_bin_ms': spike_bin_ms}
        with open('{0}{1}{2}'.format(data_path, sep, cc_file), 'wb') as f:
            pkl.dump(cc_dict, f)

    if make_plots:

        # dFF cross correlation heatmap
        plt.figure()
        min_val = np.min(dFF_cc)
        max_val = np.max(dFF_cc)
        extremum = np.max([np.abs(min_val), np.abs(max_val)])
        tvec_cc = np.linspace(-range_ms, range_ms, 2*range_frames)
        plt.imshow(dFF_cc, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        plt.xticks(ticks = [0, int(range_frames/2), range_frames, int(3*range_frames/2), 2*range_frames],
                    labels = [-range_ms, int(range_ms/2), 0, int(range_ms/2), range_ms])
        plt.colorbar(label = 'dF/F cross-correlation')
        plt.ylabel('Pair #')
        plt.xlabel('Lag (ms)')
        if save_plots:
            if save_path == None:
                save_path = '{0}{1}Plots{1}Pairwise correlations'.format(data_path, sep)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            plt.savefig('{0}{1}dFF_cc_heatmap.png'.format(save_path, sep))

        # Spike cross correlation heatmap
        plt.figure()
        min_val = np.min(spike_cc)
        max_val = np.max(spike_cc)
        extremum = np.max([np.abs(min_val), np.abs(max_val)])
        tvec_cc = np.linspace(-range_ms, range_ms, 2*range_frames)
        plt.imshow(spike_cc, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        plt.xticks(ticks = [0, int(range_frames/2), range_frames, int(3*range_frames/2), 2*range_frames],
                    labels = [-range_ms, int(range_ms/2), 0, int(range_ms/2), range_ms])
        plt.colorbar(label = 'Spike cross-correlation')
        plt.ylabel('Pair #')
        plt.xlabel('Lag (ms)')
        if save_plots:
            plt.savefig('{0}{1}Spike_cc_heatmap.png'.format(save_path, sep))

        # Spike-dFF cross correlation heatmap
        plt.figure()
        min_val = np.min(spike_dFF_cc)
        max_val = np.max(spike_dFF_cc)
        extremum = np.max([np.abs(min_val), np.abs(max_val)])
        tvec_cc = np.linspace(-range_ms, range_ms, 2*range_frames)
        plt.imshow(spike_dFF_cc, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        plt.xticks(ticks = [0, int(range_frames/2), range_frames, int(3*range_frames/2), 2*range_frames],
                    labels = [-range_ms, int(range_ms/2), 0, int(range_ms/2), range_ms])
        plt.colorbar(label = 'Spike-dFF cross-correlation')
        plt.ylabel('Pair #')
        plt.xlabel('Lag (ms)')
        if save_plots:
            plt.savefig('{0}{1}Spike_dFF_cc_heatmap.png'.format(save_path, sep))

    return cc_dict

def peak_correlation(data_path, metadata_file, cc_dict, cell_ids, peaks_file = 'Cross_correlogram_peaks.pkl', overwrite = False, make_plots = False, save_plots = False):

    try:
        with open('{0}{1}{2}'.format(data_path, sep, peaks_file), 'rb') as f:
            peaks = pkl.load(f)
        dFF_peaks = peaks['dFF_peaks']
        spike_peaks = peaks['spike_peaks']
        spike_dFF_peaks = peaks['spike_dFF_peaks']
        dFF_peaks_array = peaks['dFF_peaks_array']
        spike_peaks_array = peaks['spike_peaks_array']
        spike_dFF_peaks_array = peaks['spike_dFF_peaks_array']
        dFF_lags = peaks['dFF_lags']
        spike_lags = peaks['spike_lags']
        spike_dFF_lags = peaks['spike_dFF_lags']

    except:
        overwrite = True

    if overwrite:
        peaks = {}
        n_cells = len(cell_ids)
        n_pairs = int(n_cells*(n_cells - 1)/2)
        dFF_peaks = np.zeros(n_pairs)
        spike_peaks = np.zeros(n_pairs)
        spike_dFF_peaks = np.zeros(n_pairs*2)
        dFF_peaks_array = np.zeros([n_cells, n_cells])
        spike_peaks_array = np.zeros([n_cells, n_cells])
        spike_dFF_peaks_array = np.zeros([n_cells, n_cells])
        dFF_lags = np.zeros(n_pairs)
        spike_lags = np.zeros(n_pairs)
        spike_dFF_lags = np.zeros(n_pairs*2)

        dFF_cc = cc_dict['dFF']
        spike_cc = cc_dict['spike']
        spike_dFF_cc = cc_dict['spike_dFF']

        pair_no = 0
        for i in range(n_cells):
            for j in range(i):
                dFF_lags[pair_no] = get_lag_cc(dFF_cc[pair_no, :])
                dFF_peaks[pair_no] = dFF_cc[pair_no, int(dFF_lags[pair_no])]
                dFF_peaks_array[i, j] = dFF_peaks[pair_no]
                dFF_peaks_array[j, i] = dFF_peaks[pair_no]

                spike_lags[pair_no] = get_lag_cc(spike_cc[pair_no, :])
                spike_peaks[pair_no] = spike_cc[pair_no, int(spike_lags[pair_no])]
                spike_peaks_array[i, j] = spike_peaks[pair_no]
                spike_peaks_array[j, i] = spike_peaks[pair_no]

                spike_dFF_lags[pair_no] = get_lag_cc(spike_dFF_cc[pair_no, :])
                spike_dFF_peaks[pair_no] = spike_dFF_cc[pair_no, int(spike_dFF_lags[pair_no])]
                spike_dFF_peaks_array[i, j] = spike_dFF_peaks[pair_no]

                pair_no += 1

        for i in range(n_cells):
            for j in range(i):
                spike_dFF_lags[pair_no] = get_lag_cc(spike_dFF_cc[pair_no, :])
                spike_dFF_peaks[pair_no] = spike_dFF_cc[pair_no, int(spike_dFF_lags[pair_no])]
                spike_dFF_peaks_array[j, i] = spike_dFF_peaks[pair_no]

                pair_no += 1

        peaks['dFF_peaks'] = dFF_peaks
        peaks['spike_peaks'] = spike_peaks
        peaks['spike_dFF_peaks'] = spike_dFF_peaks
        peaks['dFF_peaks_array'] = dFF_peaks_array
        peaks['spike_peaks_array'] = spike_peaks_array
        peaks['spike_dFF_peaks_array'] = spike_dFF_peaks_array
        peaks['dFF_lags'] = dFF_lags
        peaks['spike_lags'] = spike_lags
        peaks['spike_dFF_lags'] = spike_dFF_lags

        with open('{0}{1}{2}'.format(data_path, sep, peaks_file), 'wb') as f:
            pkl.dump(peaks, f)

    return peaks

def cc_significance(data_path, metadata_file, cc_dict, peaks, cell_ids, dFF, spike_frames, tvec, frame_rate, distance, sig_file = 'cc_significance', n_iter = 1000, spike_jitter_ms = 5, dFF_jitter_ms = 10, pval_thresh = 0.05, show_cell_labels = True, make_plots = False, overwrite = False, save_plots = False, save_path = None, sig_color_pos = 'r', sig_color_neg = 'b', sig_color = 'r'):

    try:
        with open('{0}{1}{3}_{2}_iters.pkl'.format(data_path, sep, n_iter, sig_file), 'rb') as f:
            sig_dict = pkl.load(f)
        dFF_pvals = sig_dict['dFF_pvals']
        spike_pvals = sig_dict['spike_pvals']
        spike_dFF_pvals = sig_dict['spike_dFF_pvals']
        dFF_peaks_jitter = sig_dict['dFF_peaks_jitter']
        spike_peaks_jitter = sig_dict['spike_peaks_jitter']
        spike_dFF_peaks_jitter = sig_dict['spike_dFF_peaks_jitter']
        dFF_peaks_array_sig = sig_dict['dFF_peaks_array_sig']
        spike_peaks_array_sig = sig_dict['spike_peaks_array_sig']
        spike_dFF_peaks_array_sig = sig_dict['spike_dFF_peaks_array_sig']

    except:
        overwrite = True

    n_cells = len(cell_ids)
    n_pairs = int(n_cells*(n_cells - 1)/2)
    if frame_rate == 0:
        frame_rate = 1/(np.mean(np.diff(tvec)))
    range_frames = cc_dict['range_frames']

    if overwrite:

        dFF_pvals = np.zeros(n_pairs)
        spike_pvals = np.zeros(n_pairs)
        spike_dFF_pvals = np.zeros(n_pairs*2)
        dFF_peaks_jitter = np.zeros([n_pairs, n_iter])
        spike_peaks_jitter = np.zeros([n_pairs, n_iter])
        spike_dFF_peaks_jitter = np.zeros([n_pairs*2, n_iter])
        dFF_peaks_array_sig = np.zeros([n_cells, n_cells])
        spike_peaks_array_sig = np.zeros([n_cells, n_cells])
        spike_dFF_peaks_array_sig = np.zeros([n_cells, n_cells])

        dFF_jitter_frames = int(dFF_jitter_ms*frame_rate/1000)
        spike_jitter_frames = int(spike_jitter_ms*frame_rate/1000)
        range_frames = cc_dict['range_frames']
        spike_bin_frames = cc_dict['spike_bin_frames']
        spike_bin_ms = cc_dict['spike_bin_ms']

        mean_dFF = np.mean(dFF, axis = 1)
        assert(len(mean_dFF) == n_cells)
        mean_dFF = np.reshape(mean_dFF, [n_cells, 1])
        dFF = dFF - mean_dFF

        n_frames = len(tvec)

        assert(len(list(spike_frames.keys())) == len(cell_ids))
        print('     Binning spikes for {0} cells'.format(n_cells))
        spike_vectors = np.zeros([n_cells, n_frames])
        for cell in range(n_cells):
            spike_frames_cell = spike_frames[cell_ids[cell]]
            for frame in range(n_frames):
                first_frame = np.max([0, frame - spike_bin_frames])
                spike_frames_causal = spike_frames_cell[spike_frames_cell < frame]
                spike_vectors[cell, frame] = np.sum(spike_frames_causal > first_frame)
            spike_vectors[cell, :] = spike_vectors[cell, :]*1000/spike_bin_ms
            spike_vectors[cell, :] -= np.mean(spike_vectors[cell, :])

        frame_order = np.arange(0, n_frames)
        n_bins_dFF = int(np.ceil(n_frames/dFF_jitter_frames))
        n_bins_spike = int(np.ceil(n_frames/spike_jitter_frames))

        print('     Calculating significance with {0} bootstrap iterations'.format(n_iter))
        for iter in tqdm(range(n_iter)):

            dFF_jitter = get_shuffled_activity(dFF, dFF_jitter_frames, n_frames, n_bins_dFF)
            spike_vectors_jitter = get_shuffled_activity(spike_vectors, spike_jitter_frames, n_frames, n_bins_spike)

            pair_no = 0
            for i in range(n_cells):
                for j in range(i):

                    dFF_cc_jitter = cc(dFF[i, :], dFF_jitter[j, :], range_frames)
                    lag = get_lag_cc(dFF_cc_jitter)
                    dFF_peaks_jitter[pair_no, iter] = dFF_cc_jitter[lag]

                    spike_cc_jitter = cc(spike_vectors[i], spike_vectors_jitter[j], range_frames)
                    lag = get_lag_cc(spike_cc_jitter)
                    spike_peaks_jitter[pair_no, iter] = spike_cc_jitter[lag]

                    spike_dFF_cc_jitter = cc(spike_vectors[i], dFF_jitter[j], range_frames)
                    lag = get_lag_cc(spike_dFF_cc_jitter)
                    spike_dFF_peaks_jitter[pair_no, iter] = spike_dFF_cc_jitter[lag]

                    pair_no += 1

            for i in range(n_cells):
                for j in range(i):

                    spike_dFF_cc_jitter = cc(spike_vectors[j], dFF_jitter[i], range_frames)
                    lag = get_lag_cc(spike_dFF_cc_jitter)
                    spike_dFF_peaks_jitter[pair_no, iter] = spike_dFF_cc_jitter[lag]
                    pair_no += 1

        pair_no = 0
        for i in range(n_cells):
            for j in range(i):

                dFF_pvals[pair_no] = np.sum(np.abs(dFF_peaks_jitter[pair_no, :]) >= np.abs(peaks['dFF_peaks'][pair_no]))/n_iter
                if dFF_pvals[pair_no] < pval_thresh:
                    dFF_peaks_array_sig[i, j] = peaks['dFF_peaks'][pair_no]
                    dFF_peaks_array_sig[j, i] = peaks['dFF_peaks'][pair_no]

                spike_pvals[pair_no] = np.sum(np.abs(spike_peaks_jitter[pair_no, :]) >= np.abs(peaks['spike_peaks'][pair_no]))/n_iter
                if spike_pvals[pair_no] < pval_thresh:
                    spike_peaks_array_sig[i, j] = peaks['spike_peaks'][pair_no]
                    spike_peaks_array_sig[j, i] = peaks['spike_peaks'][pair_no]

                spike_dFF_pvals[pair_no] = np.sum(np.abs(spike_dFF_peaks_jitter[pair_no, :]) >= np.abs(peaks['spike_dFF_peaks'][pair_no]))/n_iter
                if spike_dFF_pvals[pair_no] < pval_thresh:
                    spike_dFF_peaks_array_sig[i, j] = peaks['spike_dFF_peaks'][pair_no]

                pair_no += 1

        for i in range(n_cells):
            for j in range(i):
                spike_dFF_pvals[pair_no] = np.sum(np.abs(spike_dFF_peaks_jitter[pair_no, :]) >= np.abs(peaks['spike_dFF_peaks'][pair_no]))/n_iter
                if spike_dFF_pvals[pair_no] < pval_thresh:
                    spike_dFF_peaks_array_sig[j, i] = peaks['spike_dFF_peaks'][pair_no]

                pair_no += 1

        sig_dict = {}
        sig_dict['dFF_pvals'] = dFF_pvals
        sig_dict['spike_pvals'] = spike_pvals
        sig_dict['spike_dFF_pvals'] = spike_dFF_pvals
        sig_dict['dFF_peaks_jitter'] = dFF_peaks_jitter
        sig_dict['spike_peaks_jitter'] = spike_peaks_jitter
        sig_dict['spike_dFF_peaks_jitter'] = spike_dFF_peaks_jitter
        sig_dict['dFF_peaks_array_sig'] = dFF_peaks_array_sig
        sig_dict['spike_peaks_array_sig'] = spike_peaks_array_sig
        sig_dict['spike_dFF_peaks_array_sig'] = spike_dFF_peaks_array_sig

        with open('{0}{1}{3}_{2}_iters.pkl'.format(data_path, sep, n_iter, sig_file), 'wb') as f:
            pkl.dump(sig_dict, f)

    if make_plots:

        # Observed vs shuffled peak correlation
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5], constrained_layout = True)
        for pair_no in range(n_pairs):
            if dFF_pvals[pair_no] < pval_thresh:
                ax[0].scatter(np.ones(n_iter)*peaks['dFF_peaks'][pair_no], dFF_peaks_jitter[pair_no, :],
                            color = sig_color, marker = '.', alpha = 0.01)
            else:
                ax[0].scatter(np.ones(n_iter)*peaks['dFF_peaks'][pair_no], dFF_peaks_jitter[pair_no, :],
                                color = 'k', marker = '.', alpha = 0.01)

            if spike_pvals[pair_no] < pval_thresh:
                ax[1].scatter(np.ones(n_iter)*peaks['spike_peaks'][pair_no], spike_peaks_jitter[pair_no, :],
                              color = sig_color, marker = '.', alpha = 0.01)
            else:
                ax[1].scatter(np.ones(n_iter)*peaks['spike_peaks'][pair_no], spike_peaks_jitter[pair_no, :],
                                  color = 'k', marker = '.', alpha = 0.01)

            if spike_dFF_pvals[pair_no] < pval_thresh:
                ax[2].scatter(np.ones(n_iter)*peaks['spike_dFF_peaks'][pair_no], spike_dFF_peaks_jitter[pair_no, :],
                                  color = sig_color, marker = '.', alpha = 0.01)
            else:
                ax[2].scatter(np.ones(n_iter)*peaks['spike_dFF_peaks'][pair_no], spike_dFF_peaks_jitter[pair_no, :],
                                      color = 'k', marker = '.', alpha = 0.01)
        for pair_no in range(n_pairs, 2*n_pairs):
            if spike_dFF_pvals[pair_no] < pval_thresh:
                ax[2].scatter(np.ones(n_iter)*peaks['spike_dFF_peaks'][pair_no], spike_dFF_peaks_jitter[pair_no, :],
                                  color = sig_color, marker = '.', alpha = 0.01)
            else:
                ax[2].scatter(np.ones(n_iter)*peaks['spike_dFF_peaks'][pair_no], spike_dFF_peaks_jitter[pair_no, :],
                                      color = 'k', marker = '.', alpha = 0.01)

        for axis in [ax[0], ax[1], ax[2]]:
            xlim = axis.get_xlim()
            axis.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color = 'gray', linewidth = 0.8, linestyle = '--')
            ylim = axis.get_ylim()
            axis.plot([0, 0], [ylim[0], ylim[1]], color = 'gray', linewidth = 0.8, linestyle = '--')

        ax[0].set_title('dFF')
        ax[1].set_title('Spikes')
        ax[2].set_title('Spike - dFF')

        ax[0].set_xlabel('Peak correlation')
        ax[1].set_xlabel('Peak correlation')
        ax[2].set_xlabel('Peak correlation')
        ax[0].set_ylabel('Shuffled correlation')

        if save_plots:
            if save_path == None:
                save_path = '{0}{1}Plots{1}Pairwise correlations'.format(data_path, sep)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            fig.savefig('{0}{1}Shuffled_vs_observed_peak_correlation.png'.format(save_path, sep))

        # Peak correlation vs lag
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5], constrained_layout = True)

        ax[0].scatter((peaks['dFF_lags'] - range_frames)*1000/frame_rate, peaks['dFF_peaks'], color = 'k', marker = '.')
        dFF_sig = np.where(dFF_pvals < pval_thresh)[0]
        dFF_sig_pos = np.where(peaks['dFF_peaks'][dFF_sig] > 0)[0]
        dFF_sig_neg = np.where(peaks['dFF_peaks'][dFF_sig] < 0)[0]
        ax[0].scatter((peaks['dFF_lags'][dFF_sig][dFF_sig_pos] - range_frames)*1000/frame_rate, peaks['dFF_peaks'][dFF_sig][dFF_sig_pos], color = sig_color_pos, marker = '.')
        ax[0].scatter((peaks['dFF_lags'][dFF_sig][dFF_sig_neg] - range_frames)*1000/frame_rate, peaks['dFF_peaks'][dFF_sig][dFF_sig_neg], color = sig_color_neg, marker = '.')

        ax[1].scatter((peaks['spike_lags'] - range_frames)*1000/frame_rate, peaks['spike_peaks'], color = 'k', marker = '.')
        spike_sig = np.where(spike_pvals < pval_thresh)[0]
        spike_sig_pos = np.where(peaks['spike_peaks'][spike_sig] > 0)[0]
        spike_sig_neg = np.where(peaks['spike_peaks'][spike_sig] < 0)[0]
        ax[1].scatter((peaks['spike_lags'][spike_sig][spike_sig_pos] - range_frames)*1000/frame_rate, peaks['spike_peaks'][spike_sig][spike_sig_pos], color = sig_color_pos, marker = '.')
        ax[1].scatter((peaks['spike_lags'][spike_sig][spike_sig_neg] - range_frames)*1000/frame_rate, peaks['spike_peaks'][spike_sig][spike_sig_neg], color = sig_color_neg, marker = '.')

        ax[2].scatter((peaks['spike_dFF_lags'] - range_frames)*1000/frame_rate, peaks['spike_dFF_peaks'], color = 'k', marker = '.')
        spike_dFF_sig = np.where(spike_dFF_pvals < pval_thresh)[0]
        spike_dFF_sig_pos = np.where(peaks['spike_dFF_peaks'][spike_dFF_sig] > 0)[0]
        spike_dFF_sig_neg = np.where(peaks['spike_dFF_peaks'][spike_dFF_sig] < 0)[0]
        ax[2].scatter((peaks['spike_dFF_lags'][spike_dFF_sig][spike_dFF_sig_pos] - range_frames)*1000/frame_rate, peaks['spike_dFF_peaks'][spike_dFF_sig][spike_dFF_sig_pos], color = sig_color_pos, marker = '.')
        ax[2].scatter((peaks['spike_dFF_lags'][spike_dFF_sig][spike_dFF_sig_neg] - range_frames)*1000/frame_rate, peaks['spike_dFF_peaks'][spike_dFF_sig][spike_dFF_sig_neg], color = sig_color_neg, marker = '.')

        ax[0].set_title('dFF')
        ax[1].set_title('Spikes')
        ax[2].set_title('Spike - dFF')

        ax[0].set_ylabel('Peak correlation')
        for axis in [ax[0], ax[1], ax[2]]:
            axis.set_xlabel('Lag (ms)')

        if save_plots:
            fig.savefig('{0}{1}Peak_correlation_vs_lag.png'.format(save_path, sep))

        # Peak correlation vs distance
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5], constrained_layout = True)

        ax[0].scatter(distance, peaks['dFF_peaks'], color = 'k', marker = '.')
        ax[0].scatter(distance[dFF_sig][dFF_sig_pos], peaks['dFF_peaks'][dFF_sig][dFF_sig_pos], color = sig_color_pos, marker = '.')
        ax[0].scatter(distance[dFF_sig][dFF_sig_neg], peaks['dFF_peaks'][dFF_sig][dFF_sig_neg], color = sig_color_neg, marker = '.')

        ax[1].scatter(distance, peaks['spike_peaks'], color = 'k', marker = '.')
        ax[1].scatter(distance[spike_sig][spike_sig_pos], peaks['spike_peaks'][spike_sig][spike_sig_pos], color = sig_color_pos, marker = '.')
        ax[1].scatter(distance[spike_sig][spike_sig_neg], peaks['spike_peaks'][spike_sig][spike_sig_neg], color = sig_color_neg, marker = '.')

        d2 = np.concatenate([distance, distance])
        ax[2].scatter(d2, peaks['spike_dFF_peaks'], color = 'k', marker = '.')
        ax[2].scatter(d2[spike_dFF_sig][spike_dFF_sig_pos], peaks['spike_dFF_peaks'][spike_dFF_sig][spike_dFF_sig_pos], color = sig_color_pos, marker = '.')
        ax[2].scatter(d2[spike_dFF_sig][spike_dFF_sig_neg], peaks['spike_dFF_peaks'][spike_dFF_sig][spike_dFF_sig_neg], color = sig_color_neg, marker = '.')

        ax[0].set_ylabel('Peak correlation')
        for axis in [ax[0], ax[1], ax[2]]:
            axis.set_xlabel('Distance (um)')

        ax[0].set_title('dFF')
        ax[1].set_title('Spikes')
        ax[2].set_title('Spike - dFF')

        if save_plots:
            fig.savefig('{0}{1}Peak_correlation_vs_distance.png'.format(save_path, sep))

        # Peak correlation matrix
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5], constrained_layout = True)

        min_val = np.min([np.min(dFF_peaks_array_sig), np.min(spike_peaks_array_sig), np.min(spike_dFF_peaks_array_sig)])
        max_val = np.max([np.max(dFF_peaks_array_sig), np.max(spike_peaks_array_sig), np.max(spike_dFF_peaks_array_sig)])
        extremum = np.max(np.abs([min_val, max_val]))

        ax[0].imshow(dFF_peaks_array_sig, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        ax[1].imshow(spike_peaks_array_sig, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        heatmap = ax[2].imshow(spike_dFF_peaks_array_sig, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        plt.colorbar(mappable = heatmap, label = 'Peak correlation')

        for axis in [ax[0], ax[1], ax[2]]:
            axis.set_ylabel('Cell #')
            axis.set_xlabel('Cell #')
            if show_cell_labels:
                axis.set_xticks(np.linspace(0, n_cells -1 , n_cells))
                axis.set_xticklabels((cell_ids + 1).astype(int))
                axis.set_yticks(np.linspace(0, n_cells -1, n_cells))
                axis.set_yticklabels((cell_ids + 1).astype(int))

        ax[0].set_title('dFF')
        ax[1].set_title('Spikes')
        ax[2].set_title('Spike - dFF')

        if save_plots:
            fig.savefig('{0}{1}Peak_correlation_heatmap.png'.format(save_path, sep))

    return sig_dict

def sta_spike_rate(data_path, metadata_file, cc_dict, cell_ids, spike_frames, tvec, frame_rate, calculate_peak = False, sub_baseline = True, sta_dur_ms = 10, range_ms = 50, spike_bin_ms = 100,  cc_file = 'cross_correlograms.pkl',):

    if frame_rate == 0:
        frame_rate = 1/np.mean(np.diff(tvec))
    range_frames = int(range_ms*frame_rate/1000)
    spike_bin_frames = int(spike_bin_ms*frame_rate/1000)
    sta_dur_frames = int(sta_dur_ms*frame_rate/1000)

    n_cells = len(cell_ids)
    pair_no = 0
    n_pairs = int(n_cells*(n_cells - 1)/2)

    assert(len(list(spike_frames.keys())) == len(cell_ids))
    n_frames = len(tvec)
    spike_rate = np.zeros([n_cells, n_frames])
    #print('           Binning spikes for {0} cells'.format(n_cells))
    for cell in range(n_cells):
        spike_frames_cell = spike_frames[cell_ids[cell]]
        spike_vector = np.zeros(n_frames)
        spike_vector[spike_frames_cell.astype(int)] = np.ones(len(spike_frames_cell))
        spike_rate[cell, :] = get_spike_rate(spike_vector, spike_bin_frames)
        spike_rate[cell, :] = spike_rate[cell, :]*1000/spike_bin_ms
        spike_rate[cell, :] -= np.mean(spike_rate[cell, :])

    # Spike triggered average spike rate
    print('     Calculating spike triggered average spike rate')
    sta_mean = np.zeros([n_pairs*2, 2*range_frames])
    sta_sem = np.zeros([n_pairs*2, 2*range_frames])
    sta_peak_frame = np.zeros(n_pairs*2)
    sta_peak = np.zeros(n_pairs*2)
    pair_no = 0
    for i in range(n_cells):
        for j in range(i):
            sta_all = sta(spike_frames[cell_ids[i]], spike_rate[j], range_frames)
            sta_mean[pair_no, :] = np.mean(sta_all, axis = 0)
            sta_sem[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[i]]))
            if calculate_peak:
                sta_peak_frame[pair_no] = get_lag_cc(sta_mean[pair_no])
                sta_peak[pair_no] = sta_mean[pair_no, :][int(sta_peak_frame[pair_no])]
            else:
                sta_peak_frame[pair_no] = range_frames
                if sub_baseline:
                    baseline = np.mean(sta_all[:, :range_frames], axis = 1)
                    response = np.mean(sta_all[:, range_frames:range_frames + sta_dur_frames], axis = 1)
                    assert(len(baseline) == sta_all.shape[0])
                    assert(len(response) == sta_all.shape[0])
                    sta_peak[pair_no] = np.mean(response - baseline)
                else:
                    sta_peak[pair_no] = np.mean(sta_mean[pair_no, range_frames:range_frames + sta_dur_frames])
            pair_no += 1

    for i in range(n_cells):
        for j in range(i):
            sta_all = sta(spike_frames[cell_ids[j]], spike_rate[i], range_frames)
            sta_mean[pair_no, :] = np.mean(sta_all, axis = 0)
            sta_sem[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[j]]))
            if calculate_peak:
                sta_peak_frame[pair_no] = get_lag_cc(sta_mean[pair_no])
                sta_peak[pair_no] = sta_mean[pair_no, :][int(sta_peak_frame[pair_no])]
            else:
                sta_peak_frame[pair_no] = range_frames
                if sub_baseline:
                    baseline = np.mean(sta_all[:, :range_frames], axis = 1)
                    response = np.mean(sta_all[:, range_frames:range_frames + sta_dur_frames], axis = 1)
                    assert(len(baseline) == sta_all.shape[0])
                    assert(len(response) == sta_all.shape[0])
                    sta_peak[pair_no] = np.mean(response - baseline)
                else:
                    sta_peak[pair_no] = np.mean(sta_mean[pair_no, range_frames:range_frames + sta_dur_frames])
            pair_no += 1

    cc_dict['sta_spike_rate_mean'] = sta_mean
    cc_dict['sta_spike_rate_sem'] = sta_sem
    cc_dict['sta_spike_rate_peak'] = sta_peak
    cc_dict['sta_spike_rate_peak_frame'] = sta_peak_frame

    with open('{0}{1}{2}'.format(data_path, sep, cc_file), 'wb') as f:
        pkl.dump(cc_dict, f)

    return cc_dict

def sta_raw_dFF(data_path, metadata_file, cc_dict, cell_ids, spike_frames, raw_dFF, tvec, frame_rate, calculate_peak = True, sub_baseline = False, sta_dur_ms = 10, range_ms = 50, cc_file = 'cross_correlograms.pkl',):

    if frame_rate == 0:
        frame_rate = 1/np.mean(np.diff(tvec))
    range_frames = int(range_ms*frame_rate/1000)
    sta_dur_frames = int(sta_dur_ms*frame_rate/1000)

    n_cells = len(cell_ids)
    pair_no = 0
    n_pairs = int(n_cells*(n_cells - 1)/2)

    assert(len(list(spike_frames.keys())) == len(cell_ids))
    assert(raw_dFF.shape[0] == n_cells)
    n_frames = len(tvec)
    assert(raw_dFF.shape[1] == n_frames)

    # Spike triggered average of raw dF/F
    print('     Calculating spike triggered average of raw dF/F')
    sta_mean = np.zeros([n_pairs*2, 2*range_frames])
    sta_sem = np.zeros([n_pairs*2, 2*range_frames])
    sta_peak_frame = np.zeros(n_pairs*2)
    sta_peak = np.zeros(n_pairs*2)
    pair_no = 0
    for i in range(n_cells):
        for j in range(i):
            sta_all = sta(spike_frames[cell_ids[i]], raw_dFF[j, :], range_frames)
            sta_mean[pair_no, :] = np.mean(sta_all, axis = 0)
            sta_sem[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[i]]))
            if calculate_peak:
                sta_peak_frame[pair_no] = get_lag_cc(sta_mean[pair_no])
                sta_peak[pair_no] = sta_mean[pair_no, :][int(sta_peak_frame[pair_no])]
            else:
                sta_peak_frame[pair_no] = range_frames
                if sub_baseline:
                    baseline = np.mean(sta_all[:, :range_frames], axis = 1)
                    response = np.mean(sta_all[:, range_frames:range_frames + sta_dur_frames], axis = 1)
                    assert(len(baseline) == sta_all.shape[0])
                    assert(len(response) == sta_all.shape[0])
                    sta_peak[pair_no] = np.mean(response - baseline)
                else:
                    sta_peak[pair_no] = np.mean(sta_mean[pair_no, range_frames:range_frames + sta_dur_frames])
            pair_no += 1

    for i in range(n_cells):
        for j in range(i):
            sta_all = sta(spike_frames[cell_ids[j]], raw_dFF[i, :], range_frames)
            sta_mean[pair_no, :] = np.mean(sta_all, axis = 0)
            sta_sem[pair_no, :] = np.std(sta_all, axis = 0)/np.sqrt(len(spike_frames[cell_ids[j]]))
            if calculate_peak:
                sta_peak_frame[pair_no] = get_lag_cc(sta_mean[pair_no])
                sta_peak[pair_no] = sta_mean[pair_no, :][int(sta_peak_frame[pair_no])]
            else:
                sta_peak_frame[pair_no] = range_frames
                if sub_baseline:
                    baseline = np.mean(sta_all[:, :range_frames], axis = 1)
                    response = np.mean(sta_all[:, range_frames:range_frames + sta_dur_frames], axis = 1)
                    assert(len(baseline) == sta_all.shape[0])
                    assert(len(response) == sta_all.shape[0])
                    sta_peak[pair_no] = np.mean(response - baseline)
                else:
                    sta_peak[pair_no] = np.mean(sta_mean[pair_no, range_frames:range_frames + sta_dur_frames])
            pair_no += 1

    cc_dict['sta_raw_dFF_mean'] = sta_mean
    cc_dict['sta_raw_dFF_sem'] = sta_sem
    cc_dict['sta_raw_dFF_peak'] = sta_peak
    cc_dict['sta_raw_dFF_peak_frame'] = sta_peak_frame

    with open('{0}{1}{2}'.format(data_path, sep, cc_file), 'wb') as f:
        pkl.dump(cc_dict, f)

    return cc_dict

def prop_connect(data_path, metadata_file, cc_dict, peaks, cell_ids, sig_dict, frame_rate, overwrite = False):

    try:
        with open('{0}{1}synaptic_connections.pkl'.format(data_path, sep), 'rb') as f:
            connections = pkl.load(f)
    except:
        overwrite = True

    n_cells = len(cell_ids)
    n_pairs = int(n_cells*(n_cells - 1)/2)

    if overwrite:

        connections = np.zeros(n_pairs*2)

        pair_no = 0
        for i in range(n_cells):
            for j in range(i):

                sta = cc_dict['sta_mean'][pair_no, :]
                peak_spike_dFF_cc = peaks['spike_dFF_peaks'][pair_no]
                spike_dFF_pval = sig_dict['spike_dFF_pvals'][pair_no]
                connections[pair_no] = connection(sta, peak_spike_dFF_cc, spike_dFF_pval, frame_rate)
                pair_no += 1

        for i in range(n_cells):
            for j in range(i):

                sta = cc_dict['sta_mean'][pair_no, :]
                peak_spike_dFF_cc = peaks['spike_dFF_peaks'][pair_no]
                spike_dFF_pval = sig_dict['spike_dFF_pvals'][pair_no]
                connections[pair_no] = connection(sta, peak_spike_dFF_cc, spike_dFF_pval, frame_rate)
                pair_no += 1

        with open('{0}{1}synaptic_connections.pkl'.format(data_path, sep), 'wb') as f:
            pkl.dump(connections, f)

    print('      {0}% of pairs are likely to be synaptically connected'.format(np.round((np.sum(connections)/(n_pairs*2))*100, 2)))
    return connections

def cluster_session(data_path, metadata_file, cell_ids, sig_dict, select_n_clusters = True, n_iter =10, n_iter_clust_score = 10000, method = 'hierarchical', n_clusters = 2, make_plots = False, save_plots = False, save_path = None, show_cell_labels = True):

    if select_n_clusters:

        max_n_clust = int(len(cell_ids)/2)
        n_clust = list(range(2, max_n_clust))

        fig, ax = plt.subplots(nrows = 1, ncols = 3, sharey = False, constrained_layout = True, figsize = [12, 3])
        ax[0].set_ylabel('K means clustering score')
        for x in [0, 1, 2]:
            ax[x].set_xlabel('Number of clusters')

        # dF/F correlations
        scores = clustering.select_n_clusters(sig_dict['dFF_peaks_array_sig'], max_n_clust, n_iter = n_iter)
        ax[0].errorbar(n_clust, np.mean(scores, axis = 0), yerr = np.std(scores, axis = 0), color = 'k')
        ax[0].set_title('dF/F')
        ax2 = ax[0].twinx()
        ax2.plot(n_clust[1:], np.diff(np.mean(scores, axis = 0)), color = 'gray')
        ax2.tick_params(axis = 'y', colors = 'gray')

        # spike - spike correlations
        scores = clustering.select_n_clusters(sig_dict['spike_peaks_array_sig'], max_n_clust, n_iter = n_iter)
        ax[1].errorbar(n_clust, np.mean(scores, axis = 0), yerr = np.std(scores, axis = 0), color = 'k')
        ax[1].set_title('Spikes')
        ax2 = ax[1].twinx()
        ax2.plot(n_clust[1:], np.diff(np.mean(scores, axis = 0)), color = 'gray')
        ax2.tick_params(axis = 'y', colors = 'gray')

        # Spike - dF/F correlations
        scores = clustering.select_n_clusters(sig_dict['spike_dFF_peaks_array_sig'], max_n_clust, n_iter = n_iter)
        ax[2].errorbar(n_clust, np.mean(scores, axis = 0), yerr = np.std(scores, axis = 0), color = 'k')
        ax[2].set_title('Spike - dF/F')
        ax2 = ax[2].twinx()
        ax2.plot(n_clust[1:], np.diff(np.mean(scores, axis = 0)), color = 'gray')
        ax2.tick_params(axis = 'y', colors = 'gray')
        ax2.set_ylabel('Derivative of score', color = 'gray')

        if save_plots:
            if save_path == None:
                save_path = '{0}{1}Plots{1}Pairwise correlations'.format(data_path, sep)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            plt.savefig('{0}{1}kmeans_cluster_scores.png'.format(save_path, sep, method))

        #n_clusters = int(input('Number of clusters:'))

    if method == 'hierarchical':
        dFF_clusters = clustering.hier_clust(sig_dict['dFF_peaks_array_sig'], n_clusters)
        dFF_cluster_labels = dFF_clusters.labels_
        spike_clusters = clustering.hier_clust(sig_dict['spike_peaks_array_sig'], n_clusters)
        spike_cluster_labels = spike_clusters.labels_
        spike_dFF_clusters = clustering.hier_clust(sig_dict['spike_dFF_peaks_array_sig'], n_clusters)
        spike_dFF_cluster_labels = spike_dFF_clusters.labels_

    else:
        if method == 'k-means':
            dFF_cluster_labels = clustering.k_means_clust(sig_dict['dFF_peaks_array_sig'], n_clusters)
            spike_cluster_labels = clustering.k_means_clust(sig_dict['spike_peaks_array_sig'], n_clusters)
            spike_dFF_cluster_labels = clustering.k_means_clust(sig_dict['spike_dFF_peaks_array_sig'], n_clusters)

    dFF_score = db(sig_dict['dFF_peaks_array_sig'], dFF_cluster_labels)
    spike_score = db(sig_dict['spike_peaks_array_sig'], spike_cluster_labels)
    spike_dFF_score = db(sig_dict['spike_dFF_peaks_array_sig'], spike_dFF_cluster_labels)

    # P-value of score
    print('     Calculating p value of clustering score')
    dFF_null_scores = np.zeros(n_iter_clust_score)
    spike_null_scores = np.zeros(n_iter_clust_score)
    spike_dFF_null_scores = np.zeros(n_iter_clust_score)
    for iter in tqdm(range(n_iter_clust_score)):
        cluster_labels = permutation(dFF_cluster_labels)
        dFF_null_scores[iter] = db(sig_dict['dFF_peaks_array_sig'], cluster_labels)
        cluster_labels = permutation(spike_cluster_labels)
        spike_null_scores[iter] = db(sig_dict['spike_peaks_array_sig'], cluster_labels)
        cluster_labels = permutation(spike_dFF_cluster_labels)
        spike_dFF_null_scores[iter] = db(sig_dict['spike_dFF_peaks_array_sig'], cluster_labels)

    dFF_score_pval = np.sum(dFF_null_scores < dFF_score)/n_iter_clust_score
    spike_score_pval = np.sum(spike_null_scores < spike_score)/n_iter_clust_score
    spike_dFF_score_pval = np.sum(spike_dFF_null_scores < spike_dFF_score)/n_iter_clust_score

    if make_plots:

        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 5], constrained_layout = True)

        n_cells = len(cell_ids)

        min_val = np.min([np.min(sig_dict['dFF_peaks_array_sig']), np.min(sig_dict['spike_peaks_array_sig']),
                            np.min(sig_dict['spike_dFF_peaks_array_sig'])])
        max_val = np.max([np.max(sig_dict['dFF_peaks_array_sig']), np.max(sig_dict['spike_peaks_array_sig']),
                            np.max(sig_dict['spike_dFF_peaks_array_sig'])])
        extremum = np.max(np.abs([min_val, max_val]))

        dFF_order = np.argsort(dFF_cluster_labels)
        dFF_peaks_clust_order = np.zeros([n_cells, n_cells])
        for i in range(n_cells):
            for j in range(n_cells):
                dFF_peaks_clust_order[i, j] = sig_dict['dFF_peaks_array_sig'][dFF_order[i], dFF_order[j]]
        dFF_clust_boundaries = np.where(np.abs(np.diff(np.sort(dFF_cluster_labels))))[0]

        ax[0].imshow(dFF_peaks_clust_order, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()
        for cb in dFF_clust_boundaries:
            ax[0].plot([cb + 0.5, cb + 0.5], [ylim[0], ylim[1]], color = 'k', linestyle = '--')
            ax[0].plot([xlim[0], xlim[1]], [cb + 0.5, cb + 0.5], color = 'k', linestyle = '--')

        spike_order = np.argsort(spike_cluster_labels)
        spike_peaks_clust_order = np.zeros([n_cells, n_cells])
        for i in range(n_cells):
            for j in range(n_cells):
                spike_peaks_clust_order[i, j] = sig_dict['spike_peaks_array_sig'][spike_order[i], spike_order[j]]
        spike_clust_boundaries = np.where(np.abs(np.diff(np.sort(spike_cluster_labels))))[0]

        ax[1].imshow(spike_peaks_clust_order, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        xlim = ax[1].get_xlim()
        ylim = ax[1].get_ylim()
        for cb in spike_clust_boundaries:
            ax[1].plot([cb + 0.5, cb + 0.5], [ylim[0], ylim[1]], color = 'k', linestyle = '--')
            ax[1].plot([xlim[0], xlim[1]], [cb + 0.5, cb + 0.5], color = 'k', linestyle = '--')

        spike_dFF_order = np.argsort(spike_dFF_cluster_labels)
        spike_dFF_peaks_clust_order = np.zeros([n_cells, n_cells])
        for i in range(n_cells):
            for j in range(n_cells):
                spike_dFF_peaks_clust_order[i, j] = sig_dict['spike_dFF_peaks_array_sig'][spike_dFF_order[i], spike_dFF_order[j]]
        spike_dFF_clust_boundaries = np.where(np.abs(np.diff(np.sort(spike_dFF_cluster_labels))))[0]

        heatmap = ax[2].imshow(spike_dFF_peaks_clust_order, cmap = 'bwr', vmin = -extremum, vmax = extremum, aspect = 'auto')
        plt.colorbar(mappable = heatmap, label = 'Peak correlation')
        xlim = ax[2].get_xlim()
        ylim = ax[2].get_ylim()
        for cb in spike_dFF_clust_boundaries:
            ax[2].plot([cb + 0.5, cb + 0.5], [ylim[0], ylim[1]], color = 'k', linestyle = '--')
            ax[2].plot([xlim[0], xlim[1]], [cb + 0.5, cb + 0.5], color = 'k', linestyle = '--')

        for axis in [ax[0], ax[1], ax[2]]:
            axis.set_ylabel('Cell #')
            axis.set_xlabel('Cell #')
            if show_cell_labels:
                axis.set_xticks(np.linspace(0, n_cells -1 , n_cells))
                axis.set_yticks(np.linspace(0, n_cells -1, n_cells))

        if show_cell_labels:
            ax[0].set_xticklabels(np.array([cell_ids[cell] + 1 for cell in dFF_order]).astype(int))
            ax[0].set_yticklabels(np.array([cell_ids[cell] + 1 for cell in dFF_order]).astype(int))

            ax[1].set_xticklabels(np.array([cell_ids[cell] + 1 for cell in spike_order]).astype(int))
            ax[1].set_yticklabels(np.array([cell_ids[cell] + 1 for cell in spike_order]).astype(int))

            ax[2].set_xticklabels(np.array([cell_ids[cell] + 1 for cell in spike_dFF_order]).astype(int))
            ax[2].set_yticklabels(np.array([cell_ids[cell] + 1 for cell in spike_dFF_order]).astype(int))

        ax[0].set_title('dFF')
        ax[1].set_title('Spikes')
        ax[2].set_title('Spike - dFF')

        if save_plots:
            if save_path == None:
                save_path = '{0}{1}Plots{1}Pairwise correlations'.format(data_path, sep)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            plt.savefig('{0}{1}Peak_correlation_heatmap_{2}_clusters.png'.format(save_path, sep, method))

    cluster_labels = {'dFF': dFF_cluster_labels, 'spike': spike_cluster_labels, 'spike_dFF': spike_dFF_cluster_labels,
                      'dFF_score': dFF_score, 'spike_score': spike_score, 'spike_dFF_score': spike_dFF_score,
                      'dFF_null_score': dFF_null_scores, 'spike_null_score': spike_null_scores, 'spike_dFF_null_score': spike_dFF_null_scores,
                      'dFF_score_pval': dFF_score_pval, 'spike_score_pval': spike_score_pval, 'spike_dFF_score_pval': spike_dFF_score_pval}

    return cluster_labels

def plot_example_pair(session, cell1, cell2, tvec, frame_rate, dFF, spike_frames, cc_dict, sig_dict, peaks, pval_thresh, cell_ids, n_frames, frame0, plots_path, corr_type = 'spike_dFF', raw_dFF = None, plot_raw_dFF = False, plot_sta_spike_rate = False, dFF_fig_width = 15):

    print('    Cell {0} and Cell {1}'.format(cell_ids[cell1] + 1, cell_ids[cell2] + 1))
    # dF/F and spikes
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = [dFF_fig_width, 4], constrained_layout = True, sharex = True)
    if plot_raw_dFF:
        ax[0].plot(tvec[frame0:frame0 + n_frames], raw_dFF[cell1, frame0:frame0 + n_frames], color = 'k', linewidth = 0.8)
        max_dFF = np.max(raw_dFF[cell1, frame0:frame0 + n_frames])
    else:
        ax[0].plot(tvec[frame0:frame0 + n_frames], dFF[cell1, frame0:frame0 + n_frames], color = 'k', linewidth = 0.8)
        max_dFF = np.max(dFF[cell1, frame0:frame0 + n_frames])
    spikes_show = spike_frames[cell_ids[cell1]]
    spikes_show = spikes_show[spikes_show > frame0]
    spikes_show = spikes_show[spikes_show < frame0 + n_frames].astype(int)
    ax[0].scatter(tvec[spikes_show], np.ones(len(spikes_show))*max_dFF, marker = '.', color = 'b')
    ax[0].set_title('Cell {0}'.format(cell_ids[cell1] + 1))
    ax[0].set_ylabel('- dF/F')
    #ax[0].set_xticks([])

    if plot_raw_dFF:
        ax[1].plot(tvec[frame0:frame0 + n_frames], raw_dFF[cell2, frame0:frame0 + n_frames], color = 'k', linewidth = 0.8)
        max_dFF = np.max(raw_dFF[cell2, frame0:frame0 + n_frames])
    else:
        ax[1].plot(tvec[frame0:frame0 + n_frames], dFF[cell2, frame0:frame0 + n_frames], color = 'k', linewidth = 0.8)
        max_dFF = np.max(dFF[cell2, frame0:frame0 + n_frames])
    spikes_show = spike_frames[cell_ids[cell2]]
    spikes_show = spikes_show[spikes_show > frame0]
    spikes_show = spikes_show[spikes_show < frame0 + n_frames].astype(int)
    ax[1].scatter(tvec[spikes_show], np.ones(len(spikes_show))*max_dFF, marker = '.', color = 'b')
    ax[1].set_title('Cell {0}'.format(cell_ids[cell2] + 1))
    ax[1].set_ylabel('- dF/F')
    ax[1].set_xlabel('Time (s)')

    if plot_raw_dFF:
        plt.savefig('{0}{1}raw dFF Cell {2} Cell {3}'.format(plots_path, sep, cell_ids[cell1] + 1, cell_ids[cell2] + 1))
    else:
        plt.savefig('{0}{1}Cell {2} Cell {3}'.format(plots_path, sep, cell_ids[cell1] + 1, cell_ids[cell2] + 1))

    # Cross correlation
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = [3, 3], constrained_layout = True)
    n_cells = len(cell_ids)
    range_ms = cc_dict['range_frames']*1000/frame_rate
    tvec_cc = np.linspace(-range_ms, range_ms, cc_dict['range_frames']*2)

    pair_no = 0
    flag = 0
    for i in range(n_cells):
        if flag:
            break
        for j in range(i):
            if np.logical_and(i == cell1, j == cell2):
                flag = 1
                break
            pair_no += 1
    pair_no = int(pair_no)
    ax[0].plot(tvec_cc, cc_dict[corr_type][pair_no, :], color = 'k')
    ax[0].set_ylabel('Cross-corr')
    ax[0].set_title('Cell {0} -> Cell {1}'.format(cell_ids[cell1] + 1, cell_ids[cell2] + 1))

    if corr_type == 'spike_dFF':
        pair_no = n_cells*(n_cells - 1)/2 - 1
        flag = 0
        for i in range(n_cells):
            if flag:
                break
            for j in range(i):
                if np.logical_and(i == cell1, j == cell2):
                    flag = 1
                    break
                pair_no += 1
        pair_no = int(pair_no)
    ax[1].plot(tvec_cc, cc_dict[corr_type][pair_no, :], color = 'k')
    ax[1].set_ylabel('Cross-corr')
    ax[1].set_xlabel('Lag (ms)')
    ax[1].set_title('Cell {0} -> Cell {1}'.format(cell_ids[cell2] + 1, cell_ids[cell1] + 1))

    plt.savefig('{0}{1}{4}_corr Cell {2} Cell {3}'.format(plots_path, sep, cell_ids[cell1] + 1, cell_ids[cell2] + 1, corr_type))

    # Spike triggered average
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = [3, 3], constrained_layout = True)
    n_cells = len(cell_ids)
    range_ms = cc_dict['range_frames']*1000/frame_rate
    tvec_cc = np.linspace(-range_ms, range_ms, cc_dict['range_frames']*2)

    pair_no = 0
    flag = 0
    for i in range(n_cells):
        if flag:
            break
        for j in range(i):
            if np.logical_and(i == cell1, j == cell2):
                flag = 1
                break
            pair_no += 1
    pair_no = int(pair_no)
    if plot_raw_dFF:
        ax[0].plot(tvec_cc, cc_dict['sta_raw_dFF_mean'][pair_no, :], color = 'k')
        ax[0].fill_between(tvec_cc, cc_dict['sta_raw_dFF_mean'][pair_no, :] - cc_dict['sta_raw_dFF_sem'][pair_no, :],
                            cc_dict['sta_raw_dFF_mean'][pair_no, :] + cc_dict['sta_raw_dFF_sem'][pair_no, :], color = 'gray', alpha = 0.5)
        ax[0].set_ylabel('- dFF')
    else:
        if plot_sta_spike_rate:
            ax[0].plot(tvec_cc, cc_dict['sta_spike_rate_mean'][pair_no, :], color = 'k')
            ax[0].fill_between(tvec_cc, cc_dict['sta_spike_rate_mean'][pair_no, :] - cc_dict['sta_spike_rate_sem'][pair_no, :],
                                cc_dict['sta_spike_rate_mean'][pair_no, :] + cc_dict['sta_spike_rate_sem'][pair_no, :], color = 'gray', alpha = 0.5)
            ax[0].set_ylabel('Spike rate (Hz)')
        else:
            ax[0].plot(tvec_cc, cc_dict['sta_mean'][pair_no, :], color = 'k')
            ax[0].fill_between(tvec_cc, cc_dict['sta_mean'][pair_no, :] - cc_dict['sta_sem'][pair_no, :],
                                cc_dict['sta_mean'][pair_no, :] + cc_dict['sta_sem'][pair_no, :], color = 'gray', alpha = 0.5)
            ax[0].set_ylabel('- dFF')
    ax[0].set_title('Cell {0} -> Cell {1}'.format(cell_ids[cell1] + 1, cell_ids[cell2] + 1))

    pair_no = n_cells*(n_cells - 1)/2 - 1
    flag = 0
    for i in range(n_cells):
        if flag:
            break
        for j in range(i):
            if np.logical_and(i == cell1, j == cell2):
                flag = 1
                break
            pair_no += 1
    pair_no = int(pair_no)
    if plot_raw_dFF:
        ax[1].plot(tvec_cc, cc_dict['sta_raw_dFF_mean'][pair_no, :], color = 'k')
        ax[1].fill_between(tvec_cc, cc_dict['sta_raw_dFF_mean'][pair_no, :] - cc_dict['sta_raw_dFF_sem'][pair_no, :],
                                cc_dict['sta_raw_dFF_mean'][pair_no, :] + cc_dict['sta_raw_dFF_sem'][pair_no, :], color = 'gray', alpha = 0.5)
        ax[1].set_ylabel('-dFF')
    else:
        if plot_sta_spike_rate:
            ax[1].plot(tvec_cc, cc_dict['sta_spike_rate_mean'][pair_no, :], color = 'k')
            ax[1].fill_between(tvec_cc, cc_dict['sta_spike_rate_mean'][pair_no, :] - cc_dict['sta_spike_rate_sem'][pair_no, :],
                                    cc_dict['sta_spike_rate_mean'][pair_no, :] + cc_dict['sta_spike_rate_sem'][pair_no, :], color = 'gray', alpha = 0.5)
            ax[1].set_ylabel('Spike rate (Hz)')
        else:
            ax[1].plot(tvec_cc, cc_dict['sta_mean'][pair_no, :], color = 'k')
            ax[1].fill_between(tvec_cc, cc_dict['sta_mean'][pair_no, :] - cc_dict['sta_sem'][pair_no, :],
                                    cc_dict['sta_mean'][pair_no, :] + cc_dict['sta_sem'][pair_no, :], color = 'gray', alpha = 0.5)
            ax[1].set_ylabel('-dFF')

    ax[1].set_xlabel('Time from spike (ms)')
    ax[1].set_title('Cell {0} -> Cell {1}'.format(cell_ids[cell2] + 1, cell_ids[cell1] + 1))

    if plot_raw_dFF:
        plt.savefig('{0}{1}STA raw dFF Cell {2} Cell {3}'.format(plots_path, sep, cell_ids[cell1] + 1, cell_ids[cell2] + 1))
    else:
        if plot_sta_spike_rate:
            plt.savefig('{0}{1}STA spike rate Cell {2} Cell {3}'.format(plots_path, sep, cell_ids[cell1] + 1, cell_ids[cell2] + 1))
        else:
            plt.savefig('{0}{1}STA Cell {2} Cell {3}'.format(plots_path, sep, cell_ids[cell1] + 1, cell_ids[cell2] + 1))

def fov_image_cluster_labels(data_path, metadata_file, cells, roi_colors, default_color = 'r', cell_labels = [], scalebar_width_um = 100, scalebar_text = True, label_fontsize = 10, roi_width = 1.5, fig_width = 15, save_fig = False, save_path = None, title = 'seg_image_annotated.png',):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    um_per_px = metadata['um_per_px']
    roi_file = metadata['roi_file']
    with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
        rois = pkl.load(f)

    try:
        try:
            sessions = metadata['sessions_to_process']
        except: session = 1
        im = Image.open('{0}{1}Session{2}_seg_image_registered.tif'.format(data_path, sep, sessions[0]))
    except:
        im = Image.open('{0}{1}seg_image.tif'.format(data_path, sep))

    im_array =np.array(im)
    h = im_array.shape[0]
    w = im_array.shape[1]

    plt.figure(figsize = (fig_width, fig_width*h/w))
    plt.imshow(im_array, cmap = 'Greys_r')

    # For each cell roi, draw mask boundary
    cell_ids = [cell + 1 for cell in cells]
    no_cells = len(cell_ids)

    for cell in range(no_cells):
        cell_id = cell_ids[cell]
        if len(rois.keys()) > 3:
            roi = rois[cell_id]
        else:
            sessions = metadata['sessions_to_process']
            roi = rois[sessions[0]][cell_id]
        if len(cell_labels) == 0:
            cell_label = cell_id
        else:
            cell_label = cell_labels[cell]
        vertices = roi['mask']
        if np.sum(vertices == None) > 0:
            print('Session {0} Cell {1}: mask missing'.format(session, cell_id))
            continue
        if len(roi_colors) > 0:
            roi_color = roi_colors[cell]
        else:
            roi_color = default_color
        plt.plot(vertices[:, 1], vertices[:, 0], color = roi_color, linewidth = roi_width)
        text_x = np.mean(vertices[:, 1]) + w*0.01
        text_y = np.mean(vertices[:, 0])
        plt.text(text_x, text_y, '{0}'.format(int(cell_label)), color = 'w', fontsize = label_fontsize)

    # Plot scalebar
    scalebar_x = w*0.7
    scalebar_y = h*0.8
    text_x = w*0.68
    text_y = h*0.98
    scalebar_width_px = scalebar_width_um/um_per_px
    x = np.linspace(scalebar_x, scalebar_x + scalebar_width_px, 10)
    y = np.ones(10)*scalebar_y
    plt.plot(x, y, linewidth = 10, color = 'w')
    if scalebar_text:
        plt.text(text_x, text_y, '{0} um'.format(scalebar_width_um), color = 'w', fontsize = label_fontsize)

    if save_path == None:
        save_path = data_path

    if save_fig:
        plt.savefig('{0}{1}Seg_image_cluster_labels.png'.format(save_path, sep))

def fov_image_synaptic_connections(data_path, metadata_file, cells, connected, default_color = 'w', cell_labels = [], xlim = [], scalebar_width_um = 100, scalebar_text = True, label_fontsize = 10, roi_width = 0.5, arrowhead_width = 1, fig_width = 15, fig_title = '', save_fig = False, save_path = None, title = 'seg_image_annotated.png',):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    um_per_px = metadata['um_per_px']
    roi_file = metadata['roi_file']
    with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
        rois = pkl.load(f)

    try:
        try:
            sessions = metadata['sessions_to_process']
        except: session = 1
        im = Image.open('{0}{1}Session{2}_seg_image_registered.tif'.format(data_path, sep, sessions[0]))
    except:
        im = Image.open('{0}{1}seg_image.tif'.format(data_path, sep))

    im_array =np.array(im)
    h = im_array.shape[0]
    w = im_array.shape[1]

    plt.figure(figsize = (fig_width, fig_width*h/w))
    plt.imshow(-im_array, cmap = 'Greys')

    # For each cell roi, draw mask boundary
    cell_ids = [cell + 1 for cell in cells]
    no_cells = len(cell_ids)

    for cell in range(no_cells):
        cell_id = cell_ids[cell]
        if len(rois.keys()) > 3:
            roi = rois[cell_id]
        else:
            sessions = metadata['sessions_to_process']
            roi = rois[sessions[0]][cell_id]
        if len(cell_labels) == 0:
            cell_label = cell_id
        else:
            cell_label = cell_labels[cell]
        vertices = roi['mask']
        if np.sum(vertices == None) > 0:
            print('Session {0} Cell {1}: mask missing'.format(session, cell_id))
            continue
        roi_color = default_color
        plt.plot(vertices[:, 1], vertices[:, 0], color = roi_color, linewidth = roi_width)
        text_x = np.mean(vertices[:, 1]) + w*0.01
        text_y = np.mean(vertices[:, 0])
        plt.text(text_x, text_y, '{0}'.format(int(cell_label)), color = 'w', fontsize = label_fontsize)

    # For each pair of connected cells, draw an arrow
    pair_no = 0
    for i in range(no_cells):

        pre_syn_cell = cell_ids[i]
        if len(rois.keys()) > 3:
            roi_pre_syn = rois[pre_syn_cell]
        else:
            sessions = metadata['sessions_to_process']
            roi_pre_syn = rois[sessions[0]][pre_syn_cell]
        x0 = np.mean(roi_pre_syn['mask'][:, 1])
        y0 = np.mean(roi_pre_syn['mask'][:, 0])

        for j in range(i):
            if connected[pair_no]:
                post_syn_cell = cell_ids[j]
                if len(rois.keys()) > 3:
                    roi_post_syn = rois[post_syn_cell]
                else:
                    sessions = metadata['sessions_to_process']
                    roi_post_syn = rois[sessions[0]][post_syn_cell]
                x1 = np.mean(roi_post_syn['mask'][:, 1])
                y1 = np.mean(roi_post_syn['mask'][:, 0])

                plt.arrow(x0, y0, x1 - x0, y1 - y0, color = 'w', linewidth = roi_width, head_width = arrowhead_width)

            pair_no += 1

    for i in range(no_cells):

        post_syn_cell = cell_ids[i]
        if len(rois.keys()) > 3:
            roi_post_syn = rois[post_syn_cell]
        else:
            sessions = metadata['sessions_to_process']
            roi_post_syn = rois[sessions[0]][post_syn_cell]
        x1 = np.mean(roi_post_syn['mask'][:, 1])
        y1 = np.mean(roi_post_syn['mask'][:, 0])

        for j in range(i):

            if connected[pair_no]:
                pre_syn_cell = cell_ids[j]
                if len(rois.keys()) > 3:
                    roi_pre_syn = rois[pre_syn_cell]
                else:
                    sessions = metadata['sessions_to_process']
                    roi_pre_syn = rois[sessions[0]][pre_syn_cell]
                x0 = np.mean(roi_pre_syn['mask'][:, 1])
                y0 = np.mean(roi_pre_syn['mask'][:, 0])

                plt.arrow(x0, y0, x1 - x0, y1 - y0, color = 'w', linewidth = roi_width, head_width = arrowhead_width)

            pair_no += 1


    # Plot scalebar
    scalebar_x = w*0.7
    scalebar_y = h*0.8
    text_x = w*0.68
    text_y = h*0.98
    scalebar_width_px = scalebar_width_um/um_per_px
    x = np.linspace(scalebar_x, scalebar_x + scalebar_width_px, 10)
    y = np.ones(10)*scalebar_y
    plt.plot(x, y, linewidth = 10, color = 'w')
    if scalebar_text:
        plt.text(text_x, text_y, '{0} um'.format(scalebar_width_um), color = 'w', fontsize = label_fontsize)

    if len(xlim) > 0:
        plt.xlim(xlim)

    if len(fig_title) > 0:
        plt.title(fig_title)
    else:
        fig_title = 'Seg_image_cluster_labels'

    if save_path == None:
        save_path = data_path

    if save_fig:
        plt.savefig('{0}{1}{2}.png'.format(save_path, sep, fig_title))

def plot_sta_all_pairs(data_path, metadata_file, cc_dict, peaks_dict, sig_dict, frame_rate, save_path = None, n_pairs = 10, pval_thresh = 0.05, ):

    spike_dFF_peaks = peaks_dict['spike_dFF_peaks']
    spike_dFF_pvals = sig_dict['spike_dFF_pvals']
    range_ms = cc_dict['range_frames']*1000/frame_rate
    tvec_cc = np.linspace(-range_ms, range_ms, cc_dict['range_frames']*2)

    # Pairs with largest +ve correlations
    order = np.argsort(-spike_dFF_peaks)
    order = [pair_no for pair_no in order if spike_dFF_pvals[pair_no] < pval_thresh]
    if len(order) < n_pairs:
        print('     Less than {0} pairs with significant correlations'.format(n_pairs))
        n_pairs = len(order)
    pair_ids = order[:n_pairs]

    fig, ax = plt.subplots(nrows = int(n_pairs/2), ncols = 2, constrained_layout = True, figsize = [6, 8])
    for pair_no in range(n_pairs):
        row = int(pair_no - n_pairs/2)
        col = int(2*pair_no/n_pairs)
        ax_pair = ax[row, col]

        # Plot STA
        ax_pair.plot(tvec_cc, cc_dict['sta_mean'][pair_ids[pair_no], :], color = 'k')
        ax_pair.fill_between(tvec_cc, cc_dict['sta_mean'][pair_ids[pair_no], :] - cc_dict['sta_sem'][pair_ids[pair_no], :],
                                cc_dict['sta_mean'][pair_ids[pair_no], :] + cc_dict['sta_sem'][pair_ids[pair_no], :], color = 'gray', alpha = 0.5)
        #if col == 0:
        #    ax_pair.set_ylabel('STA (- dF/F)')
        if row == int(n_pairs/2) - 1:
            ax_pair.set_xlabel('Lag (ms)')
        #ax_pair.set_title('Cell {0} -> Cell {1}'.format(cell_ids[cell1] + 1, cell_ids[cell2] + 1))

        ax2 = ax_pair.twinx()
        ax2.plot(tvec_cc, cc_dict['spike_dFF'][pair_ids[pair_no], :], color = 'r')
        lag = int(peaks_dict['spike_dFF_lags'][pair_ids[pair_no]])
        peak = peaks_dict['spike_dFF_peaks'][pair_ids[pair_no]]
        ax2.scatter(tvec_cc[lag], peak, color = 'r', edgecolors = 'k')
        #if col == 1:
        #    ax2.set_ylabel('Cross-corr', color = 'r')
        ax2.tick_params(axis= 'y', colors = 'r')
        plt.show()

    if save_path == None:
        save_path = data_path
    plt.savefig('{0}{1}STA_largest_positive_correlations.png'.format(save_path, sep))

    # Pairs with largest -ve correlations
    order = np.argsort(spike_dFF_peaks)
    order = [pair_no for pair_no in order if spike_dFF_pvals[pair_no] < pval_thresh]
    pair_ids = order[:n_pairs]

    fig, ax = plt.subplots(nrows = int(n_pairs/2), ncols = 2, constrained_layout = True, figsize = [6, 8])
    for pair_no in range(n_pairs):
        row = int(pair_no - n_pairs/2)
        col = int(2*pair_no/n_pairs)
        ax_pair = ax[row, col]

        # Plot STA
        ax_pair.plot(tvec_cc, cc_dict['sta_mean'][pair_ids[pair_no], :], color = 'k')
        ax_pair.fill_between(tvec_cc, cc_dict['sta_mean'][pair_ids[pair_no], :] - cc_dict['sta_sem'][pair_ids[pair_no], :],
                                cc_dict['sta_mean'][pair_ids[pair_no], :] + cc_dict['sta_sem'][pair_ids[pair_no], :], color = 'gray', alpha = 0.5)
        #if col == 0:
        #    ax_pair.set_ylabel('STA (- dF/F)')
        if row == int(n_pairs/2) - 1:
            ax_pair.set_xlabel('Lag (ms)')
        #ax_pair.set_title('Cell {0} -> Cell {1}'.format(cell_ids[cell1] + 1, cell_ids[cell2] + 1))

        ax2 = ax_pair.twinx()
        ax2.plot(tvec_cc, cc_dict['spike_dFF'][pair_ids[pair_no], :], color = 'b')
        lag = int(peaks_dict['spike_dFF_lags'][pair_ids[pair_no]])
        peak = peaks_dict['spike_dFF_peaks'][pair_ids[pair_no]]
        ax2.scatter(tvec_cc[lag], peak, color = 'b', edgecolors = 'k')
        #if col == 1:
            #ax2.set_ylabel('Cross-corr', color = 'b')
        ax2.tick_params(axis= 'y', colors = 'b')
        plt.show()

    if save_path == None:
        save_path = data_path
    plt.savefig('{0}{1}STA_largest_negative_correlations.png'.format(save_path, sep))

def plot_sta_all_pairs2(data_path, cc_dict, peak_frames_show, neg_thresh, cells, fig_title = '', save_fig = False, save_path = None):

    sta_mean = cc_dict['sta_mean']
    sta_peak = cc_dict['sta_peak']
    sta_peak_frame = cc_dict['sta_peak_frame']

    pairs_show = [pair for pair in range(len(sta_peak)) if sta_peak_frame[pair] in peak_frames_show]
    connected = np.zeros(len(sta_peak))
    connected[pairs_show] = np.ones(len(pairs_show))
    connected[sta_peak > neg_thresh] = np.zeros(np.sum(sta_peak > neg_thresh))

    n_cells = len(cells)
    fig, ax = plt.subplots(nrows = n_cells, ncols = n_cells, figsize = (18, 8))

    n_pairs = sta_mean.shape[0]
    peak_sta = np.zeros(n_pairs)
    peak_frame = np.zeros(n_pairs)

    pair_no = 0
    for i in range(n_cells):

        ax[i, i].axis('off')

        for j in range(i):

            sta = sta_mean[pair_no, :]
            peak_frame[pair_no] = get_lag_cc(sta)
            peak_sta[pair_no] = sta[int(peak_frame[pair_no])]

            ax[i, j].axis('off')
            ax[i, j].plot(sta, color = 'k', linewidth = 0.8)

            if peak_sta[pair_no] > 0:
                col = 'r'
            else:
                col = 'b'

            if connected[pair_no]:
                ax[i, j].scatter(peak_frame[pair_no], peak_sta[pair_no], color = col, marker = 'o')


            pair_no += 1

    for i in range(n_cells):

        for j in range(i):

            sta = sta_mean[pair_no, :]
            peak_frame[pair_no] = get_lag_cc(sta)
            peak_sta[pair_no] = sta[int(peak_frame[pair_no])]

            ax[j, i].axis('off')
            ax[j, i].plot(sta, color = 'k', linewidth = 0.8)

            if peak_sta[pair_no] > 0:
                col = 'r'
            else:
                col = 'b'

            if connected[pair_no]:
                ax[j, i].scatter(peak_frame[pair_no], peak_sta[pair_no], color = col, marker = 'o')


            pair_no += 1

    assert(pair_no == n_pairs)

    if len(fig_title) > 0:
        title = fig_title
    else:
        title = 'STA all pairs'

    if save_path == None:
        save_path = data_path

    if save_fig:
        plt.savefig('{0}{1}{2}.png'.format(save_path, sep, title))

def plot_sta_connected_pairs(data_path, cc_dict, peak_frames_show, neg_thresh, cells, frame_rate, dFF, spike_frames, range_ms = 15, n_rows = 4, n_cols = 6, plot_correlation = False, fig_title = '', save_fig = False, save_path = None):

    sta_peak = cc_dict['sta_peak']
    sta_peak_frame = cc_dict['sta_peak_frame']

    pairs_show = [pair for pair in range(len(sta_peak)) if sta_peak_frame[pair] in peak_frames_show]
    connected = np.zeros(len(sta_peak))
    connected[pairs_show] = np.ones(len(pairs_show))
    connected[sta_peak > neg_thresh] = np.zeros(np.sum(sta_peak > neg_thresh))

    #range_frames = cc_dict['range_frames']
    #range_ms = cc_dict['range_frames']*1000/frame_rate
    range_frames = int(range_ms*frame_rate/1000)
    tvec = np.linspace(-range_ms, range_ms, range_frames*2)

    n_cells = len(cells)
    n_pairs = np.sum(connected)
    n_pairs_per_page = int(n_rows*n_cols/2)
    n_pages = int(np.ceil(n_pairs/n_pairs_per_page))
    spike_heights = {}
    psp_heights = {}

    fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, figsize = [15, 10])
    page = 0
    row = -1
    col = -1
    pair_no = -1
    for i in range(n_cells):
        for j in range(i):
            pair_no += 1
            if connected[pair_no]:
                col += 1
                if np.mod(col, n_cols) == 0:
                    col = 0
                    row += 1
                    if row > n_rows/2 - 1:
                        page += 1
                        for c in range(n_cols):
                            ax[row*2 - 1, c].set_xlabel('Time from\nspike (ms)')
                        for r in range(n_rows):
                            ax[r, 0].set_ylabel('-dF/F')

                        if len(fig_title) > 0:
                            title = fig_title
                        else:
                            title = 'STA connected pairs'

                        if save_path == None:
                            save_path = data_path

                        if save_fig:
                            plt.savefig('{0}{1}{2}_{3}.png'.format(save_path, sep, title, page))
                        fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, figsize = [15, 10])
                        row = 0

                sta_all = sta(spike_frames[cells[i]], dFF[j], range_frames)
                sta_mean = np.mean(sta_all, axis = 0)

                spikes_all = sta(spike_frames[cells[i]], dFF[i], range_frames)
                spike_mean = np.mean(spikes_all, axis = 0)

                spike_heights[pair_no] = [spikes_all[spike_no, range_frames] for spike_no in range(len(sta_mean))]
                assert(spikes_all[0, range_frames] == np.max(spikes_all[0, :]))
                psp_heights[pair_no] = [sta_all[spike_no, range_frames] for spike_no in range(len(sta_mean))]

                n_spikes = sta_all.shape[0]
                for spike in range(n_spikes):
                    ax[row*2 + 1, col].plot(tvec, sta_all[spike, :], color = 'gray', alpha = 0.2, linewidth = 0.8)
                    ax[row*2, col].plot(tvec, spikes_all[spike, :], color = 'gray', alpha = 0.2, linewidth = 0.8)
                ax[row*2 + 1, col].plot(tvec, sta_mean, color = 'k', linewidth = 1.5, marker = '.')
                ax[row*2, col].plot(tvec, spike_mean, color = 'k', linewidth = 1.5, marker = '.')
                ax[row*2, col].set_xlim([-5, range_ms])
                ax[row*2 + 1, col].set_xlim([-5, range_ms])
                ax[row*2, col].set_title('Cell {0} --> Cell {1}'.format(cells[i] + 1, cells[j] + 1))


    for i in range(n_cells):
        for j in range(i):
            pair_no += 1
            if connected[pair_no]:
                col += 1
                if np.mod(col, n_cols) == 0:
                    col = 0
                    row += 1
                    if row > n_rows/2 - 1:
                        page += 1
                        for c in range(n_cols):
                            ax[row*2 - 1, c].set_xlabel('Time from\nspike (ms)')
                        for r in range(n_rows):
                            ax[r, 0].set_ylabel('-dF/F')
                        if len(fig_title) > 0:
                            title = fig_title
                        else:
                            title = 'STA connected pairs'

                        if save_path == None:
                            save_path = data_path

                        if save_fig:
                            plt.savefig('{0}{1}{2}_{3}.png'.format(save_path, sep, title, page))
                        fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, figsize = [15, 10])
                        row = 0

                sta_all = sta(spike_frames[cells[j]], dFF[i], range_frames)
                sta_mean = np.mean(sta_all, axis = 0)

                spikes_all = sta(spike_frames[cells[j]], dFF[j], range_frames)
                spike_mean = np.mean(spikes_all, axis = 0)

                spike_heights[pair_no] = [spikes_all[spike_no, range_frames] for spike_no in range(len(sta_mean))]
                assert(spikes_all[0, range_frames] == np.max(spikes_all[0, :]))
                psp_heights[pair_no] = [sta_all[spike_no, range_frames] for spike_no in range(len(sta_mean))]

                n_spikes = sta_all.shape[0]
                for spike in range(n_spikes):
                    ax[row*2 + 1, col].plot(tvec, sta_all[spike, :], color = 'gray', alpha = 0.2, linewidth = 0.8)
                    ax[row*2, col].plot(tvec, spikes_all[spike, :], color = 'gray', alpha = 0.2, linewidth = 0.8)
                ax[row*2 + 1, col].plot(tvec, sta_mean, color = 'k', linewidth = 1.5, marker = '.')
                ax[row*2, col].plot(tvec, spike_mean, color = 'k', linewidth = 1.5, marker = '.')
                ax[row*2, col].set_xlim([-5, range_ms])
                ax[row*2 + 1, col].set_xlim([-5, range_ms])
                ax[row*2, col].set_title('Cell {0} --> Cell {1}'.format(cells[j], cells[i]))

    for c in range(n_cols):
        ax[row*2 + 1, c].set_xlabel('Time from\nspike (ms)')
    for r in range(n_rows):
        ax[r, 0].set_ylabel('-dF/F')

    if len(fig_title) > 0:
        title = fig_title
    else:
        title = 'STA connected pairs'

    if save_path == None:
        save_path = data_path

    if save_fig:
        plt.savefig('{0}{1}{2}_{3}.png'.format(save_path, sep, title, page + 1))

    if plot_correlation:
        plot_sta_spike_correlation(spike_heights, psp_heights, save_fig = save_fig, save_path = save_path)

def plot_sta_spike_correlation(spike_heights, psp_heights, save_fig = False, save_path = None):

    n_pairs = len(list(spike_heights.keys()))
    corr = [np.corrcoeff(spike_heights[pair], psp_heights[pair])[0][0] for pair in range(n_pairs)]
    plt.figure(figsize = [5, 4])
    plt.hist(corr, n_bins = int(n_pairs/5), color = 'k')
    plt.xlabel('Correlation between\npre-synaptic spike and\npost-synaptic potential')
    plt.ylabel('Number of pairs')
    if save_fig:
        plt.save('{0}{1}spike_sta_correlation.png'.format(save_path, sep))

def get_spike_rate(spike_vector, spike_bin_frames):

    kernel = np.ones(spike_bin_frames - 1)
    spike_rate = np.convolve(spike_vector, kernel, mode = 'full')
    assert(len(spike_rate) == len(spike_vector) + spike_bin_frames - 2)
    spike_rate = spike_rate[:len(spike_vector) - 1]
    spike_rate = np.insert(spike_rate, 0, 0)
    return spike_rate

def get_shuffled_activity(a, jitter_frames, n_frames, n_bins):

    # 'a' is the activity matrix with dimensions n_cells X n_timepoints
    assert(n_frames == a.shape[1])
    shuffle_bin = permutation(np.arange(0, jitter_frames))
    shuffle_frame_order = np.concatenate([shuffle_bin + bin*jitter_frames for bin in range(n_bins - 1)])
    shuffle_frame_order = np.concatenate([shuffle_frame_order, np.arange(len(shuffle_frame_order), n_frames)])

    return a[:, shuffle_frame_order]

def get_lag_cc(cc):

    med_cc = np.median(cc)
    max_val_idx = np.argmax(cc - med_cc)
    min_val_idx = np.argmin(cc - med_cc)

    pos_peaks = find_peaks(cc)[0]
    neg_peaks = find_peaks(-cc)[0]

    flag = False
    if np.abs(cc[max_val_idx]) >= np.abs(cc[min_val_idx]):
        if max_val_idx in pos_peaks:
            lag = max_val_idx
        else:
            if cc[max_val_idx] > 0:
                lag = max_val_idx
            else:
                flag = True
    else:
        if min_val_idx in neg_peaks:
            lag = min_val_idx
        else:
            if cc[min_val_idx] < 0:
                lag = min_val_idx
            else:
                flag = True

    if flag:
        print('Condition for finding peak correlation not satisfied')
        lag = np.argmax(np.abs(cc - med_cc))

    return lag

def cc(vector1, vector2, range_frames):

    # Return cross-correlation of vector1 and vector2 at lags from -range_frames to range_frames
    cc = correlate(vector1, vector2, mode = 'same')
    assert(len(vector1) == len(vector2))
    mid_frame = int(len(cc)/2)
    first_frame = mid_frame - range_frames
    last_frame = mid_frame + range_frames

    return cc[first_frame:last_frame]/(norm(vector1)*norm(vector2))

def sta(spikes, dFF, range_frames):

    spikes = spikes[spikes > range_frames]
    spikes = spikes[spikes < len(dFF) - range_frames].astype(int)
    sta_all = np.zeros([len(spikes), range_frames*2])
    for i in range(len(spikes)):
        sta_all[i, :] = dFF[spikes[i] - range_frames:spikes[i] + range_frames]

    return sta_all

def connection(sta, peak_spike_dFF_cc, spike_dFF_pval, frame_rate, pval_thresh = 0.05, max_lag_ms = 10, sta_peak_height_threshold = 0.001):

    # Needs to be changed to be based on null distribution of STA peak + peak timing < 10ms after spike
    # Determine if a pair of neurons is synaptically connected
    connect = True

    # Spike - dF/F cc is significant
    if spike_dFF_pval > pval_thresh:
        connect = False

    # STA peak has same sign as spike - dF/F cc peak
    max_lag_frames = int(max_lag_ms*frame_rate/1000)
    middle_frame = int(len(sta)/2)

    f1 = middle_frame
    f2 = middle_frame + max_lag_frames

    max_pos = 0
    max_neg = 0

    pos_peaks = find_peaks(sta[f1:f2])[0]
    if len(pos_peaks) > 0:
        pos_peaks = pos_peaks[pos_peaks > 0]
        pos_peaks = pos_peaks[pos_peaks < f2 - f1]
        max_pos = np.max(sta[f1 + pos_peaks])
        if max_pos < 0:
            max_pos = 0

    neg_peaks = find_peaks(-sta[f1:f2])[0]
    if len(neg_peaks) > 0:
        neg_peaks = neg_peaks[neg_peaks > 0]
        neg_peaks = neg_peaks[neg_peaks < f2 - f1]
        max_neg = np.max(-sta[f1 + neg_peaks])
        if max_neg < 0:
            max_neg = 0

    if peak_spike_dFF_cc < 0:
        if max_neg == 0:
            connect = False

    if peak_spike_dFF_cc > 0:
        if max_pos == 0:
            connect = False

    # STA peak height is above threshold
    if np.max([max_neg, max_pos]) < sta_peak_height_threshold:
        connect = False

    return connect
