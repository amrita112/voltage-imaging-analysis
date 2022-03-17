from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.signal import correlate

from segmentation import get_roi_arrays

def main(population_data_path, data_paths, metadata_file,
            make_cc_plots = False, make_sig_plots = False,
            overwrite_cc = False, overwrite_sig = False,
            n_iter_bootstrap = 1000,):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        snr_cutoff = qc_results['snr_cutoff']
        cells = qc_results['cells'][float(snr_cutoff)]
        blocks = qc_results['blocks'][float(snr_cutoff)]

    if make_sig_plots:
        make_cc_plots = False
    if overwrite_cc:
        overwrite_sig = True

    for movie in movies:

        print('Movie {0}'.format(movie + 1))
        cc_dict = cross_correlograms_movie(data_paths[movie], metadata_file, list(cells[movie].astype(int)), blocks[movie],
                                    make_plots = make_cc_plots, overwrite = overwrite_cc)
        cc_significance(cc_dict, data_paths[movie], metadata_file, list(cells[movie].astype(int)), blocks[movie],
                                    make_plots = make_sig_plots, overwrite = overwrite_sig, n_iter = n_iter_bootstrap)


def cross_correlograms_movie(data_path, metadata_file, good_cells, good_blocks, range_ms = 50,
                                make_plots = False, n_rows = 4, pairs_per_page = 50, overwrite = False):

    try:
        with open('{0}{1}Cross_correlograms.pkl'.format(data_path, sep), 'rb') as f:
            cc_dict = pkl.load(f)
        dFF_cross_corr = cc_dict['dFF']
        spike_cross_corr = cc_dict['spikes']
        spike_dFF_cross_corr = cc_dict['spike_dFF']
        range_frames = cc_dict['range_frames']
        n_pairs = cc_dict['n_pairs']
        frame_rate = cc_dict['frame_rate']
        print('     Cross correlograms loaded')

    except:
        overwrite = True

    if overwrite:

        print('Overwriting cross correlograms')
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
        sessions_to_process = metadata['sessions_to_process']
        batch_data = metadata['batch_data']
        roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)
        n_cells = roi_arrays[sessions_to_process[0]].shape[0]

        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_rate = output['frame_and_trial_times']['frame_rate'][sessions_to_process[0]]
        range_frames = int(range_ms*frame_rate/1000)


        # Get dF/F and spike trains of all neurons
        volpy_results_file = metadata['volpy_results_file']
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
            volpy_results = pkl.load(f)
        dFF_all = {cell: {} for cell in range(n_cells)}
        spikes_all = {cell: {} for cell in range(n_cells)}

        batch_no = 0

        for session in sessions_to_process:

            n_batches = batch_data[session]['n_batches']
            for batch in range(n_batches):
                estimates = volpy_results[session][batch]['vpy']
                dFF_batch = estimates['dFF']
                spikes_batch = estimates['spikes']

                for cell in range(n_cells):
                    if cell in good_cells:
                        if batch > good_blocks[good_cells.index(cell)]:
                            continue
                        else:
                            dFF_all[cell][batch_no] = dFF_batch[cell]
                            spikes_all[cell][batch_no] = np.zeros(len(dFF_batch[cell]))
                            spikes_all[cell][batch_no][spikes_batch[cell]] = np.ones(len(spikes_batch[cell]))

                batch_no += 1
        total_batches = batch_no

        # Get cross correlograms
        n_good_cells = len(good_cells)
        n_pairs = int(n_good_cells*(n_good_cells - 1)/2)
        dFF_cross_corr = cross_correlograms(dFF_all, dFF_all, good_cells, range_frames)
        spike_cross_corr = cross_correlograms(spikes_all, spikes_all, good_cells, range_frames)
        spike_dFF_cross_corr = cross_correlograms(spikes_all, dFF_all, good_cells, range_frames)

        cc_dict = {'dFF': dFF_cross_corr, 'spikes': spike_cross_corr, 'spike_dFF': spike_dFF_cross_corr,
                    'dFF_all': dFF_all, 'spikes_all': spikes_all,
                    'range_frames': range_frames, 'frame_rate': frame_rate, 'n_pairs': n_pairs}
        with open('{0}{1}Cross_correlograms.pkl'.format(data_path, sep), 'wb') as f:
            pkl.dump(cc_dict, f)

    if make_plots:

        tvec_cc = np.array(list(range(-range_frames, range_frames)))
        tvec_cc = (tvec_cc*1000/frame_rate).astype(int)

        n_figs = int(np.ceil(n_pairs/pairs_per_page))
        print('     Making {0} figures'.format(n_figs))
        n_cols = int(np.ceil(pairs_per_page/n_rows))

        for fig_no in range(n_figs):

            print('         Fig {0}'.format(fig_no + 1))
            fig_dFF, ax_dFF = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True,
                                    sharey = True,
                                    figsize = [30, 15])
            fig_spike, ax_spike = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True,
                                        sharey = True,
                                        figsize = [30, 15])
            fig_spike_dFF, ax_spike_dFF = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True,
                                                sharey = True,
                                                figsize = [30, 15])

            first_pair = fig_no*pairs_per_page
            last_pair = np.min([(fig_no + 1)*pairs_per_page, n_pairs])
            for pair in range(first_pair, last_pair):

                row = int(np.floor((pair - first_pair)/n_cols))
                col = int(np.mod((pair - first_pair), n_cols))

                ax_dFF[row, col].plot(tvec_cc, dFF_cross_corr[pair, :], color = 'k', linewidth = 2)
                ax_spike[row, col].plot(tvec_cc, spike_cross_corr[pair, :], color = 'k', linewidth = 2)
                ax_spike_dFF[row, col].plot(tvec_cc, spike_dFF_cross_corr[pair, :], color = 'k', linewidth = 2)

                for ax_pair in (ax_dFF[row, col], ax_spike[row, col], ax_spike_dFF[row, col]):
                    ylims = ax_pair.get_ylim()
                    #ax_pair.plot(np.zeros(10), np.linspace(ylims[0], ylims[1], 10), color = 'k', linewidth = 0.5, linestyle = '--')
                    ax_pair.plot(tvec_cc, np.zeros(range_frames*2), color = 'k', linestyle = '--', linewidth = 0.5)
                    ax_pair.set_xlabel('Time (ms)')
                    ax_pair.set_ylabel('Cross corr')

            fig_dFF.savefig('{0}{1}Plots{1}dFF_cross_correlation{2}.png'.format(data_path, sep, fig_no + 1))
            fig_spike.savefig('{0}{1}Plots{1}Spike_cross_correlation{2}.png'.format(data_path, sep, fig_no + 1))
            fig_spike_dFF.savefig('{0}{1}Plots{1}Spike_dFF_cross_correlation{2}.png'.format(data_path, sep, fig_no + 1))

    return cc_dict

def cc_significance(cc_dict, data_path, metadata_file, good_cells, good_blocks,
                        range_ms = 50, n_iter = 1000, spike_jitter_ms = 5, dFF_jitter_ms = 10, sig_thresh = 0.05,
                        make_plots = False, n_rows = 4, pairs_per_page = 50, overwrite = False):

    try:
        with open('{0}{1}Cross_correlogram_significance.pkl'.format(data_path, sep), 'rb') as f:
            sig_dict = pkl.load(f)
        dFF_pos = sig_dict['dFF_pos']
        spike_pos = sig_dict['spike_pos']
        spike_dFF_pos = sig_dict['spike_dFF_pos']
        dFF_neg = sig_dict['dFF_neg']
        spike_neg = sig_dict['spike_neg']
        spike_dFF_neg = sig_dict['spike_dFF_neg']
        print('     Cross correlogram significance loaded')

    except:
        overwrite = True

    dFF_cross_corr = cc_dict['dFF']
    spike_cross_corr = cc_dict['spikes']
    spike_dFF_cross_corr = cc_dict['spike_dFF']
    dFF_all = cc_dict['dFF_all']
    spikes_all = cc_dict['spikes_all']
    range_frames = cc_dict['range_frames']
    n_pairs = cc_dict['n_pairs']
    frame_rate = cc_dict['frame_rate']

    if overwrite:

        print('     Bootstrapping to find significant cross-correlogram peaks')
        sig_dict = {}

        # Find pairs with significant peak in cross-correlation
        dFF_pos_peaks = np.max(dFF_cross_corr, axis = 1)
        spike_pos_peaks = np.max(spike_cross_corr, axis = 1)
        spike_dFF_pos_peaks = np.max(spike_dFF_cross_corr, axis = 1)

        dFF_neg_peaks = np.min(dFF_cross_corr, axis = 1)
        spike_neg_peaks = np.min(spike_cross_corr, axis = 1)
        spike_dFF_neg_peaks = np.min(spike_dFF_cross_corr, axis = 1)

        dFF_jitter_pos_peaks = np.zeros([n_pairs, n_iter])
        spike_jitter_pos_peaks = np.zeros([n_pairs, n_iter])
        spike_dFF_jitter_pos_peaks = np.zeros([n_pairs, n_iter])

        dFF_jitter_neg_peaks = np.zeros([n_pairs, n_iter])
        spike_jitter_neg_peaks = np.zeros([n_pairs, n_iter])
        spike_dFF_jitter_neg_peaks = np.zeros([n_pairs, n_iter])

        dFF_jitter_frames = int(dFF_jitter_ms*frame_rate/1000)
        spike_jitter_frames = int(spike_jitter_ms*frame_rate/1000)

        for iter in tqdm(range(n_iter)):


            dFF_all_jitter = jitter_dFF(dFF_all, dFF_jitter_frames)
            spikes_all_jitter = jitter_spikes(spikes_all, spike_jitter_frames)

            dFF_cross_corr_jitter = cross_correlograms(dFF_all_jitter, dFF_all, good_cells, range_frames)
            spike_cross_corr_jitter = cross_correlograms(spikes_all_jitter, spikes_all, good_cells, range_frames)
            spike_dFF_cross_corr_jitter = cross_correlograms(spikes_all_jitter, dFF_all, good_cells, range_frames)

            dFF_jitter_pos_peaks[:, iter] = np.max(dFF_cross_corr_jitter, axis = 1)
            spike_jitter_pos_peaks[:, iter] = np.max(spike_cross_corr_jitter, axis = 1)
            spike_dFF_jitter_pos_peaks[:, iter] = np.max(spike_dFF_cross_corr_jitter, axis = 1)

            dFF_jitter_neg_peaks[:, iter] = np.min(dFF_cross_corr_jitter, axis = 1)
            spike_jitter_neg_peaks[:, iter] = np.min(spike_cross_corr_jitter, axis = 1)
            spike_dFF_jitter_neg_peaks[:, iter] = np.min(spike_dFF_cross_corr_jitter, axis = 1)

        # Find number of iterations with larger peaks than experimentally observed
        dFF_pos = (np.sum((np.reshape(dFF_pos_peaks, [n_pairs, 1]) - dFF_jitter_pos_peaks) <= 0, axis = 1)/n_iter) < sig_thresh
        spike_pos = (np.sum((np.reshape(spike_pos_peaks, [n_pairs, 1]) - spike_jitter_pos_peaks) <= 0, axis = 1)/n_iter) < sig_thresh
        spike_dFF_pos = (np.sum((np.reshape(spike_dFF_pos_peaks, [n_pairs, 1]) - spike_dFF_jitter_pos_peaks) <= 0, axis = 1)/n_iter) < sig_thresh

        dFF_neg = (np.sum((np.reshape(dFF_neg_peaks, [n_pairs, 1]) - dFF_jitter_neg_peaks) >= 0, axis = 1)/n_iter) < sig_thresh
        spike_neg = (np.sum((np.reshape(spike_neg_peaks, [n_pairs, 1]) - spike_jitter_neg_peaks) >= 0, axis = 1)/n_iter) < sig_thresh
        spike_dFF_neg = (np.sum((np.reshape(spike_dFF_neg_peaks, [n_pairs, 1]) - spike_dFF_jitter_neg_peaks) >= 0, axis = 1)/n_iter) < sig_thresh

        sig_dict['dFF_pos'] = dFF_pos
        sig_dict['spike_pos'] = spike_pos
        sig_dict['spike_dFF_pos'] = spike_dFF_pos
        sig_dict['dFF_neg'] = dFF_neg
        sig_dict['spike_neg'] = spike_neg
        sig_dict['spike_dFF_neg'] = spike_dFF_neg
        with open('{0}{1}Cross_correlogram_significance.pkl'.format(data_path, sep), 'wb') as f:
            pkl.dump(sig_dict, f)

    if make_plots:

        tvec_cc = np.array(list(range(-range_frames, range_frames)))
        tvec_cc = (tvec_cc*1000/frame_rate).astype(int)

        n_figs = int(np.ceil(n_pairs/pairs_per_page))
        print('     Making {0} figures'.format(n_figs))
        n_cols = int(np.ceil(pairs_per_page/n_rows))

        for fig_no in range(n_figs):

            print('         Fig {0}'.format(fig_no + 1))
            fig_dFF, ax_dFF = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, sharey = True, figsize = [30, 15])
            fig_spike, ax_spike = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, sharey = True, figsize = [30, 15])
            fig_spike_dFF, ax_spike_dFF = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, sharey = True, figsize = [30, 15])

            first_pair = fig_no*pairs_per_page
            last_pair = np.min([(fig_no + 1)*pairs_per_page, n_pairs])
            for pair in range(first_pair, last_pair):

                row = int(np.floor((pair - first_pair)/n_cols))
                col = int(np.mod((pair - first_pair), n_cols))

                color = 'k'
                if dFF_pos[pair]:
                    color = 'r'
                    if dFF_neg[pair]:
                        color = 'm'
                else:
                    if dFF_neg[pair]:
                        color = 'b'
                ax_dFF[row, col].plot(tvec_cc, dFF_cross_corr[pair, :], color = color, linewidth = 2)
                color = 'k'
                if spike_pos[pair]:
                    color = 'r'
                    if spike_neg[pair]:
                        color = 'm'
                else:
                    if spike_neg[pair]:
                        color = 'b'
                ax_spike[row, col].plot(tvec_cc, spike_cross_corr[pair, :], color = color, linewidth = 2)
                color = 'k'
                if spike_dFF_pos[pair]:
                    color = 'r'
                    if spike_dFF_neg[pair]:
                        color = 'm'
                else:
                    if spike_dFF_neg[pair]:
                        color = 'b'
                ax_spike_dFF[row, col].plot(tvec_cc, spike_dFF_cross_corr[pair, :], color = color, linewidth = 2)

                for ax_pair in (ax_dFF[row, col], ax_spike[row, col], ax_spike_dFF[row, col]):
                    ylims = ax_pair.get_ylim()
                    #ax_pair.plot(np.zeros(10), np.linspace(ylims[0], ylims[1], 10), color = 'k', linewidth = 0.5, linestyle = '--')
                    ax_pair.plot(tvec_cc, np.zeros(range_frames*2), color = 'k', linestyle = '--', linewidth = 0.5)
                    ax_pair.set_xlabel('Time (ms)')
                    ax_pair.set_ylabel('Cross corr')

            fig_dFF.savefig('{0}{1}Plots{1}dFF_cross_correlation{2}.png'.format(data_path, sep, fig_no + 1))
            fig_spike.savefig('{0}{1}Plots{1}Spike_cross_correlation{2}.png'.format(data_path, sep, fig_no + 1))
            fig_spike_dFF.savefig('{0}{1}Plots{1}Spike_dFF_cross_correlation{2}.png'.format(data_path, sep, fig_no + 1))


def cross_correlograms(vectors1, vectors2, good_cells, range_frames):

    n_good_cells = len(good_cells)
    n_pairs = int(n_good_cells*(n_good_cells - 1)/2)
    cross_corr = np.zeros([n_pairs, range_frames*2])
    pair_no = -1

    for i in range(n_good_cells):

        cell1 = good_cells[i]
        cell1_batches = list(vectors1[cell1].keys())

        for j in range(i):

            pair_no += 1

            cell2 = good_cells[j]
            cell2_batches = list(vectors2[cell2].keys())
            overlap = get_overlap(cell1_batches, cell2_batches)

            vector_cell1 = np.concatenate([vectors1[cell1][batch] for batch in overlap])
            vector_cell2 = np.concatenate([vectors2[cell2][batch] for batch in overlap])
            cross_corr[pair_no, :] = cc(vector_cell1, vector_cell2, range_frames)

    return cross_corr

def cc(vector1, vector2, range_frames):

    full_correlation = correlate(vector1, vector2, mode = 'same')
    len_full = len(full_correlation)

    mid = int(len_full/2)
    indices_keep = np.array(list(range(mid - range_frames, mid + range_frames)))
    indices_keep = indices_keep.astype(int)

    return full_correlation[indices_keep]

def get_overlap(list1, list2):

    # Find maximum overlapping continuous string of digits between two lists of digits
    assert(len(list1) > 0)
    assert(len(list2) > 0)

    dif1 = np.diff(list1)
    dif2 = np.diff(list2)

    dif1 = np.append(dif1, 2)
    dif2 = np.append(dif2, 2)

    max_cont1 = np.where(dif1 > 1)[0][0]
    max_cont2 = np.where(dif2 > 1)[0][0]

    max_cont = np.min([max_cont1, max_cont2])
    overlap = list(range(0, max_cont + 1))

    return overlap

def jitter_dFF(dFF_all, jitter_frames):

    cells = list(dFF_all.keys())
    jittered_dFF = {cell: {} for cell in cells}
    for cell in cells:

        blocks = list(dFF_all[cell].keys())
        for block in blocks:

            n_frames = len(dFF_all[cell][block])
            n_sub_blocks = int(np.floor(n_frames/jitter_frames)) - 1
            sub_block_starts = np.linspace(0, n_sub_blocks*jitter_frames, n_sub_blocks)
            sub_block_starts = np.random.permutation(sub_block_starts).astype(int)
            jittered_frame_order = np.concatenate([list(range(sub_block_start, sub_block_start + jitter_frames)) for sub_block_start in sub_block_starts])
            jittered_frame_order = np.append(jittered_frame_order, list(range(n_sub_blocks*jitter_frames, n_frames)))
            #assert(len(jittered_frame_order) == n_frames)
            jittered_dFF[cell][block] = dFF_all[cell][block][jittered_frame_order]

    return jittered_dFF

def jitter_spikes(spikes_all, jitter_frames):

    cells = list(spikes_all.keys())
    jittered_spikes = {cell: {} for cell in cells}
    for cell in cells:

        blocks = list(spikes_all[cell].keys())
        for block in blocks:

            n_frames = len(spikes_all[cell][block])
            spike_frames = np.where(spikes_all[cell][block])[0]
            jittered_spike_frames = spike_frames + np.random.randint(-jitter_frames, jitter_frames + 1, size = len(spike_frames))

            # Remove jittered spike times < 0 or > n_frames
            early = np.where(jittered_spike_frames < 0)[0]
            late = np.where(jittered_spike_frames >= n_frames)[0]
            jittered_spike_frames[early] = np.zeros(len(early))
            jittered_spike_frames[late] = np.ones(len(late))*(n_frames - 1)

            jittered_spikes[cell][block] = np.zeros(len(spikes_all[cell][block]))
            jittered_spikes[cell][block][jittered_spike_frames] = np.ones(len(jittered_spike_frames))

    return jittered_spikes
