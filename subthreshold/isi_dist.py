from os.path import sep
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from volpy import quality_control

def get_isi_data_session_wise(data_path, metadata_file, volpy_results,
                                max_isi_s = 0.025):

    print('Analyzing all ISIs < {0}s'.format(max_isi_s))

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    isi_data = {}
    sessions_to_process = metadata['sessions_to_process']
    batch_data = metadata['batch_data']
    plots_path = metadata['plots_path']
    total_batches = np.sum([dict['n_batches'] for dict in list(batch_data.values())])

    # Load frame times
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_times_concat = output['frame_and_trial_times']['frame_times_concat']

    # Get QC data
    good_cells = volpy_results['good_cells']
    cells = np.zeros(good_cells[sessions_to_process[0]].shape[0])
    for session in sessions_to_process:
        for batch in range(batch_data[session]['n_batches']):
            cells = cells + 1 - good_cells[session][:, batch]
    cells = np.where(cells == 0)[0] # Cells that are in good cells for all sessions, all batches
    n_cells = len(cells)

    n_frames_total = 0
    n_batches_total = 0

    for session in sessions_to_process:
        print('     Session {0}'.format(session))
        n_batches = batch_data[session]['n_batches']
        n_frames_session = 0

        isi_data[session] = {}

        for batch in range(n_batches):
            print('         Batch {0} of {1}'.format(batch + 1, n_batches))
            estimates = volpy_results[session][batch]['vpy']
            isi_data[session][batch] = {}

            # For each good cell, for each spike, get time to nearest spike
            for cell in range(n_cells):

                cell_id = cells[cell]
                spike_frames_batch = estimates['spikes'][cell]
                spike_times_batch = frame_times_concat[spike_frames_batch + n_frames_total + n_frames_session]
                isis_batch = np.diff(spike_times_batch)*1000
                short_isis_batch = isis_batch[isis_batch < max_isi_s*1000]
                [bimodal, thresh_ms] = bimodal_isi(short_isis_batch)
                isi_data[session][batch][cell_id] = {'isis': isis_batch,
                                     'short_isis': short_isis_batch,
                                     'max_isi_s': max_isi_s,
                                     'bimodal': bimodal,
                                     'thresh_ms': thresh_ms
                }

            n_batches_total += 1
            n_frames_session += len(estimates['dFF'][0])

        n_frames_total += n_frames_session


    return isi_data

def get_isi_data(data_path, metadata_file, volpy_results,
                 overwrite = False,
                 make_plot = False,
                 max_isi_s = 0.025, n_bins = 10, log_scale = False,
                 n_rows = 3, figsize = [15, 5]):

    print('Analyzing all ISIs < {0}s'.format(max_isi_s))

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    isi_data_file = metadata['isi_data_file']

    try:
        with open('{0}{1}{2}'.format(data_path, sep, isi_data_file), 'rb') as f:
            isi_data = pkl.load(f)
        print('ISI data loaded')
    except:
        overwrite = True

    if np.logical_or(overwrite, make_plot):
        print('Overwriting ISI data')
        isi_data = {}
        sessions_to_process = metadata['sessions_to_process']
        batch_data = metadata['batch_data']
        plots_path = metadata['plots_path']

        # Get QC data
        good_cells = volpy_results['good_cells']
        cells = np.zeros(good_cells[sessions_to_process[0]].shape[0])
        for session in sessions_to_process:
            for batch in range(batch_data[session]['n_batches']):
                cells = cells + 1 - good_cells[session][:, batch]
        cells = np.where(cells == 0)[0] # Cells that are in good cells for all sessions, all batches
        n_cells = len(cells)

        # Make figure (number of subplots = number of good cells)
        if make_plot:
            n_cols = int(np.ceil(n_cells/n_rows))
            fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = figsize, constrained_layout = True)

        # For each good cell, for each spike, get time to nearest spike
        for cell in range(n_cells):

            cell_id = cells[cell]
            print('Cell {0}'.format(cell_id + 1))
            isis = []
            spike_times = volpy_results['combined_data']['spike_times'][cell_id]
            isis = np.diff(spike_times)*1000

            short_isis = isis[isis < max_isi_s*1000]
            [bimodal, thresh_ms] = bimodal_isi(short_isis)
            isi_data[cell_id] = {'isis': isis,
                                 'short_isis': short_isis,
                                 'max_isi_s': max_isi_s,
                                 'bimodal': bimodal,
                                 'thresh_ms': thresh_ms
            }

            # Plot distribution of nearest-spike ISI
            if make_plot:
                row = int(np.floor(cell/n_cols))
                col = int(np.mod(cell, n_cols))
                if bimodal:
                    hist_color = 'k'
                    spine_color = 'r'
                else:
                    hist_color = 'gray'
                    spine_color = 'k'
                ax[row, col].hist(short_isis, n_bins, color = hist_color)
                if bimodal:
                    ylims = ax[row, col].get_ylim()
                    ax[row, col].plot(np.ones(10)*thresh_ms, np.linspace(ylims[0], ylims[1], 10), color = 'r', linewidth = 0.8, linestyle = '--')
                ax[row, col].set_xlim([0, np.max(short_isis)])
                if log_scale:
                    ax[row, col].set_yscale('log')
                ax[row, col].set_ylabel('# spikes')
                ax[row, col].set_xlabel('ISI (ms)')
                ax[row, col].set_title('Cell {0}'.format(cell_id + 1))
                ax[row, col].spines['bottom'].set_color(spine_color)
                ax[row, col].spines['top'].set_color(spine_color)
                ax[row, col].spines['right'].set_color(spine_color)
                ax[row, col].spines['left'].set_color(spine_color)

        if make_plot:
            plt.savefig('{0}{1}ISI_distributions.png'.format(plots_path, sep))

    return isi_data

def plot_burst_thresh(isi_data, plots_path):

    cells = list(isi_data.keys())
    cells = [cell for cell in cells if not (type(cell) is str)]
    thresh = [isi_data[cell]['thresh_ms'] for cell in cells]
    thresh = [t for t in thresh if not t == None]
    plt.figure(figsize = [3, 2], constrained_layout = True)
    plt.hist(thresh)
    plt.xlabel('ISI threshold for burst (ms)')
    plt.ylabel('Number of cells')
    plt.savefig('{0}{1}Burst_threshold_distribution.png'.format(plots_path, sep))

def bimodal_isi(isis, min_spikes = 50, n_bins = 10, max_thresh_ms = 20, dist_gap_ms = 2):

    bimodal = False
    thresh_ms = None
    if len(isis) < min_spikes:
        print('     Less than {0} spikes'.format(min_spikes))
        return [bimodal, thresh_ms]
    else:
        hist = np.histogram(isis, n_bins)
        n_spikes = hist[0]
        bins = hist[1]
        i = 0
        thresh_ms = bins[i]
        while thresh_ms < max_thresh_ms:
            i += 1
            if i >= len(bins):
                return [bimodal, thresh_ms]
            else:
                thresh_ms = bins[i]

                lower = isis[isis < thresh_ms - dist_gap_ms]
                n_lower = len(lower)/(thresh_ms - dist_gap_ms - bins[0])

                higher = isis[isis > thresh_ms + dist_gap_ms]
                n_higher = len(higher)/(max_thresh_ms - (thresh_ms + dist_gap_ms))

                middle = isis[np.logical_and(isis >= thresh_ms - dist_gap_ms, isis <= thresh_ms + dist_gap_ms)]
                n_middle = len(middle)/(2*dist_gap_ms)

                if np.logical_and(n_lower > n_middle, n_higher > n_middle):
                    bimodal = True
                    return [bimodal, thresh_ms]
        return [bimodal, thresh_ms]






















#def get_burst_isi_thresh(data_path, metadata_file):
