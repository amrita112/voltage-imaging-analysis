from os.path import sep
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def get_bursts(data_path, metadata_file, isi_data, overwrite = False, verbose = False):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    try:
        bursts = isi_data['bursts']
        first_spikes = isi_data['first_spikes']
        print('Loaded bursts info')
    except KeyError:
        overwrite = True
        bursts = {}
        first_spikes = {}

    if overwrite:
        print('Overwriting bursts info')
        cells = list(isi_data.keys())
        cells = [cell for cell in cells if not (type(cell) is str)]
        n_cells = len(cells)
        for cell_n in range(n_cells):
            cell = cells[cell_n]
            print('     Cell {0}'.format(cell + 1))
            bimodal = isi_data[cell]['bimodal']
            if not bimodal:
                bursts[cell] = None
                first_spikes[cell] = None
                continue
            else:

                isis = isi_data[cell]['isis']
                thresh_ms = isi_data[cell]['thresh_ms']

                bursts[cell] = {}
                first_spikes[cell] = []
                spike_in_burst = isis < thresh_ms
                print('         {0} out of {1} spikes are in bursts'.format(np.sum(spike_in_burst), len(isis)))
                i = 0
                n_spikes_burst = 1
                first_spike_burst = i
                flag = 0
                for i in range(len(isis)):
                    if spike_in_burst[i]:
                        n_spikes_burst += 1
                        if flag == 0:
                            flag = 1
                            first_spike_burst = i
                    else:
                        if flag:
                            first_spikes[cell] = np.append(first_spikes[cell], first_spike_burst)
                            try:
                                bursts[cell][n_spikes_burst] = np.append(bursts[cell][n_spikes_burst], first_spike_burst)
                                spike_flags = spike_in_burst[first_spike_burst:first_spike_burst + n_spikes_burst - 1]
                                assert(np.linalg.norm(spike_flags - np.ones(n_spikes_burst - 1)) == 0)
                            except KeyError:
                                bursts[cell][n_spikes_burst] = [first_spike_burst]
                            n_spikes_burst = 1
                            flag = 0

                if verbose:
                    for n_spikes_burst in np.sort(list(bursts[cell].keys())):
                        print('     {0} bursts with {1} spikes'.format(len(bursts[cell][n_spikes_burst]), n_spikes_burst))

        isi_data['bursts'] = bursts
        isi_data['first_spikes'] = first_spikes

    return isi_data


def get_burst_dff(data_path, metadata_file, isi_data,
                    frames_before_first_spike = 10, frames_after_last_spike = 10,
                    pre_burst_dff_avg_frames = 2,
                    make_plots = False, n_rows = 3,
                    n_bursts_plot = 10,
                    verbose = False, overwrite = False):

    try:
        burst_start_frames = isi_data['burst_start_frames']
        burst_end_frames = isi_data['burst_end_frames']
        burst_dff = isi_data['burst_dff']
        peak_to_trough = isi_data['peak_to_trough']
        burst_dff_avg = isi_data['burst_dff_avg']
        print('Loaded burst dFF')
    except KeyError:
        overwrite = True

    if overwrite or make_plots:
        print('Overwriting burst dFF')

        # Load metadata
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)

        if make_plots:
            plots_path = metadata['plots_path']
            if not os.path.isdir('{0}{1}Burst dFF'.format(plots_path, sep)):
                os.mkdir('{0}{1}Burst dFF'.format(plots_path, sep))

        # Load frame rate
        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_rate = output['frame_and_trial_times']['frame_rate']
        frame_rate = np.mean(list(frame_rate.values()))

        # Get burst data
        bursts = isi_data['bursts']
        cells = list(isi_data.keys())
        cells = [cell for cell in cells if not (type(cell) is str)]

        # Load volpy results + QC data
        volpy_results_file = metadata['volpy_results_file']
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
            volpy_results = pkl.load(f)

        dFF = volpy_results['combined_data']['dFF']
        dFF_sub = volpy_results['combined_data']['dFF_sub']
        spike_frames = volpy_results['combined_data']['spike_frames']
        spike_times = volpy_results['combined_data']['spike_times']
        snr = volpy_results['combined_data']['snr']
        burst_start_frames = {cell: {} for cell in cells}
        burst_end_frames = {cell: {} for cell in cells}
        burst_dff = {cell: {} for cell in cells}
        burst_dff_sub = {cell: {} for cell in cells}
        burst_dff_avg = {cell: {} for cell in cells}
        peak_to_trough = {cell: {} for cell in cells}
        n_frames_total = 0
        cell_idx = -1

        for cell in cells:

            cell_idx += 1
            if not isi_data[cell]['bimodal']:
                continue
            else:
                burst_lengths = np.sort(list(bursts[cell].keys()))
                n_burst_lengths = len(burst_lengths)
                if make_plots:
                    n_cols = int(np.ceil(n_burst_lengths/n_rows))
                    fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, figsize = [n_cols*4, n_rows*4])
                print('    Cell {0}'.format(cell + 1))

                for idx in range(n_burst_lengths):
                    n_spikes_burst = burst_lengths[idx]
                    burst_dff[cell][n_spikes_burst] = {}
                    burst_dff_sub[cell][n_spikes_burst] = {}
                    burst_dff_avg[cell][n_spikes_burst] = {}
                    peak_to_trough[cell][n_spikes_burst] = {}
                    start = []
                    end = []
                    if verbose:
                        print('        First spikes in bursts with {0} spikes: {1}'.format(n_spikes_burst, bursts[cell][n_spikes_burst]))
                    for first_spike in bursts[cell][n_spikes_burst]:
                        last_spike = first_spike + n_spikes_burst - 1
                        burst_spike_times = spike_times[cell][first_spike:last_spike + 1]
                        assert(np.max(np.diff(burst_spike_times))*1000 < isi_data[cell]['thresh_ms'])
                        if spike_frames[cell][first_spike] > frames_before_first_spike:
                            dff_start = int(spike_frames[cell][first_spike] - frames_before_first_spike)
                        else:
                            dff_start = 0
                        dff_avg_start = int(spike_frames[cell][first_spike] - pre_burst_dff_avg_frames)
                        if dff_avg_start < 0:
                            dff_avg_start = 0
                        start = np.append(start, dff_start)
                        dff_end = int(spike_frames[cell][last_spike] + frames_after_last_spike)
                        end = np.append(end, dff_end)
                        burst_dff[cell][n_spikes_burst][first_spike] = dFF[cell][dff_start:dff_end]
                        burst_dff_sub[cell][n_spikes_burst][first_spike] = dFF_sub[cell][dff_start:dff_end]
                        burst_dff_avg[cell][n_spikes_burst][first_spike] = np.mean(dFF_sub[cell][dff_avg_start:int(spike_frames[cell][first_spike])])

                        spike_frames_burst = spike_frames[cell][first_spike:first_spike + n_spikes_burst].astype(int)
                        peaks = dFF[cell][spike_frames_burst[1:]]
                        troughs = [np.min(dFF[cell][spike_frames_burst[i]:spike_frames_burst[i + 1]]) for i in range(n_spikes_burst - 1)]
                        peak_to_trough[cell][n_spikes_burst][first_spike] = np.mean(peaks - troughs)

                    burst_start_frames[cell][n_spikes_burst] = start
                    burst_end_frames[cell][n_spikes_burst] = end

                    if make_plots:
                        row = int(np.floor(idx/n_cols))
                        col = int(np.mod(idx, n_cols))
                        if n_cols > 1:
                            ax_plot = ax[row, col]
                        else:
                            ax_plot = ax[row]
                        n_bursts = len(start)
                        for first_spike in bursts[cell][n_spikes_burst][:n_bursts_plot]:
                            if verbose:
                                print('         {0} spikes in burst'.format(n_spikes_burst))
                            signal = burst_dff[cell][n_spikes_burst][first_spike]
                            signal_sub = burst_dff_sub[cell][n_spikes_burst][first_spike]

                            last_spike = first_spike + n_spikes_burst - 1
                            burst_spike_frames = np.array(spike_frames[cell][first_spike:last_spike + 1]).astype(int)

                            if spike_frames[cell][first_spike] > frames_before_first_spike:
                                burst_spike_frames = burst_spike_frames - spike_frames[cell][first_spike] + frames_before_first_spike
                            else:
                                burst_spike_frames = burst_spike_frames - spike_frames[cell][first_spike]
                            burst_spike_frames = burst_spike_frames.astype(int)

                            burst_spike_times = spike_times[cell][first_spike:last_spike + 1]
                            burst_spike_times = (burst_spike_times - burst_spike_times[0])*1000

                            tvec = np.linspace(0, len(signal)/frame_rate, len(signal))
                            if spike_frames[cell][first_spike] > frames_before_first_spike:
                                tvec = tvec - tvec[frames_before_first_spike]
                            else:
                                tvec = tvec - tvec[int(spike_frames[cell][first_spike])]
                            tvec = tvec*1000

                            trough_frames = [burst_spike_frames[i] + np.argmin(signal[burst_spike_frames[i]:burst_spike_frames[i + 1]]) for i in range(n_spikes_burst - 1)]
                            troughs = signal[trough_frames]

                            ax_plot.plot(tvec, signal, linewidth = 0.8, color = 'k')
                            ax_plot.plot(tvec, signal_sub, linewidth = 2, color = 'gray', alpha = 0.5)
                            ax_plot.scatter(burst_spike_times, signal[burst_spike_frames], marker = '.', color = 'r')
                            ax_plot.scatter(tvec[trough_frames], troughs, marker = 'o', color = 'b', alpha = 0.6)

                        [y0, y1] = ax_plot.get_ylim()
                        rect_left = -pre_burst_dff_avg_frames*1000/frame_rate
                        highlight = patches.Rectangle([rect_left, y0], -rect_left, y1 - y0, color = 'lightcoral', alpha = 0.2)
                        ax_plot.add_patch(highlight)
                        ax_plot.set_xlabel('Time from first spike in burst (ms)')
                        ax_plot.set_ylabel('dF/F')
                        ax_plot.set_title('{0} spikes'.format(n_spikes_burst))

            if make_plots:
                fig.suptitle('Cell {0} SNR {1}.png'.format(cell, np.round(np.mean(snr[cell_idx, :]), decimals = 1)))
                fig.savefig('{0}{1}Burst dFF{1}Cell {2} SNR {3}_{4} frames.png'.format(plots_path, sep, cell, np.round(np.mean(snr[cell_idx, :]), decimals = 1),
                                                                                        pre_burst_dff_avg_frames))
                plt.close()

    isi_data['burst_start_frames'] = burst_start_frames
    isi_data['burst_end_frames'] = burst_end_frames
    isi_data['burst_dff'] = burst_dff
    isi_data['peak_to_trough'] = peak_to_trough
    isi_data['burst_dff_avg'] = burst_dff_avg

    return isi_data
