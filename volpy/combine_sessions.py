from os.path import sep
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
from PIL import Image
import os
from natsort import natsorted

from subthreshold import burst_subth
from subthreshold import isi_dist

def combine_sessions(data_path, metadata_file, volpy_results,
                        dff_sub_freq = 20, noise_freq = 30,
                        burst_snr_n_spikes = 5, # SNR calculated from bursts with number of spikes <= burst_snr_n_spikes
                        calc_burst_snr = False,
                        overwrite = False, make_plot = False,
                        show_trial_starts = False,
                        dff_scalebar_height = 0.1, scalebar_width = 1):

    try:
        # Load metadata
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
        sessions_to_process = metadata['sessions_to_process']
        batch_data = metadata['batch_data']

        # Load frame times
        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_times_concat = output['frame_and_trial_times']['frame_times_concat']
        trial_start_frames = output['frame_and_trial_times']['trial_start_frames']
        frame_rate = output['frame_and_trial_times']['frame_rate']
        frame_rate = np.mean(list(frame_rate.values()))

        combined_data = volpy_results['combined_data']
        dFF = combined_data['dFF']
        dFF_sub = combined_data['dFF_sub']
        F0 = combined_data['F0']
        cells = combined_data['cells']
        n_cells = len(cells)
        spike_times = combined_data['spike_times']
        spike_frames = combined_data['spike_frames']
        snr = combined_data['snr']
        if calc_burst_snr:
            burst_snr = combined_data['burst_snr']
        tvec = combined_data['tvec']
        trial_start_frames_concat = combined_data['trial_start_frames_concat']
        print('Combined data loaded')
    except:
        print('Combined data could not be loaded')
        overwrite = True

    if overwrite:
        print('Overwriting combined data')

        # Load frame times
        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_times_concat = output['frame_and_trial_times']['frame_times_concat']
        trial_start_frames = output['frame_and_trial_times']['trial_start_frames']
        n_frames_per_trial = output['frame_and_trial_times']['n_frames_per_trial']
        frame_rate = output['frame_and_trial_times']['frame_rate']
        frame_rate = np.mean(list(frame_rate.values()))

        # Load burst data
        if calc_burst_snr:
            isi_data = isi_dist.get_isi_data_session_wise(data_path, metadata_file, volpy_results)
            isi_data = burst_subth.get_bursts_session_wise(data_path, metadata_file, isi_data)
            isi_data = burst_subth.get_burst_dff_session_wise(data_path, metadata_file, isi_data, volpy_results)
            bursts = isi_data['bursts']
            peak_to_trough = isi_data['peak_to_trough']

        # Get QC data
        good_cells = volpy_results['good_cells']
        cells = np.zeros(good_cells[sessions_to_process[0]].shape[0])
        for session in sessions_to_process:
            for batch in range(batch_data[session]['n_batches']):
                cells = cells + 1 - good_cells[session][:, batch]
        cells = np.where(cells == 0)[0] # Cells that are in good cells for all sessions, all batches
        n_cells = len(cells)

        dFF = {cell: [] for cell in cells}
        dFF_sub = {cell: [] for cell in cells}
        F0 = {cell: [] for cell in cells}
        spike_times = {cell: [] for cell in cells}
        spike_frames = {cell: [] for cell in cells}
        total_batches = np.sum([dict['n_batches'] for dict in list(batch_data.values())])
        snr = np.zeros([n_cells, total_batches])
        burst_snr = np.zeros([n_cells, total_batches])
        tvec = frame_times_concat
        trial_start_frames_concat = []
        n_frames_total = 0
        n_batches_total = 0

        trial_tiff_image_path = metadata['trial_tiff_image_path']
        fnames = os.listdir(trial_tiff_image_path[session])
        fnames = natsorted(fnames)
        fnames = ['{0}{1}{2}'.format(trial_tiff_image_path[session], sep, fname) for fname in fnames if fname.endswith('.tif')]

        for session in sessions_to_process:
            print('     Session {0}'.format(session))
            n_batches = batch_data[session]['n_batches']
            n_frames_session = 0

            for batch in range(n_batches):
                print('         Batch {0} of {1}'.format(batch + 1, n_batches))
                estimates = volpy_results[session][batch]['vpy']

                cell_idx = 0
                for cell in cells:
                    spike_frames_batch = estimates['spikes'][cell]
                    spike_times[cell] = np.append(spike_times[cell], frame_times_concat[spike_frames_batch + n_frames_total + n_frames_session])
                    spike_frames[cell] = np.append(spike_frames[cell], spike_frames_batch + n_frames_total + n_frames_session)

                    dFF[cell] = np.append(dFF[cell], estimates['dFF'][cell])
                    F0[cell] = np.append(F0[cell], estimates['F0'][cell])
                    assert(len(estimates['dFF'][cell]) == len(estimates['t_sub'][cell]))
                    dFF_sub[cell] = np.append(dFF_sub[cell], signal_filter(estimates['dFF'][cell], dff_sub_freq, frame_rate, order=5, mode='low'))

                    snr[cell_idx, n_batches_total] = estimates['snr'][cell]
                    cell_idx += 1

                n_batches_total += 1
                n_frames_session += len(estimates['dFF'][0])

                t1 = int(batch_data[session]['first_trials'][batch])
                t2 = int(batch_data[session]['last_trials'][batch])
                sum_trial_frames = np.sum([n_frames_per_trial[session][trial] for trial in range(t1, t2)])

                if not sum_trial_frames == len(estimates['dFF'][0]):
                    print('             {0} frames'.format(len(estimates['dFF'][0])))
                    print('             {0} trials'.format(t2 - t1))
                    print('             {0} frame numbers'.format(sum_trial_frames))
                    for trial in range(t1, t2):
                        print('                 Trial {0}: {1}'.format(trial, fnames[trial]))
                        print('                     {0}'.format(n_frames_per_trial[session][trial]))
                        im = Image.open(fnames[trial])
                        try:
                            print('                     {0} frames in tiff file'.format(im.n_frames))
                        except:
                            print('                     Not able to find number of frames in tiff file')

            trial_start_frames_concat = np.append(trial_start_frames_concat, trial_start_frames[session] + n_frames_total)
            n_frames_total += n_frames_session

        trial_start_frames_concat = trial_start_frames_concat.astype(int)
        if calc_burst_snr:
            cell_idx = 0
            for cell in cells:
                if not isi_data[cell]['bimodal']:
                    burst_snr[cell_idx] = np.nan
                    cell_idx += 1
                else:
                    burst_amp_cell = 0
                    n_bursts = 0
                    burst_lengths = np.sort(list(bursts[cell].keys()))
                    burst_lengths = burst_lengths[burst_lengths <= burst_snr_n_spikes]
                    for length in burst_lengths:
                        for first_spike in bursts[cell][length]:
                            spike_frames_burst = spike_frames[cell][first_spike:first_spike + length].astype(int)
                            peaks = dFF[cell][spike_frames_burst[1:]]
                            troughs = [np.min(dFF[cell][spike_frames_burst[i]:spike_frames_burst[i + 1]]) for i in range(length - 1)]
                            assert(np.mean(peaks - troughs) == peak_to_trough[cell][length][first_spike])
                            burst_amp_cell += peak_to_trough[cell][length][first_spike]
                            n_bursts += 1
                    noise_std = np.std(signal_filter(dFF[cell], noise_freq, frame_rate, order = 5, mode = 'high'))
                    burst_snr[cell_idx] = burst_amp_cell/n_bursts/noise_std
                    cell_idx += 1


        if not len(tvec) == len(dFF[cells[0]]):
            print('frame_times_concat is not the same length as dFF')
            print(len(tvec))
            print(len(dFF[cells[0]]))
        assert(len(tvec) == len(dFF[cells[0]]))

        combined_data = {
            'cells':                     cells,
            'dFF':                       dFF,
            'dFF_sub':                   dFF_sub,
            'F0':                        F0,
            'spike_times':               spike_times,
            'spike_frames':              spike_frames,
            'snr':                       snr,
            'burst_snr':                 burst_snr,
            'tvec':                      tvec,
            'trial_start_frames_concat': trial_start_frames_concat
        }

        volpy_results['combined_data'] = combined_data
        volpy_results_file = metadata['volpy_results_file']
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'wb') as f:
            pkl.dump(volpy_results, f)

    if make_plot:

        # Load metadata
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
        plots_path = metadata['plots_path']

        # Plot dF/F
        n_frames_total = len(tvec)
        plt.figure(figsize = [n_frames_total/5000, n_cells*2])
        levels = np.zeros(n_cells)
        for idx in range(n_cells):
            cell = cells[idx]
            plt.plot(tvec, dFF[cell] + levels[idx], color = 'k', linewidth = 0.8, alpha = 1)
            noise = signal_filter(dFF[cell], noise_freq, frame_rate, order = 5, mode = 'high')
            plt.plot(tvec, noise + levels[idx], color = 'b', linewidth = 0.8, alpha = 0.6 )
            plt.plot(tvec, dFF_sub[cell] + levels[idx], color = 'gray', linewidth = 2)
            plt.scatter(spike_times[cell], levels[idx] + dFF[cell][np.array(spike_frames[cell]).astype(int)], color = 'r', marker = '.')
            if idx < n_cells - 1:
                levels[idx + 1] = levels[idx] + np.max(dFF[cell]) - np.min(dFF[cells[idx + 1]])

        # Scalebar for dF/F
        [x0, x1] = plt.gca().get_xlim()
        dff_bottom = levels[0]
        dff_right = x1 - 10
        dff_left = dff_right - scalebar_width
        dff_scalebar = patches.Rectangle([dff_left, dff_bottom], scalebar_width, dff_scalebar_height, color = 'k',)
        plt.gca().add_patch(dff_scalebar)
        plt.text(dff_right + 1, dff_bottom + dff_scalebar_height/2, '-{0}% \ndF/F'.format(int(dff_scalebar_height*100)))

        # Plot trial starts
        if show_trial_starts:
            [y0, y1] = plt.gca().get_ylim()
            for trial_start_frame in trial_start_frames_concat:
                trial_start_time = frame_times_concat[int(trial_start_frame)]

                plt.plot(np.ones(10)*trial_start_time, np.linspace(y0, y1, 10),
                            color = 'gray', linestyle = '--', linewidth = 0.8)

        plt.xlabel('Time (s)')
        plt.ylabel('Cell # ')
        plt.yticks(ticks = levels, labels = cells + 1)
        plt.savefig('{0}{1}dFF_spikes_combined.png'.format(plots_path, sep))

def signal_filter(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter

    Args:
        sg: 1-d array
            input signal

        freq: float
            cutoff frequency

        order: int
            order of the filter

        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering

    Returns:
        sg: 1-d array
            signal after filtering
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg
