from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltpath
from scipy import signal
import time
import json
from scipy.optimize import curve_fit

from ground_truth import volpy

def main(data_path, sub_ids, cells, movies, cell_folders, ephys_times, make_plots = False, overwrite_ephys_data = False):

    ephys_data = get_ephys_data(data_path, sub_ids, cells, movies, cell_folders, overwrite = overwrite_ephys_data)
    volpy_results = volpy.main(data_path, sub_ids, cells, cell_folders, movies)
    sd_accuracy = spike_detection_accuracy(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, ephys_times)
    if make_plots:
        plot_ephys_and_imaging(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, sd_accuracy)

def plot_ephys_and_imaging(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, sd_accuracy):

    for sid in sub_ids:
        for cell in cells[sid]:
            movie_idx = 1
            for movie in movies[sid][cell]:
                if sd_accuracy[sid][cell][movie]['true_pos'] == None:
                    continue
                snr = volpy_results[sid][cell][movie]['snr']
                title = 'ANM{0} Cell {1} Movie {2} SNR {3} Ephys + dFF'.format(sid, cell, movie_idx, snr)
                fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, constrained_layout = True, figsize = [15, 6])

                # Plot voltage imaging trace and spikes
                frame_times = np.load('{0}{1}{2}{1}{3}{1}frame_times.npy'.format(data_path, sep, cell_folders[sid][cell], movie))
                dff = volpy_results[sid][cell][movie]['dFF']
                ax[0].plot(frame_times, dff[0, :], color = 'k')
                spikes = np.reshape(volpy_results[sid][cell][movie]['spikes'], [-1])
                im_spikes_det = sd_accuracy[sid][cell][movie]['im_spikes_detection']
                ax[0].scatter(frame_times[spikes[np.where(im_spikes_det == -1)[0]]],
                                dff[0, spikes[np.where(im_spikes_det == -1)[0]]],
                                marker = 'o', color = 'r', label = 'False positive')
                ax[0].scatter(frame_times[spikes[np.where(im_spikes_det == 1)[0]]],
                                dff[0, spikes[np.where(im_spikes_det == 1)[0]]],
                                marker = 'o', color = 'b', label = 'True positive')
                ax[0].set_ylabel('dF/F')
                ax[0].legend()

                # Plot ephys trace and spikes
                times = ephys_data[sid]['timings'][cell][movie]
                trace = ephys_data[sid]['traces'][cell][movie]
                spikes = np.array(ephys_data[sid]['spikes'][cell][movie]).astype(int)
                ephys_spikes_det = sd_accuracy[sid][cell][movie]['ephys_spikes_detection']
                ax[1].plot(times, trace, color = 'k')
                ax[1].scatter(times[spikes[np.where(ephys_spikes_det == -1)[0]]],
                                trace[spikes[np.where(ephys_spikes_det == -1)[0]]],
                                marker = 'o', color = 'g', label = 'False negative')
                ax[1].scatter(times[spikes[np.where(ephys_spikes_det == 1)[0]]],
                                trace[spikes[np.where(ephys_spikes_det == 1)[0]]],
                                marker = 'o', color = 'b', label = 'True positive')
                ax[1].set_ylabel('mV')
                ax[1].set_xlabel('Time (s)')
                ax[1].legend()

                fig.suptitle(title)
                fig.savefig('{0}{1}Ephys_and_imaging_plots{1}{2}.png'.format(data_path, sep, title))
                movie_idx += 1

def spike_detection_accuracy(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, ephys_times,
                                peak_jitter_frames = 1, f1_cutoff = 0.7):

    spike_detection_accuracy = {sid: {cell: {} for cell in cells[sid]} for sid in sub_ids}
    snr = []
    tp_all = []
    fn_all = []
    fp_all = []
    for sid in sub_ids:
        for cell in cells[sid]:
            movie_idx = 1
            for movie in movies[sid][cell]:
                spike_detection_accuracy[sid][cell][movie] = {}

                # Get ophys spike times
                im_frame_times = np.load('{0}{1}{2}{1}{3}{1}frame_times.npy'.format(data_path, sep, cell_folders[sid][cell], movie))
                im_spikes_vec = np.zeros(len(im_frame_times))
                im_frame_rate = 1/np.mean(np.diff(im_frame_times))

                im_spikes_all = volpy_results[sid][cell][movie]['spikes']
                im_spike_times = im_frame_times[im_spikes_all]
                im_spike_detection_vec = np.zeros(len(np.reshape(im_spikes_all, [-1])))

                # Get ephys spike times
                ephys_sample_times = ephys_data[sid]['timings'][cell][movie]
                ephys_spikes_vec = np.zeros(len(ephys_sample_times))
                ephys_sampling_rate = 1/min(np.diff(ephys_sample_times)) # Ephys is not necessarily continuous, so take minimum instead of average

                ephys_spikes_all = np.array(ephys_data[sid]['spikes'][cell][movie]).astype(int)
                ephys_spike_times = ephys_sample_times[ephys_spikes_all]
                ephys_spike_detection_vec = np.zeros(len(ephys_spikes_all))

                if len(ephys_spike_times) == 1:
                    spike_detection_accuracy[sid][cell][movie]['true_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_neg'] = None
                    spike_detection_accuracy[sid][cell][movie]['ephys_spikes_detection'] = None
                    spike_detection_accuracy[sid][cell][movie]['im_spikes_detection'] = None
                    continue

                min_isi_ephys = min(np.abs(np.diff(ephys_spike_times)))
                peak_jitter_time = peak_jitter_frames/im_frame_rate

                if min_isi_ephys > 2*peak_jitter_time:

                    print('ANM{0} Cell {1} Movie {2}'.format(sid, cell, movie))

                    n_true_pos = 0
                    n_false_neg = 0

                    im_spikes = im_spikes_all[np.logical_and(im_spike_times > ephys_times[sid][cell][movie_idx][0],
                                                            im_spike_times < ephys_times[sid][cell][movie_idx][1])]
                    first_im_spike = len(np.where(im_spike_times <= ephys_times[sid][cell][movie_idx][0])[0])
                    im_spikes_vec[im_spikes] = np.ones(len(im_spikes))

                    ephys_spikes = ephys_spikes_all[np.logical_and(ephys_spike_times > ephys_times[sid][cell][movie_idx][0],
                                                                ephys_spike_times < ephys_times[sid][cell][movie_idx][1])]
                    ephys_spike_id = sum(ephys_spike_times <= ephys_times[sid][cell][movie_idx][0])
                    ephys_spikes_vec[ephys_spikes] = np.ones(len(ephys_spikes))


                    for spike in ephys_spikes: # Iterate over ephys spikes
                        if np.logical_and(ephys_sample_times[spike] > im_frame_times[0], ephys_sample_times[spike] < im_frame_times[-1]):
                            correct_frame = np.argmin(np.abs(im_frame_times - ephys_sample_times[spike]))
                            possible_frames = list(range(correct_frame - peak_jitter_frames, correct_frame + peak_jitter_frames + 1))
                            spikes_in_possible_frames = np.where(im_spikes_vec[possible_frames])[0]
                            if len(spikes_in_possible_frames) == 0:
                                # False negative
                                n_false_neg += 1
                                ephys_spike_detection_vec[ephys_spike_id] = -1
                            else:
                                # True positive
                                n_true_pos += 1
                                ephys_spike_detection_vec[ephys_spike_id] = 1
                                if len(spikes_in_possible_frames) == 1:
                                    im_spikes_vec[spikes_in_possible_frames[0] + possible_frames[0]] = 0
                                else: #len(spikes_in_possible_frames) > 1
                                    # False positive
                                    im_spikes_vec[int(np.argmin(np.abs(spikes_in_possible_frames - peak_jitter_frames)) + possible_frames[0])] = 0
                        ephys_spike_id += 1

                else:
                    print('ANM{0} Cell {1} Movie {2}:'.format(sid, cell, movie))
                    print('     Min ISI in ephys = {0}s'.format(np.round(min_isi_ephys, decimals = 6)))
                    print('     Peak jitter time = {0}s'.format(np.round(peak_jitter_time, decimals = 6)))
                    spike_detection_accuracy[sid][cell][movie]['true_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_neg'] = None
                    spike_detection_accuracy[sid][cell][movie]['ephys_spikes_detection'] = None
                    spike_detection_accuracy[sid][cell][movie]['im_spikes_detection'] = None
                    continue

                spike_detection_accuracy[sid][cell][movie]['true_pos'] = n_true_pos/len(ephys_spikes)
                tp_all = np.append(tp_all, spike_detection_accuracy[sid][cell][movie]['true_pos'])
                spike_detection_accuracy[sid][cell][movie]['false_pos'] = sum(im_spikes_vec)/len(ephys_spikes)
                fp_all = np.append(fp_all, spike_detection_accuracy[sid][cell][movie]['false_pos'])
                spike_detection_accuracy[sid][cell][movie]['false_neg'] = n_false_neg/len(ephys_spikes)
                fn_all = np.append(fn_all, spike_detection_accuracy[sid][cell][movie]['false_neg'])
                spike_detection_accuracy[sid][cell][movie]['ephys_spikes_detection'] = ephys_spike_detection_vec

                im_spike_detection_vec[first_im_spike + np.where(im_spikes_vec[im_spikes] == 1)[0]] = -1*np.ones(sum(im_spikes_vec[im_spikes] == 1))
                #im_spike_detection_vec[first_im_spike + np.where(im_spikes_vec[im_spikes] == 0)[0]] = np.ones(sum(im_spikes_vec[im_spikes] == 0))
                im_spike_detection_vec[first_im_spike + np.where(im_spikes_vec[im_spikes] == 0)[0]] = np.ones(n_true_pos)
                spike_detection_accuracy[sid][cell][movie]['im_spikes_detection'] = im_spike_detection_vec

                snr = np.append(snr, volpy_results[sid][cell][movie]['snr'])
                movie_idx += 1


    print(tp_all)
    print(fp_all)
    print(fn_all)
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [8, 3], constrained_layout = True)
    ax[0].scatter(snr, tp_all, color = 'k', marker = '.')
    x = np.linspace(min(snr), max(snr), 10)
    popt, pcov = curve_fit(line, snr, tp_all, p0 = [1, 0])
    ax[0].plot(x, line(x, *popt), linestyle = 'dashed', color = 'lime', linewidth = 2)
    ax[0].set_ylabel('Sensitivity')
    ax[0].set_xlabel('SNR')
    ax[1].scatter(snr, np.divide(tp_all, tp_all + fp_all), color = 'k', marker = '.')
    popt, pcov = curve_fit(line, snr, np.divide(tp_all, tp_all + fp_all), p0 = [-1, 0])
    ax[1].plot(x, line(x, *popt) , linestyle = 'dashed', color = 'lime', linewidth = 2)
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('SNR')
    f1_all = np.divide(2*tp_all, 2*tp_all + fp_all + fn_all)
    ax[2].scatter(snr, f1_all, color = 'k', marker = '.')
    popt, pcov = curve_fit(line, snr, f1_all, p0 = [1, 0])
    ax[2].plot(x, line(x, *popt), linestyle = 'dashed', color = 'lime', linewidth = 2)
    snr_cutoff = (f1_cutoff - popt[1])/popt[0]
    print('SNR cutoff: {0}'.format(snr_cutoff))
    x = np.linspace(min(snr), snr_cutoff, 10)
    y = np.linspace(min(f1_all), f1_cutoff, 10)
    ax[2].plot(np.ones(10)*snr_cutoff, y, linestyle = 'dashed', color = 'gray', linewidth = 2)
    ax[2].plot(x, np.ones(10)*f1_cutoff, linestyle = 'dashed', color = 'gray', linewidth = 2)

    ax[2].set_ylabel('F1 score')
    ax[2].set_xlabel('SNR')
    fig.savefig('{0}{1}Spike detection vs SNR.png'.format(data_path, sep))

    return spike_detection_accuracy

def line(x, a, b):
    return a*x + b

def get_ephys_data(data_path, sub_ids, cells, movies, cell_folders, overwrite = False):

    ephys_data = {}
    for sid in sub_ids:
        try:
            with open('{0}{1}ANM{2}_ephys_data.pkl'.format(data_path, sep, sid), 'rb') as f:
                ephys_data[sid] = pkl.load(f)
            print('ANM {0} ephys data loaded'.format(sid))

        except:
            print('ANM {0} ephys data could not be loaded'.format(sid))
            overwrite = True
        if overwrite:
            print('Overwriting ANM {0} ephys data'.format(sid))
            ephys_data[sid] = {cell: {} for cell in cells[sid]}

            # Load ephys traces
            ephys_data[sid]['traces'] = load_ephys_traces(data_path, sid, cells, movies, cell_folders)

            # Get ephys timings
            ephys_data[sid]['timings'] = get_ephys_timings(data_path, sid, cells, movies, cell_folders)

            # Find ephys spikes
            ephys_data[sid]['spikes'] = find_ephys_spikes(data_path, sid, cells, movies, cell_folders)

            with open('{0}{1}ANM{2}_ephys_data.pkl'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(ephys_data[sid], f)

    return ephys_data

def load_ephys_traces(data_path, sid, cells, movies, cell_folders):

    traces = {cell: {movie: [] for movie in movies[sid][cell]} for cell in cells[sid]}
    for cell in cells[sid]:
        for movie in movies[sid][cell]:
            ephys_dir = '{0}{1}{2}{1}{3}{1}ephys'.format(data_path, sep, cell_folders[sid][cell], movie)
            ephys_files = os.listdir(ephys_dir)
            ephys_files = [file for file in ephys_files if file.endswith('npz')]
            for file in ephys_files:
                data = np.load('{0}{1}{2}'.format(ephys_dir, sep, file))
                traces[cell][movie] = np.append(traces[cell][movie], data['voltage'])
    return traces

def get_ephys_timings(data_path, sid, cells, movies, cell_folders):

    timings = {cell: {movie: [] for movie in movies[sid][cell]} for cell in cells[sid]}
    for cell in cells[sid]:
        for movie in movies[sid][cell]:
            ephys_dir = '{0}{1}{2}{1}{3}{1}ephys'.format(data_path, sep, cell_folders[sid][cell], movie)
            ephys_files = os.listdir(ephys_dir)
            ephys_files = [file for file in ephys_files if file.endswith('npz')]
            for file in ephys_files:
                data = np.load('{0}{1}{2}'.format(ephys_dir, sep, file))
                timings[cell][movie] = np.append(timings[cell][movie], data['time'])
    return timings

def find_ephys_spikes(data_path, sid, cells, movies, cell_folders, spike_thresh_mv = -20):

    spikes = {cell: {movie: [] for movie in movies[sid][cell]} for cell in cells[sid]}
    for cell in cells[sid]:
        for movie in movies[sid][cell]:
            ephys_dir = '{0}{1}{2}{1}{3}{1}ephys'.format(data_path, sep, cell_folders[sid][cell], movie)
            ephys_files = os.listdir(ephys_dir)
            ephys_files = [file for file in ephys_files if file.endswith('npz')]
            total_points = 0
            for file in ephys_files:
                data = np.load('{0}{1}{2}'.format(ephys_dir, sep, file))
                trace = data['voltage']
                spikes_sweep =  signal.find_peaks(trace, height = spike_thresh_mv)[0]
                spikes[cell][movie] = np.append(spikes[cell][movie], spikes_sweep + total_points)
                total_points += len(trace)
    return spikes
