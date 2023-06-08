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
from tqdm import tqdm
from natsort import natsorted
from scipy.signal import find_peaks

from ground_truth import volpy
from ground_truth import correlation_spectrum

def main(data_path, sub_ids, cells, movies, cell_folders, ephys_times, make_plots = False, overwrite_ephys_data = False, plot_corr_spctrm = False, overwrite_csd_data = False):

    ephys_data = get_ephys_data(data_path, sub_ids, cells, movies, cell_folders, overwrite = overwrite_ephys_data)
    volpy_results = volpy.main(data_path, sub_ids, cells, cell_folders, movies)
    sd_accuracy = spike_detection_accuracy(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, ephys_times)
    #corr = get_correlation(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, ephys_times)
    #corr_spctrm = correlation_spectrum.correlation_spectrum(data_path, ephys_data, volpy_results, sub_ids, cells, movies, sd_accuracy, make_plots = plot_corr_spctrm, overwrite = overwrite_csd_data)

    if make_plots:
        plot_ephys_and_imaging(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, sd_accuracy)

def get_correlation(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, ephys_times, sd_accuracy):

    corr = {sid: {} for sid in sub_ids}
    ephys_down_sampled = {sid: {} for sid in sub_ids}
    snr = []
    corr_all = []
    for sid in sub_ids:
        print('ANM{0}'.format(sid))
        try:
            with open('{0}{1}Correlation_ANM{2}.pkl'.format(data_path, sep, sid), 'rb') as f:
                corr[sid] = pkl.load(f)
            with open('{0}{1}Ephys_down_sampled_ANM{2}.pkl'.format(data_path, sep, sid), 'rb') as f:
                ephys_down_sampled[sid] = pkl.load(f)
            for cell in cells[sid]:
                for movie_idx in list(corr[sid][cell].keys()):
                    corr_all = np.append(corr_all, corr[sid][cell][movie_idx])
                    snr = np.append(snr, volpy_results[sid][cell][movies[sid][cell][movie_idx - 1]]['snr'])
        except:
            for cell in cells[sid]:
                print('     Cell {0}'.format(cell))
                corr[sid][cell] = {}
                ephys_down_sampled[sid][cell] = {}
                movie_idx = 0
                for movie in movies[sid][cell]:
                    movie_idx += 1
                    print('         Movie {0}'.format(movie_idx))
                    if not movie_idx in list(ephys_times[sid][cell].keys()):
                        continue
                    if sd_accuracy[sid][cell][movie]['true_pos'] == None:
                        continue

                    dff = np.reshape(volpy_results[sid][cell][movie]['dFF'], [-1])
                    trace = ephys_data[sid]['traces'][cell][movie]

                    im_frame_times = np.load('{0}{1}{2}{1}{3}{1}frame_times.npy'.format(data_path, sep, cell_folders[sid][cell], movie))
                    ephys_sample_times = ephys_data[sid]['timings'][cell][movie]
                    assert(np.sum(np.diff(ephys_sample_times)) > 0)

                    im_indices_keep = np.logical_and(im_frame_times > ephys_times[sid][cell][movie_idx][0], im_frame_times < ephys_times[sid][cell][movie_idx][1])
                    ephys_indices_keep = np.logical_and(ephys_sample_times > ephys_times[sid][cell][movie_idx][0], ephys_sample_times <= ephys_times[sid][cell][movie_idx][1])


                    ephys_sample_times = ephys_sample_times[ephys_indices_keep]
                    ephys_isi = np.diff(ephys_sample_times)

                    ephys_breaks = np.where(ephys_isi > 10*np.median(ephys_isi))[0]
                    for eb in ephys_breaks:
                        s1 = ephys_sample_times[eb]
                        s2 = ephys_sample_times[eb + 1]
                        i1 = np.where(im_frame_times > s1)[0][0]
                        i2 = np.where(im_frame_times > s2)[0][1]
                        im_indices_keep[i1:i2] = np.zeros(i2 - i1)

                    im_frame_times = im_frame_times[im_indices_keep]
                    dff = dff[im_indices_keep]
                    trace = trace[ephys_indices_keep]

                    # Create binned ephys trace to calculate correlation with dff
                    ephys_trace_down_sampled = np.zeros(len(im_frame_times))
                    s0 = 0
                    for i in tqdm(range(len(im_frame_times))):
                        s1 = np.where(ephys_sample_times > im_frame_times[i])[0][0]
                        assert(s1 < len(trace))
                        if not s1 > s0:
                            plt.figure()
                            plt.plot(im_frame_times)
                            plt.plot(ephys_sample_times)
                        assert(s1 > s0)
                        ephys_trace_down_sampled[i] = np.mean(trace[s0:s1])
                        s0 = s1

                    ephys_down_sampled[sid][cell][movie] = ephys_trace_down_sampled
                    corr_coefs = np.corrcoef(dff, ephys_trace_down_sampled)
                    corr[sid][cell][movie_idx] = corr_coefs[0, 1]
                    print('         Correlation = {0}'.format(corr_coefs[0, 1]))

                    corr_all = np.append(corr_all, corr[sid][cell][movie_idx])
                    snr = np.append(snr, volpy_results[sid][cell][movie]['snr'])

                    plt.figure(figsize = [20, 3], constrained_layout = True)
                    et = ephys_trace_down_sampled - np.min(ephys_trace_down_sampled)
                    et = et/np.max(et)
                    plt.plot(im_frame_times, et, color = 'blue', linewidth = 2, alpha = 0.4, label = 'Ephys')
                    d = dff - np.min(dff)
                    d = d/np.max(d)
                    plt.plot(im_frame_times, d, color = 'k', alpha = 1, linewidth = 0.5, label = '- dF/F')
                    plt.xlabel('Time (s)')
                    plt.legend()
                    plt.title('ANM{0} Cell {1} Movie {2}: correlation = {3}'.format(sid, cell, movie, corr[sid][cell][movie_idx]))
                    plt.savefig('{0}{1}Plots{1}Downsampled ephys vs dff{1}ANM{2}_Cell{3}_Movie{4}.png'.format(data_path, sep, sid, cell, movie))

            with open('{0}{1}Correlation_ANM{2}.pkl'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(corr[sid], f)
            with open('{0}{1}Ephys_down_sampled_ANM{2}.pkl'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(ephys_down_sampled[sid], f)

    print(snr)
    print(corr_all)
    plt.figure(figsize = [3, 3], constrained_layout = True)
    plt.scatter(snr, corr_all, color = 'k', marker = 'o')
    x = np.linspace(min(snr), max(snr), 10)
    popt, pcov = curve_fit(line, snr, corr_all, p0 = [1, 0])
    plt.plot(x, line(x, *popt), linestyle = 'dashed', color = 'k', linewidth = 1)
    plt.xlabel('SNR')
    plt.ylabel('Correlation between dF/F and\ndown-sampled ephys trace')
    plt.savefig('{0}{1}Plots{1}Correlation_vs_SNR.png'.format(data_path, sep))

    return corr

def plot_ephys_and_imaging(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, sd_accuracy):

    for sid in sub_ids:
        for cell in cells[sid]:
            movie_idx = 0
            for movie in movies[sid][cell]:
                movie_idx += 1
                if not movie in list(sd_accuracy[sid][cell].keys()):
                    continue
                if sd_accuracy[sid][cell][movie]['true_pos'] == None:
                    continue
                snr = volpy_results[sid][cell][movie]['snr']
                title = 'ANM{0} Cell {1} Movie {2} SNR {3} Ephys + dFF'.format(sid, cell, movie_idx, snr)
                fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, constrained_layout = True, figsize = [10, 4])

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
                ax[0].set_ylabel('-dF/F', fontsize = 15)
                ax[0].legend(fontsize = 15)

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
                ax[1].set_ylabel('mV', fontsize = 15)
                ax[1].set_xlabel('Time (s)', fontsize = 15)
                ax[1].legend(fontsize = 15)

                fig.suptitle(title)
                fig.savefig('{0}{1}Plots{1}{2}.png'.format(data_path, sep, title))

def spike_detection_accuracy(data_path, sub_ids, cells, movies, cell_folders, ephys_data, volpy_results, ephys_times, peak_jitter_frames = 1, f1_cutoff = 0.7, make_plots = True):

    spike_detection_accuracy = {sid: {cell: {} for cell in cells[sid]} for sid in sub_ids}
    snr = []
    tp_all = []
    fn_all = []
    fp_all = []
    for sid in sub_ids:
        for cell in cells[sid]:
            movie_idx = 0
            for movie in movies[sid][cell]:
                movie_idx += 1
                spike_detection_accuracy[sid][cell][movie] = {}

                if not movie_idx in list(ephys_times[sid][cell].keys()):
                    spike_detection_accuracy[sid][cell][movie]['true_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_neg'] = None
                    spike_detection_accuracy[sid][cell][movie]['ephys_spikes_detection'] = None
                    spike_detection_accuracy[sid][cell][movie]['im_spikes_detection'] = None
                    continue

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

                if len(ephys_spike_times) < 2:
                    spike_detection_accuracy[sid][cell][movie]['true_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_pos'] = None
                    spike_detection_accuracy[sid][cell][movie]['false_neg'] = None
                    spike_detection_accuracy[sid][cell][movie]['ephys_spikes_detection'] = None
                    spike_detection_accuracy[sid][cell][movie]['im_spikes_detection'] = None
                    continue

                min_isi_ephys = np.min(np.abs(np.diff(ephys_spike_times)))
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
                    if len(ephys_spikes) == 0:
                        spike_detection_accuracy[sid][cell][movie]['true_pos'] = None
                        spike_detection_accuracy[sid][cell][movie]['false_pos'] = None
                        spike_detection_accuracy[sid][cell][movie]['false_neg'] = None
                        spike_detection_accuracy[sid][cell][movie]['ephys_spikes_detection'] = None
                        spike_detection_accuracy[sid][cell][movie]['im_spikes_detection'] = None
                        continue

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


    print(tp_all)
    print(fp_all)
    print(fn_all)

    if make_plots:
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [8, 3], constrained_layout = True)
        ax[0].scatter(snr, tp_all, color = 'k', marker = '.')
        x = np.linspace(min(snr), max(snr), 10)
        popt, pcov = curve_fit(line, snr, tp_all, p0 = [1, 0])
        ax[0].plot(x, line(x, *popt), linestyle = 'dashed', color = 'k', linewidth = 2)
        ax[0].set_ylabel('Sensitivity')
        ax[0].set_xlabel('SNR')
        ax[1].scatter(snr, np.divide(tp_all, tp_all + fp_all), color = 'k', marker = '.')
        popt, pcov = curve_fit(line, snr, np.divide(tp_all, tp_all + fp_all), p0 = [-1, 0])
        ax[1].plot(x, line(x, *popt) , linestyle = 'dashed', color = 'k', linewidth = 2)
        ax[1].set_ylabel('Precision')
        ax[1].set_xlabel('SNR')
        f1_all = np.divide(2*tp_all, 2*tp_all + fp_all + fn_all)
        ax[2].scatter(snr, f1_all, color = 'k', marker = 'o')
        popt, pcov = curve_fit(line, snr, f1_all, p0 = [1, 0])
        ax[2].plot(x, line(x, *popt), linestyle = 'dashed', color = 'k', linewidth = 1)
        snr_cutoff = (f1_cutoff - popt[1])/popt[0]
        print('SNR cutoff: {0}'.format(snr_cutoff))
        x = np.linspace(min(snr), snr_cutoff, 10)
        y = np.linspace(min(f1_all), f1_cutoff, 10)
        #ax[2].plot(np.ones(10)*snr_cutoff, y, linestyle = 'dashed', color = 'gray', linewidth = 2)
        #ax[2].plot(x, np.ones(10)*f1_cutoff, linestyle = 'dashed', color = 'gray', linewidth = 2)

        ax[2].set_ylabel('F1 score')
        ax[2].set_xlabel('SNR')
        fig.savefig('{0}{1}Plots{1}Spike detection vs SNR.png'.format(data_path, sep))

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
        print('     Cell {0}'.format(cell))
        for movie in movies[sid][cell]:
            print('         Movie {0}'.format(movie))
            ephys_dir = '{0}{1}{2}{1}{3}{1}ephys'.format(data_path, sep, cell_folders[sid][cell], movie)
            ephys_files = os.listdir(ephys_dir)
            ephys_files = [file for file in ephys_files if file.endswith('npz')]
            ephys_files = natsorted(ephys_files)
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
            ephys_files = natsorted(ephys_files)
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
            ephys_files = natsorted(ephys_files)
            total_points = 0
            for file in ephys_files:
                data = np.load('{0}{1}{2}'.format(ephys_dir, sep, file))
                trace = data['voltage']
                spikes_sweep =  signal.find_peaks(trace, height = spike_thresh_mv)[0]
                spikes[cell][movie] = np.append(spikes[cell][movie], spikes_sweep + total_points)
                total_points += len(trace)
    return spikes

def detect_spikes_simple_threshold(dff, thresh_sigma = 3, make_plot = False, plot_title = ''):

    dff = dff - np.mean(dff)
    thresh = thresh_sigma*np.std(dff)
    peaks = find_peaks(dff, height = thresh)[0]

    if make_plot:
        plt.figure(figsize = [20, 3], constrained_layout = True)
        plt.plot(dff, color = 'k', linewidth = 0.8)
        plt.scatter(peaks, dff[peaks], marker = '.', color = 'r')
        plt.title(plot_title)

    return peaks
