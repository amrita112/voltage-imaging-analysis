from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.signal import sosfilt

from behavior_responses import process_bpod_data
from segmentation import get_roi_arrays
from pre_processing import trial_tiff_stacks
from volpy import quality_control

def get_population_psth(population_data_path, data_paths, metadata_file,
                        overwrite_spike_psth = False, overwrite_dFF_psth = False,
                        filter_freq = None, filter_type = 'lp',
                        bin_size_ms = 100, plot_psths = True):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        snr_cutoff = qc_results['snr_cutoff']
        cells = qc_results['cells'][float(snr_cutoff)]
        blocks = qc_results['blocks'][float(snr_cutoff)]
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'rb') as f:
            population_psth = pkl.load(f)
            spike_psth = population_psth['spikes'] # Matrix of number of neurons X number of bins

    except:
        overwrite_spike_psth = True

    try:
        with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'rb') as f:
            population_psth = pkl.load(f)
            dFF_psth = population_psth['dFF'] # Matrix of number of neurons X number of frames
    except:
        overwrite_dFF_psth = True

    if overwrite_spike_psth and overwrite_dFF_psth:
        population_psth = {}

    if overwrite_spike_psth:

        spike_times_trials = {}
        min_n_bins = np.inf
        print('Loading spike trains')
        for movie in movies:
            print('    Movie {0}'.format(movie))
            spike_times_trials[movie] = get_spike_times_trials(data_paths[movie], metadata_file,
                                            list(cells[movie].astype(int)), blocks[movie])

            bin_edges_ms = np.arange(0, spike_times_trials[movie]['max_spike_time']*1000 + 3*bin_size_ms, bin_size_ms)
            n_bins = len(bin_edges_ms) - 1
            if n_bins < min_n_bins:
                min_n_bins = n_bins
        n_bins = min_n_bins
        bin_edges_ms = np.arange(0, (n_bins + 1)*bin_size_ms, bin_size_ms)

        population_psth['spikes'] = np.zeros([total_cells, n_bins*2])

        go_cue_time = np.zeros(len(movies))
        sample_end_time = np.zeros(len(movies))
        sample_start_time = np.zeros(len(movies))

        cell_no = 0
        print('Calculating PSTHs')
        for movie in movies:
            print('    Movie {0}'.format(movie))

            trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_paths[movie], metadata_file)
            with open('{0}{1}{2}'.format(data_paths[movie], sep, metadata_file), 'rb') as f:
                metadata = pkl.load(f)
            sessions_to_process = metadata['sessions_to_process']
            roi_arrays = get_roi_arrays.get_roi_arrays(data_paths[movie], metadata_file)
            n_cells = roi_arrays[sessions_to_process[0]].shape[0]
            psth = {cell: {'left_corr':    np.zeros([n_bins, 1]),
                           'right_corr':   np.zeros([n_bins, 1]),
                           'left_inc':     np.zeros([n_bins, 1]),
                           'right_inc':    np.zeros([n_bins, 1])}
                       for cell in range(n_cells)}

            for session in list(trial_types_left_right_cor_inc.keys()):

                types = trial_types_left_right_cor_inc[session]
                n_trials = len(types)
                for trial in range(n_trials):
                    type_string = {1: 'right_corr', 2: 'left_corr', 3: 'right_inc', 4: 'left_inc'}.get(types[trial])
                    if types[trial] > 0:
                        for cell in range(n_cells):
                            if trial in spike_times_trials[movie][cell][session].keys():
                                psth_cell_trial = np.histogram(spike_times_trials[movie][cell][session][trial]*1000, bin_edges_ms)
                                psth[cell][type_string] = np.append(psth[cell][type_string],
                                                                                  np.reshape(psth_cell_trial[0], [n_bins, 1]),
                                                                                 axis = 1)

            for cell in range(n_cells):
                if cell in list(cells[movie].astype(int)):
                    psth_cell_left_corr = psth[cell]['left_corr'][:, 1:]
                    population_psth['spikes'][cell_no, 0:n_bins] = np.mean(psth_cell_left_corr, axis = 1)*1000/bin_size_ms
                    psth_cell_right_corr = psth[cell]['right_corr'][:, 1:]
                    population_psth['spikes'][cell_no, n_bins:] = np.mean(psth_cell_right_corr, axis = 1)*1000/bin_size_ms
                    cell_no += 1

            go_cue_time[movies.index(movie)] = process_bpod_data.get_go_cue_time(data_paths[movie], metadata_file)
            sample_end_time[movies.index(movie)] = process_bpod_data.get_sample_end_time(data_paths[movie], metadata_file)
            sample_start_time[movies.index(movie)] = process_bpod_data.get_sample_start_time(data_paths[movie], metadata_file)


        var_go_cue_time = np.std(go_cue_time)
        var_sample_end_time = np.std(sample_end_time)
        var_sample_start_time = np.std(sample_start_time)
        if np.any([var_go_cue_time > 0.0000001, var_sample_end_time > 0.0000001, var_sample_start_time > 0.0000001]):
            print('.........................')
            print('WARNING: GO CUE TIME, SAMPLE END TIME AND/OR SAMPLE START TIME ARE NOT THE SAME FOR ALL MOVIES.')
            print('.........................')
            print('Go cue times: {0}'.format(go_cue_time))
            print('Sample end times: {0}'.format(sample_end_time))
            print('Sample start times: {0}'.format(sample_start_time))

        population_psth['sample_start_bin_left'] = 0
        population_psth['sample_end_bin_left'] = int((sample_end_time[0] - sample_start_time[0])*1000/bin_size_ms)
        population_psth['go_cue_bin_left'] = int((go_cue_time[0] - sample_start_time[0])*1000/bin_size_ms)

        population_psth['sample_start_bin_right'] = n_bins - 1
        population_psth['sample_end_bin_right'] = n_bins + int((sample_end_time[0] - sample_start_time[0])*1000/bin_size_ms)
        population_psth['go_cue_bin_right'] = n_bins + int((go_cue_time[0] - sample_start_time[0])*1000/bin_size_ms)

        population_tvec = np.zeros(n_bins*2)
        population_tvec[:n_bins] = bin_edges_ms[:-1]/1000 - go_cue_time[0] + sample_start_time[0]
        population_tvec[n_bins:] = bin_edges_ms[:-1]/1000 - go_cue_time[0] + sample_start_time[0]
        population_psth['tvec'] = population_tvec

    if overwrite_dFF_psth:

        dFF_trials = {}
        min_n_frames = np.inf
        print('Loading dFF traces')
        for movie in movies:
            print('    Movie {0}'.format(movie))
            dFF_trials[movie] = get_dFF_trials(data_paths[movie], metadata_file,
                                            list(cells[movie].astype(int)), blocks[movie],
                                            filter_freq = filter_freq, filter_type = filter_type)

            max_n_frames_movie = 0
            trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_paths[movie], metadata_file)
            for session in list(trial_types_left_right_cor_inc.keys()):

                types = trial_types_left_right_cor_inc[session]
                n_trials = len(types)
                assert(n_trials == len(list(dFF_trials[movie]['frames_per_trial'][session].keys())))
                for trial in range(n_trials):
                    type_string = {1: 'right_corr', 2: 'left_corr', 3: 'right_inc', 4: 'left_inc'}.get(types[trial])
                    if np.logical_or(types[trial] == 1, types[trial] == 2):
                        if dFF_trials[movie]['frames_per_trial'][session][trial] > max_n_frames_movie:
                            max_n_frames_movie = dFF_trials[movie]['frames_per_trial'][session][trial]

            if max_n_frames_movie < min_n_frames:
                min_n_frames = max_n_frames_movie

        population_psth['dFF'] = np.zeros([total_cells, min_n_frames*2])

        go_cue_time = np.zeros(len(movies))
        sample_end_time = np.zeros(len(movies))
        sample_start_time = np.zeros(len(movies))

        cell_no = 0
        print('Calculating PSTHs')
        for movie in movies:
            print('    Movie {0}'.format(movie))

            trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_paths[movie], metadata_file)
            with open('{0}{1}{2}'.format(data_paths[movie], sep, metadata_file), 'rb') as f:
                metadata = pkl.load(f)
            sessions_to_process = metadata['sessions_to_process']
            roi_arrays = get_roi_arrays.get_roi_arrays(data_paths[movie], metadata_file)
            n_cells = roi_arrays[sessions_to_process[0]].shape[0]
            psth = {cell: {'left_corr':    np.zeros([min_n_frames, 1]),
                           'right_corr':   np.zeros([min_n_frames, 1]),
                           'left_inc':     np.zeros([min_n_frames, 1]),
                           'right_inc':    np.zeros([min_n_frames, 1])}
                       for cell in range(n_cells)}

            for session in list(trial_types_left_right_cor_inc.keys()):

                types = trial_types_left_right_cor_inc[session]
                n_trials = len(types)
                for trial in range(n_trials):
                    type_string = {1: 'right_corr', 2: 'left_corr', 3: 'right_inc', 4: 'left_inc'}.get(types[trial])
                    if types[trial] > 0:
                        for cell in range(n_cells):
                            if trial in dFF_trials[movie][cell][session].keys():
                                n_frames_trial = len(dFF_trials[movie][cell][session][trial])
                                if n_frames_trial > min_n_frames:
                                    psth[cell][type_string] = np.append(psth[cell][type_string],
                                                            np.reshape(dFF_trials[movie][cell][session][trial][:min_n_frames],
                                                                            [min_n_frames, 1]), axis = 1)
                                else:
                                    dFF_trial = np.pad(dFF_trials[movie][cell][session][trial], (0, min_n_frames - n_frames_trial),
                                                        constant_values = (0, 0))
                                    psth[cell][type_string] = np.append(psth[cell][type_string],
                                                                        np.reshape(dFF_trial, [min_n_frames, 1]), axis = 1)


            for cell in range(n_cells):
                if cell in list(cells[movie].astype(int)):
                    psth_cell_left_corr = psth[cell]['left_corr'][:, 1:]
                    population_psth['dFF'][cell_no, 0:min_n_frames] = np.mean(psth_cell_left_corr, axis = 1)
                    psth_cell_right_corr = psth[cell]['right_corr'][:, 1:]
                    population_psth['dFF'][cell_no, min_n_frames:] = np.mean(psth_cell_right_corr, axis = 1)
                    cell_no += 1

            go_cue_time[movies.index(movie)] = process_bpod_data.get_go_cue_time(data_paths[movie], metadata_file)
            sample_end_time[movies.index(movie)] = process_bpod_data.get_sample_end_time(data_paths[movie], metadata_file)
            sample_start_time[movies.index(movie)] = process_bpod_data.get_sample_start_time(data_paths[movie], metadata_file)


        var_go_cue_time = np.std(go_cue_time)
        var_sample_end_time = np.std(sample_end_time)
        var_sample_start_time = np.std(sample_start_time)
        if np.any([var_go_cue_time > 0.0000001, var_sample_end_time > 0.0000001, var_sample_start_time > 0.0000001]):
            print('.........................')
            print('WARNING: GO CUE TIME, SAMPLE END TIME AND/OR SAMPLE START TIME ARE NOT THE SAME FOR ALL MOVIES.')
            print('.........................')
            print('Go cue times: {0}'.format(go_cue_time))
            print('Sample end times: {0}'.format(sample_end_time))
            print('Sample start times: {0}'.format(sample_start_time))

        frame_rate = dFF_trials[movie]['frame_rate']

        population_psth['sample_start_frame_left'] = 0
        population_psth['sample_end_frame_left'] = int((sample_end_time[0] - sample_start_time[0])*frame_rate)
        population_psth['go_cue_frame_left'] = int((go_cue_time[0] - sample_start_time[0])*frame_rate)

        population_psth['sample_start_frame_right'] = min_n_frames - 1
        population_psth['sample_end_frame_right'] = min_n_frames + int((sample_end_time[0] - sample_start_time[0])*frame_rate)
        population_psth['go_cue_frame_right'] = min_n_frames + int((go_cue_time[0] - sample_start_time[0])*frame_rate)

        population_tvec = np.zeros(min_n_frames*2)
        population_tvec[:min_n_frames] = list(range(min_n_frames))/frame_rate - go_cue_time[0] + sample_start_time[0]
        population_tvec[min_n_frames:] = list(range(min_n_frames))/frame_rate - go_cue_time[0] + sample_start_time[0]
        population_psth['dFF_tvec'] = population_tvec

    with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'wb') as f:
        pkl.dump(population_psth, f)

    return population_psth


def get_spike_times_trials(data_path, metadata_file, good_cells, good_blocks):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    batch_data = metadata['batch_data']

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_times = output['frame_and_trial_times']['frame_times']

    n_frames_per_trial = trial_tiff_stacks.get_n_frames_per_trial(data_path, metadata_file)

    volpy_results_file = metadata['volpy_results_file']
    with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
        volpy_results = pkl.load(f)

    roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)
    n_cells = roi_arrays[sessions_to_process[0]].shape[0]

    spike_times_trials = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}

    max_spike_time = 0
    for session in sessions_to_process:

        n_batches = batch_data[session]['n_batches']
        first_trials = np.array(batch_data[session]['first_trials']).astype(int)
        last_trials = np.array(batch_data[session]['last_trials']).astype(int)

        cum_frames_per_trial = np.cumsum(n_frames_per_trial[session]).astype(int)
        cum_frames_per_trial = np.insert(cum_frames_per_trial, 0, 0)

        for batch in range(n_batches):

            estimates = volpy_results[session][batch]['vpy']
            first_frame_batch = cum_frames_per_trial[first_trials[batch]]

            for trial in range(first_trials[batch], last_trials[batch]):

                frames = list(range(cum_frames_per_trial[trial], cum_frames_per_trial[trial + 1])) # 0 is the first frame in the session
                frames_batch = frames - first_frame_batch                                         # 0 is the first frame in the batch
                frame_times_trial = frame_times[session][frames]
                frame_times_trial = frame_times_trial - frame_times_trial[0]

                for cell in range(n_cells):
                    if cell in good_cells:
                        if batch > good_blocks[good_cells.index(cell)]:
                            continue
                        else:
                            spike_frames_cell = estimates['spikes'][cell]
                            spike_frames_trial = [frame for frame in spike_frames_cell if frame in frames_batch]
                            spike_frames_trial = spike_frames_trial - frames_batch[0]
                            spike_times_trials[cell][session][trial] = frame_times_trial[spike_frames_trial.astype(int)]
                            if len(spike_times_trials[cell][session][trial]) > 0:
                                max_time = np.max(spike_times_trials[cell][session][trial])
                                if max_time > max_spike_time:
                                    max_spike_time = max_time

    spike_times_trials['max_spike_time'] = max_spike_time
    return spike_times_trials

def get_dFF_trials(data_path, metadata_file, good_cells, good_blocks, filter_freq = None, filter_type = 'lp'):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    batch_data = metadata['batch_data']

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_rate = output['frame_and_trial_times']['frame_rate'][sessions_to_process[0]]

    n_frames_per_trial = trial_tiff_stacks.get_n_frames_per_trial(data_path, metadata_file)

    volpy_results_file = metadata['volpy_results_file']
    with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
        volpy_results = pkl.load(f)

    roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)
    n_cells = roi_arrays[sessions_to_process[0]].shape[0]

    dFF_trials = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}
    dFF_trials['frames_per_trial'] = {session: {} for session in sessions_to_process}

    if not filter_freq == None:
        print('Filtering dF/F')
        
    for session in sessions_to_process:

        n_batches = batch_data[session]['n_batches']
        first_trials = np.array(batch_data[session]['first_trials']).astype(int)
        last_trials = np.array(batch_data[session]['last_trials']).astype(int)

        cum_frames_per_trial = np.cumsum(n_frames_per_trial[session]).astype(int)
        cum_frames_per_trial = np.insert(cum_frames_per_trial, 0, 0)

        for batch in range(n_batches):

            estimates = volpy_results[session][batch]['vpy']
            dFF_all = estimates['dFF']
            dFF_filt = {}
            for cell in range(n_cells):
                dFF_filt[cell] = filt_dFF(dFF_all[cell], filter_freq, frame_rate, filter_type = filter_type)

            first_frame_batch = cum_frames_per_trial[first_trials[batch]]

            for trial in range(first_trials[batch], last_trials[batch]):

                frames_trial = list(range(cum_frames_per_trial[trial], cum_frames_per_trial[trial + 1])) # 0 is the first frame in the session
                frames_batch = frames_trial - first_frame_batch
                dFF_trials['frames_per_trial'][session][trial] = len(frames_trial)                                      # 0 is the first frame in the batch

                for cell in range(n_cells):
                    if cell in good_cells:
                        if batch > good_blocks[good_cells.index(cell)]:
                            continue
                        else:
                            dFF_cell = dFF_filt[cell]
                            dFF_trials[cell][session][trial] = dFF_cell[frames_batch]


    dFF_trials['frame_rate'] = frame_rate
    return dFF_trials


def filt_dFF(trace, filter_freq, frame_rate, filter_type = 'lp', order = 3):

    if not filter_freq == None:
        sos = butter(order, filter_freq, filter_type, fs = frame_rate, output = 'sos')
        signal = sosfilt(sos, trace)
    else:
        signal = trace
    return signal
