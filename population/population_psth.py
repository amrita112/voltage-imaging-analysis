from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.signal import sosfilt
from tqdm import tqdm

from behavior_responses import process_bpod_data
from segmentation import get_roi_arrays
from pre_processing import trial_tiff_stacks
from volpy import quality_control
from population import plots
from population import clustering

def get_population_psth(population_data_path, movies, data_paths, metadata_file, genotype, overwrite_spike_psth = False, overwrite_dFF_psth = False, overwrite_trial_time_points = False, overwrite_spike_times_trials = False, make_plot_spike_psth = True, make_plot_dFF_psth = True, make_plot_for_each_cell = False, filter_freq = None, filter_type = 'lp', bin_size_ms = 100, plot_psths = True, save_fig = False):

    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        snr_cutoff = qc_results['snr_cutoff']
        cells = qc_results['cells']
        blocks = qc_results['blocks']
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}trial_timepoints.py'.format(population_data_path, sep), 'rb') as f:
            trial_timepoints = pkl.load(f)
    except:
        overwrite_trial_time_points = True

    try:
        with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'rb') as f:
            population_psth = pkl.load(f)
            spike_psth = population_psth['spikes'] # Matrix of number of neurons X number of bins

    except:
        overwrite_spike_psth = True
        population_psth = {}

    #try:
    #    with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'rb') as f:
    #        population_psth = pkl.load(f)
    #        dFF_psth = population_psth['dFF'] # Matrix of number of neurons X number of frames
    #except:
    #    overwrite_dFF_psth = True

    if overwrite_trial_time_points:
        trial_timepoints = get_trial_time_points(data_paths, metadata_file, movies)
        with open('{0}{1}trial_timepoints.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(trial_timepoints, f)

    go_cue_time = trial_timepoints['go_cue_time']
    sample_start_time = trial_timepoints['sample_start_time']
    sample_end_time = trial_timepoints['sample_end_time']
    movies_keep = trial_timepoints['movies_keep']
    sample_period = trial_timepoints['sample_period']
    delay_period = trial_timepoints['delay_period']

    if overwrite_spike_psth and overwrite_dFF_psth:
        population_psth = {}

    if overwrite_spike_psth:
        population_psth['spikes'] = get_spike_psth(data_paths, metadata_file,
                                                   movies_keep, cells, total_cells,
                                                   blocks, bin_size_ms,
                                                   go_cue_time, sample_end_time, sample_start_time,
                                                   make_plot_for_each_cell = make_plot_for_each_cell,
                                                   overwrite_spike_times_trials = overwrite_spike_times_trials)
        spike_psth = population_psth['spikes']
        with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(population_psth, f)

    if make_plot_spike_psth:

        tvec = spike_psth['tvec'] - spike_psth['tvec'][spike_psth['go_cue_bin_left']]
        ticks = [0,
             spike_psth['sample_start_bin_left'],
             spike_psth['sample_end_bin_left'],
             spike_psth['go_cue_bin_left'],
             spike_psth['trial_start_bin_right'],
             spike_psth['sample_start_bin_right'],
             spike_psth['sample_end_bin_right'],
             spike_psth['go_cue_bin_right']]
        ndnf_movies = [movie for (movie, gtype) in genotype.items() if np.logical_and(gtype == 'Ndnf', movie in movies)]
        vip_movies = [movie for (movie, gtype) in genotype.items() if np.logical_and(gtype == 'Vip', movie in movies)]
        cell_order_ndnf = []
        cell_order_vip = []
        total_cells = 0
        for movie in movies:
            n_cells_movie = len(cells[movie])
            if genotype[movie] == 'Ndnf':
                cell_order_ndnf = np.append(cell_order_ndnf, list(range(total_cells, total_cells + n_cells_movie)))
            else:
                cell_order_vip = np.append(cell_order_vip, list(range(total_cells, total_cells + n_cells_movie)))
            total_cells += n_cells_movie
        cell_order = np.concatenate([cell_order_ndnf, cell_order_vip]).astype(int)
        cluster_boundaries = [len(cell_order_ndnf), len(cell_order_vip) + len(cell_order_ndnf)]
        cluster_names = ['Ndnf', 'Vip']

        # Spike PSTH
        plots.plot_spike_psth(spike_psth['psth_array'], tvec, ticks,
                                cell_order = cell_order, cluster_boundaries = cluster_boundaries,
                                cluster_names = cluster_names,
                              save_fig = save_fig,
                              save_path = '{0}{1}Plots{1}Clustering{1}Spikes{1}PSTH_population.png'.format(population_data_path, sep))

        # Z-scored spike PSTH
        spike_psth_z_scored = clustering.norm_vectors(spike_psth['psth_array'])
        plots.plot_spike_psth(spike_psth_z_scored, tvec, ticks,
                                cell_order = cell_order, cluster_boundaries = cluster_boundaries,
                                cluster_names = cluster_names,
                              colorbar_label = 'Z-scored firing rate',
                              save_fig = True,
                              save_path = '{0}{1}Plots{1}Clustering{1}Spikes{1}Normalized_PSTH_population.png'.format(population_data_path, sep))

    if overwrite_dFF_psth:
        population_psth['dFF'] = get_dFF_psth(data_paths, metadata_file, movies_keep, cells, total_cells, blocks, bin_size_ms,
        go_cue_time, sample_end_time, sample_start_time, filter_freq, filter_type)
        with open('{0}{1}population_psth.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(population_psth, f)
        dFF_psth = population_psth['dFF']

    if make_plot_dFF_psth:

        tvec = dFF_psth['tvec'] - dFF_psth['tvec'][dFF_psth['go_cue_frame_left']]
        ticks = [0,
             dFF_psth['sample_start_frame_left'],
             dFF_psth['sample_end_frame_left'],
             dFF_psth['go_cue_frame_left'],
             dFF_psth['trial_start_frame_right'],
             dFF_psth['sample_start_frame_right'],
             dFF_psth['sample_end_frame_right'],
             dFF_psth['go_cue_frame_right']]
        ndnf_movies = [movie for (movie, gtype) in genotype.items() if np.logical_and(gtype == 'Ndnf', movie in movies)]
        vip_movies = [movie for (movie, gtype) in genotype.items() if np.logical_and(gtype == 'Vip', movie in movies)]
        cell_order_ndnf = []
        cell_order_vip = []
        total_cells = 0
        for movie in movies:
            n_cells_movie = len(cells[movie])
            if genotype[movie] == 'Ndnf':
                cell_order_ndnf = np.append(cell_order_ndnf, list(range(total_cells, total_cells + n_cells_movie)))
            else:
                cell_order_vip = np.append(cell_order_vip, list(range(total_cells, total_cells + n_cells_movie)))
            total_cells += n_cells_movie
        cell_order = np.concatenate([cell_order_ndnf, cell_order_vip]).astype(int)
        cluster_boundaries = [len(cell_order_ndnf), len(cell_order_vip) + len(cell_order_ndnf)]
        cluster_names = ['Ndnf', 'Vip']

        # dFF PSTH
        plots.plot_spike_psth(dFF_psth['psth_array'], tvec, ticks,
                                cell_order = cell_order, cluster_boundaries = cluster_boundaries,
                                cluster_names = cluster_names,
                              save_fig = True,
                              save_path = '{0}{1}Plots{1}dFF_PSTH_population.png'.format(population_data_path, sep))

        # Z-scored dFF PSTH
        dFF_psth_z_scored = clustering.norm_vectors(dFF_psth['psth_array'])
        plots.plot_spike_psth(dFF_psth_z_scored, tvec, ticks,
                                cell_order = cell_order, cluster_boundaries = cluster_boundaries,
                                cluster_names = cluster_names,
                              colorbar_label = 'Z-scored dF/F',
                              save_fig = True,
                              save_path = '{0}{1}Plots{1}Normalized_dFF_PSTH_population.png'.format(population_data_path, sep))


    return population_psth

def get_trial_time_points(data_paths, metadata_file, movies, min_sample_period_s = 1.13, max_sample_period_s = 1.15, min_delay_period_s = 1.19, max_delay_period_s = 1.21):

    go_cue_time = []
    sample_end_time = []
    sample_start_time = []
    movies_keep = []
    sample_period_all = []
    delay_period_all = []

    print('Loading go cue time, sample end time and sample start time')

    for movie in tqdm(movies):

        sample_start = process_bpod_data.get_sample_start_time(data_paths[movie], metadata_file)
        sample_end = process_bpod_data.get_sample_end_time(data_paths[movie], metadata_file)
        sample_period = sample_end - sample_start
        if np.logical_or(sample_period > max_sample_period_s, sample_period < min_sample_period_s):
            print('Movie {0} skipped: Sample period = {1}s'.format(movie, sample_period))
            continue

        go_cue = process_bpod_data.get_go_cue_time(data_paths[movie], metadata_file)
        delay_period = go_cue - sample_end
        if np.logical_or(delay_period > max_delay_period_s, delay_period < min_delay_period_s):
            print('Movie {0} skipped: delay period = {1}s'.format(movie, delay_period))
            continue

        movies_keep = np.append(movies_keep, movie)
        sample_start_time = np.append(sample_start_time, sample_start)
        sample_end_time = np.append(sample_end_time, sample_end)
        go_cue_time = np.append(go_cue_time, go_cue)
        sample_period_all = np.append(sample_period_all, sample_period)
        delay_period_all = np.append(delay_period_all, delay_period)

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

    trial_timepoints = {'go_cue_time': go_cue_time,
                        'sample_end_time': sample_end_time,
                        'sample_start_time': sample_start_time,
                        'movies_keep': list(movies_keep),
                        'sample_period': sample_period_all,
                        'delay_period': delay_period_all}

    return trial_timepoints

def get_dFF_psth(data_paths, metadata_file, movies, cells, total_cells, blocks, bin_size_ms, go_cue_time, sample_end_time, sample_start_time, filter_freq, filter_type):

    output = {}
    dFF_trials = {}
    pre_sample_frames = {}
    sample_frames = {}
    delay_frames = {}
    response_frames = {}
    n_usable_trials = {}
    left_corr_trial_nos = {}
    right_corr_trial_nos = {}
    min_pre_sample_frames = np.inf
    min_response_frames = np.inf
    min_delay_frames = np.inf

    print('Loading dFF traces')
    for movie in tqdm(movies):
        #print('Movie {0}'.format(movie))
        #print('     {0}'.format(data_paths[movie]))
        if len(cells[movie]) == 0:
            print('         Movie {0}: No cells pass SNR cutoff'.format(movie))
            continue
        dFF_trials[movie] = get_dFF_trials(data_paths[movie], metadata_file,
                                        list(cells[movie].astype(int)), blocks[movie],
                                        filter_freq = filter_freq, filter_type = filter_type)

    avg_frame_rate = np.mean([dFF_trials[movie]['frame_rate'] for movie in movies])

    for movie in movies:
        if len(cells[movie]) == 0:
            print('         Movie {0}: No cells pass SNR cutoff'.format(movie))
            continue

        min_trial_time_s = dFF_trials[movie]['min_trial_time_s']
        frame_times = np.arange(0, min_trial_time_s, 1/avg_frame_rate)
        pre_sample_frames[movie] = np.where(frame_times < sample_start_time[movies.index(movie)])[0]
        sample_frames[movie] = len(pre_sample_frames[movie]) + np.where(frame_times[pre_sample_frames[movie][-1] + 1:] < sample_end_time[movies.index(movie)])[0]
        delay_frames[movie] = sample_frames[movie][-1] + 1 + np.where(frame_times[sample_frames[movie][-1] + 1:] < go_cue_time[movies.index(movie)])[0]
        response_frames[movie] = list(range(delay_frames[movie][-1] + 1, len(frame_times)))
        if len(pre_sample_frames[movie]) < min_pre_sample_frames:
            min_pre_sample_frames = len(pre_sample_frames[movie])
        if len(response_frames[movie]) < min_response_frames:
            min_response_frames = len(response_frames[movie])
        if len(delay_frames[movie]) < min_delay_frames:
            min_delay_frames = len(delay_frames[movie])

    #print('Sample frames: {0}'.format([len(v) for (k, v) in sample_frames.items()]))
    #print('Delay frames: {0}'.format([len(v) for (k, v) in delay_frames.items()]))
    #print('Sample end time: {0}'.format(sample_end_time))
    #print('Go cue time: {0}'.format(go_cue_time))
    #print('Delay period: {0}'.format(go_cue_time - sample_end_time))
    #print('Frame rate: {0}'.format([dFF_trials[movie]['frame_rate'] for movie in movies]))
    #print('Min trial time: {0}'.format([dFF_trials[movie]['min_trial_time_s'] for movie in movies]))

    # All movies have same number of frames in sample period and delay period
    assert(np.var([len(v) for (k, v) in sample_frames.items()]) == 0)
    n_delay_frames = [len(v) for (k, v) in delay_frames.items()]
    sem_delay_frames = np.std(n_delay_frames)/np.mean(n_delay_frames)
    assert(sem_delay_frames < 0.001)
    #assert(np.var([len(v) for (k, v) in delay_frames.items()]) == 0)

    print('Minimum pre-sample period = {0}s'.format(min_pre_sample_frames/400))
    print('Minimum response period = {0}s'.format(min_response_frames/400))

    n_frames = min_pre_sample_frames + len(sample_frames[movies[0]]) + min_delay_frames + min_response_frames

    dFF_psth = np.zeros([total_cells, n_frames*2])
    dFF_psth_random1 = np.zeros([total_cells, n_frames*2])
    dFF_psth_random2 = np.zeros([total_cells, n_frames*2])
    population_tvec = np.zeros(n_frames*2)
    population_tvec[0:n_frames] = np.linspace(0, n_frames/400, n_frames)
    population_tvec[n_frames:] = np.linspace(0, n_frames/400, n_frames)

    cell_no = 0
    total_cells = 0
    n_cells_zero_padded = 0

    print('Calculating PSTHs')
    for movie in tqdm(movies):
        assert(total_cells == cell_no)
        total_cells += len(cells[movie])
        if len(cells[movie]) == 0:
            print('         Movie {0}: No cells pass SNR cutoff'.format(movie))
            continue

        trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_paths[movie], metadata_file)
        with open('{0}{1}{2}'.format(data_paths[movie], sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
        sessions_to_process = metadata['sessions_to_process']
        roi_arrays = get_roi_arrays.get_roi_arrays(data_paths[movie], metadata_file)
        n_cells = roi_arrays[sessions_to_process[0]].shape[0]
        psth = {cell: {'left_corr':    np.zeros([n_frames, 1]),
                       'right_corr':   np.zeros([n_frames, 1]),
                       'left_inc':     np.zeros([n_frames, 1]),
                       'right_inc':    np.zeros([n_frames, 1])}
                   for cell in range(n_cells)}

        frames_movie = np.concatenate([pre_sample_frames[movie][-min_pre_sample_frames:],
                                        sample_frames[movie], delay_frames[movie][:min_delay_frames],
                                        response_frames[movie][:min_response_frames]])
        assert(len(frames_movie) == n_frames)

        left_corr_trial_nos[movie] = []
        right_corr_trial_nos[movie] = []
        trial_index = -1

        for session in list(trial_types_left_right_cor_inc.keys()):

            types = trial_types_left_right_cor_inc[session]
            n_trials = len(types)
            for trial in range(n_trials):
                type_string = {1: 'right_corr', 2: 'left_corr', 3: 'right_inc', 4: 'left_inc'}.get(types[trial])
                if types[trial] > 0:
                    cell_zero_padded = False
                    for cell in range(n_cells):
                        if trial in dFF_trials[movie][cell][session].keys():

                            dFF_cell = dFF_trials[movie][cell][session][trial]
                            if len(dFF_cell) < frames_movie[-1]:
                                last_frame = int(np.where(frames_movie == len(dFF_cell))[0])
                                dFF_cell = dFF_cell[frames_movie[:last_frame]]
                                dFF_cell = np.concatenate([dFF_cell, np.zeros(n_frames - len(dFF_cell))])
                                cell_zero_padded = True
                            else:
                                dFF_cell = dFF_cell[frames_movie]
                            dFF_cell = np.reshape(dFF_cell, [n_frames, 1])
                            psth[cell][type_string] = np.append(psth[cell][type_string], dFF_cell, axis = 1)
                    if cell_zero_padded:
                        n_cells_zero_padded += 1
            if types[trial] == 1:
                right_corr_trial_nos[movie] = np.append(right_corr_trial_nos[movie], trial_index)
            if types[trial] == 2:
                left_corr_trial_nos[movie] = np.append(left_corr_trial_nos[movie], trial_index)
            n_usable_trials[movie] = trial_index + 1

        for cell in range(n_cells):
            if cell in list(cells[movie].astype(int)):
                psth_cell_left_corr = psth[cell]['left_corr'][:, 1:]
                dFF_psth[cell_no, 0:n_frames] = np.mean(psth_cell_left_corr, axis = 1)
                psth_cell_right_corr = psth[cell]['right_corr'][:, 1:]
                dFF_psth[cell_no, n_frames:] = np.mean(psth_cell_right_corr, axis = 1)

                # Comput psth from two random divisions of the data, to get a better idea of ordering of cells by latency
                n_left_trials = psth_cell_left_corr.shape[1]
                left_trials_random1 = np.random.choice(list(range(n_left_trials)), int(n_left_trials/2), replace = False)
                left_trials_random2 = [trial for trial in list(range(n_left_trials)) if not trial in left_trials_random1]

                n_right_trials = psth_cell_right_corr.shape[1]
                right_trials_random1 = np.random.choice(list(range(n_right_trials)), int( n_right_trials/2), replace = False)
                right_trials_random2 = [trial for trial in list(range(n_right_trials)) if not trial in right_trials_random1]
                assert(len(left_trials_random1) + len(left_trials_random2) == n_left_trials)
                assert(len(right_trials_random1) + len(right_trials_random2) == n_right_trials)

                dFF_psth_random1[cell_no, 0:n_frames] = np.mean(psth_cell_left_corr[:, left_trials_random1], axis = 1)
                dFF_psth_random1[cell_no, n_frames:] = np.mean(psth_cell_right_corr[:, right_trials_random1], axis = 1)
                dFF_psth_random2[cell_no, 0:n_frames] = np.mean(psth_cell_left_corr[:, left_trials_random2], axis = 1)
                dFF_psth_random2[cell_no, n_frames:] = np.mean(psth_cell_right_corr[:, right_trials_random2], axis = 1)

                cell_no += 1

    print('{0} out of {1} cells 0-padded'.format(n_cells_zero_padded, total_cells))
    frame_rate = dFF_trials[movie]['frame_rate']

    output['psth_array'] = dFF_psth
    output['psth_array_random_subset1'] = dFF_psth_random1
    output['psth_array_random_subset2'] = dFF_psth_random2
    output['tvec'] = population_tvec
    output['sample_start_frame_left'] = min_pre_sample_frames - 1
    output['sample_end_frame_left'] = min_pre_sample_frames + len(sample_frames[movies[0]]) - 1
    output['go_cue_frame_left'] = min_pre_sample_frames + len(sample_frames[movies[0]]) + min_delay_frames - 1
    output['trial_start_frame_right'] = n_frames - 1
    output['sample_start_frame_right'] = n_frames + min_pre_sample_frames - 1
    output['sample_end_frame_right'] = n_frames + min_pre_sample_frames + len(sample_frames[movies[0]]) - 1
    output['go_cue_frame_right'] = n_frames + min_pre_sample_frames + len(sample_frames[movies[0]]) + min_delay_frames - 1
    output['n_trials'] = n_usable_trials
    output['left_corr_trial_nos'] = left_corr_trial_nos
    output['right_corr_trial_nos'] = right_corr_trial_nos

    return output

def get_spike_psth(data_paths, metadata_file, movies, cells, total_cells, blocks, bin_size_ms, go_cue_time, sample_end_time, sample_start_time, compute_random = False, make_plot_for_each_cell = False, overwrite_spike_times_trials = False, single_cell_psth_axes = []):

    output = {}
    spike_times_trials = {}
    pre_sample_bins = {}
    sample_bins = {}
    delay_bins = {}
    response_bins = {}
    n_usable_trials = {}
    left_corr_trial_nos = {}
    right_corr_trial_nos = {}
    min_pre_sample_bins = np.inf
    min_response_bins = np.inf

    print('Loading spike trains')
    for movie in tqdm(movies):
        #print('    Movie {0}'.format(movie))
        if len(cells[movie]) == 0:
            print('         Movie {0}: No cells pass SNR cutoff'.format(movie))
            continue
        trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_paths[movie], metadata_file)
        spike_times_trials[movie] = get_spike_times_trials(data_paths[movie], metadata_file,
                                        list(cells[movie].astype(int)), blocks[movie], go_cue_time[movies.index(movie)], trial_types_left_right_cor_inc, overwrite = overwrite_spike_times_trials)
        #bin_edges_ms = np.arange(0, spike_times_trials[movie]['max_spike_time']*1000 + 3*bin_size_ms, bin_size_ms)
        bin_edges_ms = np.arange(0, spike_times_trials[movie]['min_trial_time_s']*1000 + bin_size_ms, bin_size_ms)
        pre_sample_bins[movie] = np.where(bin_edges_ms < sample_start_time[movies.index(movie)]*1000)[0]
        sample_bins[movie] = len(pre_sample_bins[movie]) + np.where(bin_edges_ms[pre_sample_bins[movie][-1] + 1:] < sample_end_time[movies.index(movie)]*1000)[0]
        delay_bins[movie] = sample_bins[movie][-1] + 1 + np.where(bin_edges_ms[sample_bins[movie][-1] + 1:] < go_cue_time[movies.index(movie)]*1000)[0]
        response_bins[movie] = list(range(delay_bins[movie][-1] + 1, len(bin_edges_ms)))
        if len(pre_sample_bins[movie]) < min_pre_sample_bins:
            min_pre_sample_bins = len(pre_sample_bins[movie])
        if len(response_bins[movie]) < min_response_bins:
            min_response_bins = len(response_bins[movie])


    # All movies have same number of bins in sample period and delay period
    assert(np.var([len(v) for (k, v) in sample_bins.items()]) == 0)
    assert(np.var([len(v) for (k, v) in delay_bins.items()]) == 0)

    print('Minimum pre-sample period = {0}s'.format(min_pre_sample_bins*bin_size_ms/1000))
    print('Minimum response period = {0}s'.format(min_response_bins*bin_size_ms/1000))

    n_bins = min_pre_sample_bins + len(sample_bins[movies[0]]) + len(delay_bins[movies[0]]) + min_response_bins
    bin_edges_ms = np.arange(0, (n_bins + 1)*bin_size_ms, bin_size_ms)
    spike_count_bin_edges = np.cumsum([0, min_pre_sample_bins, len(sample_bins[movies[0]]),
                                        len(delay_bins[movies[0]]), min_response_bins])
    assert(spike_count_bin_edges[-1] == n_bins)
    assert(len(spike_count_bin_edges) == 5)
    population_tvec = np.zeros(n_bins*2)
#    population_tvec[:n_bins] = bin_edges_ms[:-1]/1000 - go_cue_time + sample_start_time
    population_tvec[:n_bins] = bin_edges_ms[1:]/1000
#    population_tvec[n_bins:] = bin_edges_ms[:-1]/1000 - go_cue_time + sample_start_time
    population_tvec[n_bins:] = bin_edges_ms[1:]/1000

    spike_psth = np.zeros([total_cells, n_bins*2])
    spike_psth_random1 = np.zeros([total_cells, n_bins*2])
    spike_psth_random2 = np.zeros([total_cells, n_bins*2])
    spike_count_trials_left = {}
    spike_count_trials_right = {}

    if make_plot_for_each_cell:
        if len(single_cell_psth_axes) == 0:
            single_cell_psth_axes = [None for cell in range(total_cells)]

    cell_no = 0
    total_cells = 0
    n_left_trials_cells = []
    n_right_trials_cells = []

    print('Calculating PSTHs')
    for movie in tqdm(movies):
        assert(total_cells == cell_no)
        total_cells += len(cells[movie])
        if len(cells[movie]) == 0:
            print('         Movie {0}: No cells pass SNR cutoff'.format(movie))
            continue
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
        bins_movie = np.concatenate([pre_sample_bins[movie][-min_pre_sample_bins:],
                                     sample_bins[movie], delay_bins[movie],
                                     response_bins[movie][:min_response_bins]])
        assert(len(bins_movie) == n_bins)
        bin_edges_ms_movie = np.arange(0, spike_times_trials[movie]['max_spike_time']*1000 + 3*bin_size_ms, bin_size_ms)
        assert(len(bin_edges_ms_movie) > n_bins)

        left_corr_trial_nos[movie] = []
        right_corr_trial_nos[movie] = []
        trial_index = -1

        for session in list(trial_types_left_right_cor_inc.keys()):

            types = trial_types_left_right_cor_inc[session]
            n_trials = len(types)
            for trial in range(n_trials):
                trial_index += 1
                type_string = {1: 'right_corr', 2: 'left_corr', 3: 'right_inc', 4: 'left_inc'}.get(types[trial])
                if types[trial] > 0:
                    for cell in range(n_cells):
                        if trial in spike_times_trials[movie][cell][session].keys():
                            spike_times_cell = spike_times_trials[movie][cell][session][trial]
                            psth_cell_trial = np.histogram(spike_times_cell*1000, bin_edges_ms_movie)[0]
                            psth_cell_trial = psth_cell_trial[bins_movie[0]:bins_movie[-1]]
                            assert(len(psth_cell_trial) <= n_bins)
                            if types[trial] in [1, 2]:
                                assert(len(psth_cell_trial) == n_bins - 1)
                            psth_cell_trial = np.pad(psth_cell_trial, (0, n_bins - len(psth_cell_trial)))
                            psth[cell][type_string] = np.append(psth[cell][type_string],
                                                              np.reshape(psth_cell_trial, [n_bins, 1]),
                                                                             axis = 1)
                if types[trial] == 1:
                    right_corr_trial_nos[movie] = np.append(right_corr_trial_nos[movie], trial_index)
                if types[trial] == 2:
                    left_corr_trial_nos[movie] = np.append(left_corr_trial_nos[movie], trial_index)
                n_usable_trials[movie] = trial_index + 1


        for cell in cells[movie]:

            psth_cell_left_corr = psth[cell]['left_corr'][:, 1:]
            spike_psth[cell_no, 0:n_bins] = np.mean(psth_cell_left_corr, axis = 1)*1000/bin_size_ms
            spike_count_trials_left[cell_no] = np.zeros([psth_cell_left_corr.shape[1], 4])
            for i in range(4):
                spike_count_trials_left[cell_no][:, i] = np.sum(psth_cell_left_corr[spike_count_bin_edges[i]:spike_count_bin_edges[i + 1], :], axis = 0)

            psth_cell_right_corr = psth[cell]['right_corr'][:, 1:]
            spike_psth[cell_no, n_bins:] = np.mean(psth_cell_right_corr, axis = 1)*1000/bin_size_ms
            spike_count_trials_right[cell_no] = np.zeros([psth_cell_right_corr.shape[1], 4])
            for i in range(4):
                spike_count_trials_right[cell_no][:, i] = np.sum(psth_cell_right_corr[spike_count_bin_edges[i]:spike_count_bin_edges[i + 1], :], axis = 0)

            # Comput psth from two random divisions of the data, to get a better idea of ordering of cells by latency
            if compute_random:
                n_left_trials = psth_cell_left_corr.shape[1]
                left_trials_random1 = np.random.choice(list(range(n_left_trials)), int(n_left_trials/2), replace = False).astype(int)
                left_trials_random2 = [trial for trial in list(range(n_left_trials)) if not trial in left_trials_random1]

                n_right_trials = psth_cell_right_corr.shape[1]
                right_trials_random1 = np.random.choice(list(range(n_right_trials)), int( n_right_trials/2), replace = False).astype(int)
                right_trials_random2 = [trial for trial in list(range(n_right_trials)) if not trial in right_trials_random1]

                assert(len(left_trials_random1) + len(left_trials_random2) == n_left_trials)
                assert(len(right_trials_random1) + len(right_trials_random2) == n_right_trials)

                spike_psth_random1[cell_no, 0:n_bins] = np.mean(psth_cell_left_corr[:, left_trials_random1], axis = 1)*1000/bin_size_ms
                spike_psth_random1[cell_no, n_bins:] = np.mean(psth_cell_right_corr[:, right_trials_random1], axis = 1)*1000/bin_size_ms
                spike_psth_random2[cell_no, 0:n_bins] = np.mean(psth_cell_left_corr[:, left_trials_random2], axis = 1)*1000/bin_size_ms
                spike_psth_random2[cell_no, n_bins:] = np.mean(psth_cell_right_corr[:, right_trials_random2], axis = 1)*1000/bin_size_ms

            sessions = list(spike_times_trials[movie][cell].keys())
            trials_cell = np.concatenate([list(spike_times_trials[movie][cell][session].keys()) for session in sessions], axis = 0)

            left_corr_trial_nos_cell = [trial for trial in left_corr_trial_nos[movie] if trial in trials_cell]
            n_left_trials_cells = np.append(n_left_trials_cells, len(left_corr_trial_nos_cell))

            right_corr_trial_nos_cell = [trial for trial in right_corr_trial_nos[movie] if trial in trials_cell]
            n_right_trials_cells = np.append(n_right_trials_cells, len(right_corr_trial_nos_cell))

            if make_plot_for_each_cell:
                plots.plot_single_cell_spike_psth(psth_cell_left_corr, psth_cell_right_corr,
                                                  left_corr_trial_nos_cell, right_corr_trial_nos_cell,
                                                  population_tvec[:n_bins],
                                                  go_cue_time[0], sample_end_time[0], sample_start_time[0],
                                                  axes = single_cell_psth_axes[cell_no])

            cell_no += 1

    output['psth_array'] = spike_psth
    output['spike_count_trials_left'] = spike_count_trials_left
    output['spike_count_trials_right'] = spike_count_trials_right
    if compute_random:
        output['psth_array_random_subset1'] = spike_psth_random1
        output['psth_array_random_subset2'] = spike_psth_random2
    output['tvec'] = population_tvec
    output['sample_start_bin_left'] = min_pre_sample_bins - 1 # Corresponding to last time point in last pre-sample bin
    output['sample_end_bin_left'] = min_pre_sample_bins + len(sample_bins[movies[0]]) - 1 # Corresponding to last time point in last sample bin
    output['go_cue_bin_left'] = min_pre_sample_bins + len(sample_bins[movies[0]]) + len(delay_bins[movies[0]]) - 1 # Corresponding to last time point in last delay bin
    output['trial_start_bin_right'] = n_bins - 1
    output['sample_start_bin_right'] = n_bins + min_pre_sample_bins - 1
    output['sample_end_bin_right'] = n_bins + min_pre_sample_bins + len(sample_bins[movies[0]]) - 1
    output['go_cue_bin_right'] = n_bins + min_pre_sample_bins + len(sample_bins[movies[0]]) + len(delay_bins[movies[0]]) - 1
    output['n_trials'] = n_usable_trials
    output['left_corr_trial_nos'] = left_corr_trial_nos
    output['right_corr_trial_nos'] = right_corr_trial_nos
    output['n_left_trials_cells'] = n_left_trials_cells
    output['n_right_trials_cells'] = n_right_trials_cells

    return output

def get_spike_times_trials(data_path, metadata_file, good_cells, good_blocks, go_cue_time, trial_types_left_right_cor_inc, overwrite = False):

    try:
        with open('{0}{1}spike_times_trials.pkl'.format(data_path, sep), 'rb') as f:
            spike_times_trials = pkl.load(f)
    except:
        overwrite = True

    if overwrite:
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
        sessions_to_process = metadata['sessions_to_process']
        batch_data = metadata['batch_data']

        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_times = output['frame_and_trial_times']['frame_times']
        frame_rate = output['frame_and_trial_times']['frame_rate'][sessions_to_process[0]]

        n_frames_per_trial = trial_tiff_stacks.get_n_frames_per_trial(data_path, metadata_file)

        volpy_results_file = metadata['volpy_results_file']
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
            volpy_results = pkl.load(f)

        roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)
        n_cells = roi_arrays[sessions_to_process[0]].shape[0]

        spike_times_trials = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}

        max_spike_time = 0
        min_last_spike_time = np.inf
        min_trial_frames = np.inf
        go_cue_frames = int(go_cue_time*frame_rate)

        for session in sessions_to_process:

            n_batches = batch_data[session]['n_batches']
            first_trials = np.array(batch_data[session]['first_trials']).astype(int)
            last_trials = np.array(batch_data[session]['last_trials']).astype(int)

            cum_frames_per_trial = np.cumsum(n_frames_per_trial[session]).astype(int)
            cum_frames_per_trial = np.insert(cum_frames_per_trial, 0, 0)

            types = trial_types_left_right_cor_inc[session]
            trial_no_session = 0

            for batch in range(n_batches):

                estimates = volpy_results[session][batch]['vpy']
                first_frame_batch = cum_frames_per_trial[first_trials[batch]]

                for trial in range(first_trials[batch], last_trials[batch]):

                    frames = list(range(cum_frames_per_trial[trial], cum_frames_per_trial[trial + 1])) # 0 is the first frame in the session
                    frames_batch = frames - first_frame_batch                                         # 0 is the first frame in the batch
                    frame_times_trial = frame_times[session][frames]
                    frame_times_trial = frame_times_trial - frame_times_trial[0]
                    if types[trial] in [1, 2]:
                        n_frames_trial = len(frame_times_trial)
                        if n_frames_trial < min_trial_frames:
                            min_trial_frames = n_frames_trial
                        if n_frames_trial < go_cue_frames:
                            print('Correct trial with less frames than pre_sample + sample + delay period')
                    trial_no_session += 1

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
                                    if max_time < min_last_spike_time:
                                        min_last_spike_time = max_time

        spike_times_trials['max_spike_time'] = max_spike_time
        spike_times_trials['min_last_spike_time'] = min_last_spike_time
        if min_trial_frames < 2400:
            min_trial_frames = 2400
        spike_times_trials['min_trial_time_s'] = min_trial_frames/frame_rate

        with open('{0}{1}spike_times_trials.pkl'.format(data_path, sep), 'wb') as f:
            pkl.dump(spike_times_trials, f)

    return spike_times_trials

def get_dFF_trials(data_path, metadata_file, good_cells, good_blocks, filter_freq = None, filter_type = 'lp', overwrite = False):

    try:
        with open('{0}{1}dFF_trials.pkl'.format(data_path, sep), 'rb') as f:
            dFF_trials = pkl.load(f)
    except:
        overwrite = True

    if overwrite:

        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
        sessions_to_process = metadata['sessions_to_process']
        batch_data = metadata['batch_data']

        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_rate = output['frame_and_trial_times']['frame_rate'][sessions_to_process[0]]
        frame_times = output['frame_and_trial_times']['frame_times']

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

        min_trial_frames = np.inf

        for session in sessions_to_process:

            n_batches = batch_data[session]['n_batches']
            first_trials = np.array(batch_data[session]['first_trials']).astype(int)
            last_trials = np.array(batch_data[session]['last_trials']).astype(int)

            cum_frames_per_trial = np.cumsum(n_frames_per_trial[session]).astype(int)
            cum_frames_per_trial = np.insert(cum_frames_per_trial, 0, 0)

            for batch in range(n_batches):

                estimates = volpy_results[session][batch]['vpy']

                dFF_all = estimates['dFF']
                if filter_freq == None:
                    dFF_filt = dFF_all
                else:
                    dFF_filt = {}
                    for cell in range(n_cells):
                        dFF_filt[cell] = filt_dFF(dFF_all[cell], filter_freq, frame_rate, filter_type = filter_type)

                first_frame_batch = cum_frames_per_trial[first_trials[batch]]

                for trial in range(first_trials[batch], last_trials[batch]):

                    frames_trial = list(range(cum_frames_per_trial[trial], cum_frames_per_trial[trial + 1])) # 0 is the first frame in the session
                    frames_batch = frames_trial - first_frame_batch

                    n_frames_trial = len(frames_trial)
                    if n_frames_trial < min_trial_frames:
                        min_trial_frames = n_frames_trial

                    dFF_trials['frames_per_trial'][session][trial] = len(frames_trial)                                      # 0 is the first frame in the batch

                    for cell in range(n_cells):
                        if cell in good_cells:
                            if batch > good_blocks[good_cells.index(cell)]:
                                continue
                            else:
                                dFF_cell = dFF_filt[cell]
                                dFF_trials[cell][session][trial] = dFF_cell[frames_batch]


        dFF_trials['frame_rate'] = frame_rate
        dFF_trials['min_trial_frames'] = min_trial_frames
        if min_trial_frames < 2400:
            min_trial_frames = 2400
        dFF_trials['min_trial_time_s'] = min_trial_frames/frame_rate

        with open('{0}{1}dFF_trials.pkl'.format(data_path, sep), 'wb') as f:
            pkl.dump(dFF_trials, f)

    return dFF_trials

def filt_dFF(trace, filter_freq, frame_rate, filter_type = 'lp', order = 3):

    if not filter_freq == None:
        sos = butter(order, filter_freq, filter_type, fs = frame_rate, output = 'sos')
        signal = sosfilt(sos, trace)
    else:
        signal = trace
    return signal

def get_population_tvec(movies, cells, data_paths, metadata_file, blocks, go_cue_time, sample_start_time, sample_end_time, trial_types_left_right_cor_inc, bin_size_ms, ):

    pre_sample_bins = {}
    sample_bins = {}
    delay_bins = {}
    response_bins = {}
    min_pre_sample_bins = np.inf
    min_response_bins = np.inf

    for movie in tqdm(movies):
        if len(cells[movie]) == 0:
            print('         Movie {0}: No cells pass SNR cutoff'.format(movie))
            continue
        trial_types_left_right_cor_inc = process_bpod_data.get_trial_types(data_paths[movie], metadata_file)
        spike_times_trials = get_spike_times_trials(data_paths[movie], metadata_file,
                                        list(cells[movie].astype(int)), blocks[movie], go_cue_time[movies.index(movie)], trial_types_left_right_cor_inc, overwrite = False)
        bin_edges_ms = np.arange(0, spike_times_trials['min_trial_time_s']*1000 + bin_size_ms, bin_size_ms)
        pre_sample_bins[movie] = np.where(bin_edges_ms < sample_start_time[movies.index(movie)]*1000)[0]
        sample_bins[movie] = len(pre_sample_bins[movie]) + np.where(bin_edges_ms[pre_sample_bins[movie][-1] + 1:] < sample_end_time[movies.index(movie)]*1000)[0]
        delay_bins[movie] = sample_bins[movie][-1] + 1 + np.where(bin_edges_ms[sample_bins[movie][-1] + 1:] < go_cue_time[movies.index(movie)]*1000)[0]
        response_bins[movie] = list(range(delay_bins[movie][-1] + 1, len(bin_edges_ms)))
        if len(pre_sample_bins[movie]) < min_pre_sample_bins:
            min_pre_sample_bins = len(pre_sample_bins[movie])
        if len(response_bins[movie]) < min_response_bins:
            min_response_bins = len(response_bins[movie])


    # All movies have same number of bins in sample period and delay period
    assert(np.var([len(v) for (k, v) in sample_bins.items()]) == 0)
    assert(np.var([len(v) for (k, v) in delay_bins.items()]) == 0)

    print('Minimum pre-sample period = {0}s'.format(min_pre_sample_bins*bin_size_ms/1000))
    print('Minimum response period = {0}s'.format(min_response_bins*bin_size_ms/1000))

    n_bins = min_pre_sample_bins + len(sample_bins[movies[0]]) + len(delay_bins[movies[0]]) + min_response_bins
    bin_edges_ms = np.arange(0, (n_bins + 1)*bin_size_ms, bin_size_ms)
    population_tvec = bin_edges_ms[1:]/1000

    return population_tvec
