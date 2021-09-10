from os.path import sep
import numpy as np
import pickle as pkl

def get_trial_types(data_path, metadata_file):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']

    bpod_data_file = metadata['bpod_data_file']
    with open('{0}{1}{2}'.format(data_path, sep, bpod_data_file), 'rb') as f:
        bpod_data = pkl.load(f)

    left_right = bpod_data['left_right']
    cor_inc = bpod_data['cor_inc']
    early_lick_sample = bpod_data['early_lick_sample']
    early_lick_delay = bpod_data['early_lick_delay']

    trial_types_left_right_cor_inc = {}

    for session in sessions_to_process:
        n_trials = len(left_right[session])
        trial_types_left_right_cor_inc[session] = np.zeros(n_trials)

        for trial in range(n_trials):
            if not early_lick_sample[session][trial]:
                if not early_lick_delay[session][trial]:
                    if cor_inc[session][trial]:
                        if left_right[session][trial]:
                            trial_types_left_right_cor_inc[session][trial] = 1 # Right correct
                        else:
                            trial_types_left_right_cor_inc[session][trial] = 2 # Left correct
                    else:
                        if left_right[session][trial]:
                            trial_types_left_right_cor_inc[session][trial] = 3 # Right incorrect
                        else:
                            trial_types_left_right_cor_inc[session][trial] = 4 # Left incorrect

    return trial_types_left_right_cor_inc

def get_go_cue_time(data_path, metadata_file, var_thresh_s = 0.05):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']

    bpod_data_file = metadata['bpod_data_file']
    with open('{0}{1}{2}'.format(data_path, sep, bpod_data_file), 'rb') as f:
        bpod_data = pkl.load(f)

    go_cue_time = bpod_data['go_cue_start_time']
    go_cue_time_all = []

    trial_types = get_trial_types(data_path, metadata_file)

    for session in sessions_to_process:
        n_trials = len(trial_types[session])
        for trial in range(n_trials):
            if trial_types[session][trial] > 0:
                go_cue_time_all = np.append(go_cue_time_all, go_cue_time[session][trial])

    assert(np.var(go_cue_time_all) < var_thresh_s)

    return np.median(go_cue_time_all)


def get_sample_end_time(data_path, metadata_file, var_thresh_s = 0.05):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']

    bpod_data_file = metadata['bpod_data_file']
    with open('{0}{1}{2}'.format(data_path, sep, bpod_data_file), 'rb') as f:
        bpod_data = pkl.load(f)

    sample_end_time = bpod_data['sample_end_time']
    sample_end_time_all = []

    trial_types = get_trial_types(data_path, metadata_file)

    for session in sessions_to_process:
        n_trials = len(trial_types[session])
        for trial in range(n_trials):
            if trial_types[session][trial] > 0:
                sample_end_time_all = np.append(sample_end_time_all, sample_end_time[session][trial])

    assert(np.var(sample_end_time_all) < var_thresh_s)

    return np.median(sample_end_time_all)


def get_sample_start_time(data_path, metadata_file, var_thresh_s = 0.05):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']

    bpod_data_file = metadata['bpod_data_file']
    with open('{0}{1}{2}'.format(data_path, sep, bpod_data_file), 'rb') as f:
        bpod_data = pkl.load(f)

    sample_start_time = bpod_data['sample_start_time']
    sample_start_time_all = []

    trial_types = get_trial_types(data_path, metadata_file)

    for session in sessions_to_process:
        n_trials = len(trial_types[session])
        for trial in range(n_trials):
            if trial_types[session][trial] > 0:
                sample_start_time_all = np.append(sample_start_time_all, sample_start_time[session][trial])

    assert(np.var(sample_start_time_all) < var_thresh_s)

    return np.median(sample_start_time_all)
