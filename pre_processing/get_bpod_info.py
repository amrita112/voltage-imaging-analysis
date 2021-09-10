from os.path import sep
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py

def get_bpod_info(data_path, metadata_file, overwrite = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    bpod_data_file = metadata['bpod_data_file']
    try:
        with open('{0}{1}{2}'.format(data_path, sep, bpod_data_file), 'rb') as f:
            bpod_data = pkl.load(f)
        print('Bpod data loaded')
    except:
        print('Could not load Bpod data. Overwriting')
        overwrite = True
        bpod_data = {}

    if overwrite:

        bpod_trial_numbers = metadata['bpod_trial_numbers']
        left_right = {}
        cor_inc = {}
        early_lick_sample = {}
        early_lick_delay = {}
        sample_start_time = {}
        sample_end_time = {}
        go_cue_start_time = {}
        go_cue_end_time = {}

        sessions_to_process = metadata['sessions_to_process']

        # Get trial start frames
        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        trial_start_frames = output['frame_and_trial_times']['trial_start_frames']

        # Load Bpod data
        bpod_data_mat_file = metadata['bpod_data_mat_file']
        try:
            bpod_data = sio.loadmat('{0}{1}{2}'.format(data_path, sep, bpod_data_mat_file))
        except:
            print('Please run the following commands in Matlab: \nbpod_file_path = \'{0}\'\nbpod_data_save_path = \'{1}{2}{3}\'\nget_bpod_info(bpod_file_path, bpod_data_save_path)'.format(metadata['bpod_file_path'],
            data_path, sep, metadata['bpod_data_mat_file']))
            print('Then re-run pre-processing')
            return

        trial_start_times = bpod_data['bpod_data_struct']['trial_start_times'][0][0]
        left_right_all = bpod_data['bpod_data_struct']['left_right'][0][0][0]
        cor_inc_all = bpod_data['bpod_data_struct']['cor_inc'][0][0]
        early_lick_sample_all = bpod_data['bpod_data_struct']['early_lick_sample'][0][0]
        early_lick_delay_all = bpod_data['bpod_data_struct']['early_lick_delay'][0][0]
        sample_start_time_all = bpod_data['bpod_data_struct']['sample_start'][0][0]
        sample_end_time_all = bpod_data['bpod_data_struct']['sample_end'][0][0]
        go_cue_start_time_all = bpod_data['bpod_data_struct']['go_cue_start'][0][0]
        go_cue_end_time_all = bpod_data['bpod_data_struct']['go_cue_end'][0][0]

        for session in sessions_to_process:

            # Get trial numbers for session
            n_trials_session = len(trial_start_frames[session])
            if not len(bpod_trial_numbers[session]) == 2:
                print('Bpod trial numbers not entered. Use trial times to select trial numbers for session')
                print('If figure does not show, restart the kernel and do %matplotlib qt')
                bpod_trial_numbers[session] = select_trial_numbers(trial_start_times, n_trials_session, session)

            if not (n_trials_session == bpod_trial_numbers[session][1] - bpod_trial_numbers[session][0]):
                print('Bpod trial numbers in session {0} not correct. Check DAQ data figure and re-enter trial numbers.'.format(session))
                print('{0} trial start frames'.format(n_trials_session))
                print('{0} trials according to bpod trial numbers'.format(bpod_trial_numbers[session][1] - bpod_trial_numbers[session][0]))
                return

            # Get trial types - left/right, correct/incorrect, early lick sample, early lick delay
            left_right[session] = left_right_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            cor_inc[session] = cor_inc_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            early_lick_sample[session] = early_lick_sample_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            early_lick_delay[session] = early_lick_delay_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            sample_start_time[session] = sample_start_time_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            sample_end_time[session] = sample_end_time_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            go_cue_start_time[session] = go_cue_start_time_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]
            go_cue_end_time[session] = go_cue_end_time_all[bpod_trial_numbers[session][0]:bpod_trial_numbers[session][1]]

        bpod_data['bpod_trial_numbers'] = bpod_trial_numbers
        bpod_data['left_right'] = left_right
        bpod_data['cor_inc'] = cor_inc
        bpod_data['early_lick_sample'] = early_lick_sample
        bpod_data['early_lick_delay'] = early_lick_delay
        bpod_data['sample_start_time'] = sample_start_time
        bpod_data['sample_end_time'] = sample_end_time
        bpod_data['go_cue_start_time'] = go_cue_start_time
        bpod_data['go_cue_end_time'] = go_cue_end_time

        with open('{0}{1}{2}'.format(data_path, sep, bpod_data_file), 'wb') as f:
            pkl.dump(bpod_data, f)

    return bpod_data

def select_trial_numbers(trial_start_times, n_trials_session, session):

    plt.figure()
    plt.plot(np.squeeze(trial_start_times))
    plt.xlabel('Trial #')
    plt.ylabel('Trial start time')
    plt.title('Choose first and last trials for session. Total trials = {0}'.format(n_trials_session))
    plt.grid()
    plt.pause(1)

    t1 = input('Session {0}: Enter first trial number'.format(session))
    t1 = int(t1)
    t2 = input('Session {0}: Enter last trial number'.format(session))
    t2 = int(t2)

    return [t1, t2]
