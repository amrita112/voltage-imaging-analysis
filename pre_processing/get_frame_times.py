from os.path import sep
import pickle as pkl
import numpy as np
from pre_processing import check_tiff_timestamps

# Check daq data vs tiff metadata and get final frame times, trial start times and trial start frames (not including dark frames)
def get_frame_times(data_path, metadata_file, tiff_metadata, daq_data, overwrite = False):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)

    # Check if frame and trial times are already saved
    try:
        frame_and_trial_times = output['frame_and_trial_times']
        frame_numbers = frame_and_trial_times['frame_numbers']
        frame_times = frame_and_trial_times['frame_times']
        trial_start_frames = frame_and_trial_times['trial_start_frames']
        frame_rate = frame_and_trial_times['frame_rate']
        print('Frame and trial times loaded')
    except:
        print('Could not find frame and trial times, calculating')
        overwrite = True

    if overwrite:

        sessions_to_process = metadata['sessions_to_process'].copy()
        daq_sample_rate = metadata['daq_sample_rate']

        frame_numbers = {}
        trial_start_frames = {}
        frame_times = {}
        frame_times_concat = []
        frame_rate = {session: 0 for session in sessions_to_process}

        for session in sessions_to_process:
            print('Session {0}'.format(session))

            daq_frame_samples = daq_data['frame_samples'][session]
            n_daq_frames = len(daq_frame_samples)
            n_tiff_files = tiff_metadata['n_tiff_files'][session]
            bad_timestamps = tiff_metadata['bad_timestamps'][session]
            tiff_time_from_start_sec = tiff_metadata['tiff_time_from_start_sec'][session]
            trial_start_samples = daq_data['trial_start_samples'][session]
            n_trials = len(trial_start_samples)

            led_trig = daq_data['led_trig'][session]
            n_samples = len(led_trig)
            tvec = np.linspace(0, n_samples/daq_sample_rate, n_samples)
            frame_times[session] = tvec[daq_frame_samples]

            (daq_frame_samples, tiff_file_numbers) = check_tiff_timestamps.check_tiff_timestamps(n_tiff_files, n_daq_frames,
                                                                                                 bad_timestamps, tiff_time_from_start_sec,
                                                                                                 frame_times[session], daq_frame_samples,
                                                                                                 n_trials)
            if len(daq_frame_samples) == 0:
                metadata['sessions_to_process'].remove(session)
                with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'wb') as f:
                    pkl.dump(metadata, f)
                print('SESSION {0} WILL NOT BE PROCESSED \nCHANGE SESSIONS TO PROCESS IN METADATA INITIALIZATION CELL'.format(session))
                continue

            # Frame numbers
            frame_numbers[session] = []
            led_trig = daq_data['led_trig'][session]
            for sample_no in range(len(daq_frame_samples)):
                sample = daq_frame_samples[sample_no]
                if led_trig[sample]:
                    frame_numbers[session] = np.append(frame_numbers[session], tiff_file_numbers[sample_no])
            frame_numbers[session] = np.array(frame_numbers[session]).astype(int)
            selected_frame_samples = daq_frame_samples[frame_numbers[session] - frame_numbers[session][0]]

            # Frame times
            frame_times[session] = tvec[selected_frame_samples]

            # Frame rate
            ifi = np.diff(frame_times[session])
            med_ifi = np.min(ifi) + (np.max(ifi) - np.min(ifi))/200
            print('Median IFI: {0}ms'.format(int(1000*med_ifi)))
            mean_ifi = np.mean(ifi[ifi < med_ifi])
            frame_rate[session] = 1/mean_ifi

            # Concatenated frame times (excluding ITIs and LED off time)
            temp = frame_times[session].copy()
            large_ifis = np.where(ifi >= med_ifi)[0]
            for idx in large_ifis:
                temp[idx + 1:] = temp[idx + 1:] - temp[idx + 1] + temp[idx] + mean_ifi
            print('Mean IFI: {0}'.format(mean_ifi))
            print('Median IFI: {0}'.format(med_ifi))
            print('Max IFI after removing large IFIs: {0}'.format(np.max(np.diff(temp))))
            assert(np.max(np.diff(temp)) < 2*mean_ifi)
            frame_times_concat = np.append(frame_times_concat, temp - temp[0] + (0 if session == sessions_to_process[0] else frame_times_concat[-1] + mean_ifi))

            # Trial start frames
            trial_start_frames[session] = []
            for trial_no in range(n_trials):
                sample = trial_start_samples[trial_no]
                if trial_no < n_trials - 1:
                    next_trial_sample = trial_start_samples[trial_no + 1]
                else:
                    next_trial_sample = np.inf
                frames_after_sample = frame_numbers[session][np.logical_and(selected_frame_samples >= sample, selected_frame_samples < next_trial_sample)]
                if len(frames_after_sample) > 0:
                    trial_start_frame = frames_after_sample[0]
                    trial_start_frames[session] = np.append(trial_start_frames[session], trial_start_frame)
            trial_start_frames[session] = np.array(trial_start_frames[session]).astype(int)

            print('{0} frames and {1} trials with LED on'.format(len(frame_numbers[session]), len(trial_start_frames[session])))
            print('Estimated frame rate is {0} Hz'.format(frame_rate[session]))

        frame_and_trial_times = {'frame_numbers': frame_numbers,
                                 'frame_times': frame_times,
                                 'trial_start_frames': trial_start_frames,
                                 'frame_rate': frame_rate,
                                 'frame_times_concat': frame_times_concat
        }

        output['frame_and_trial_times'] = frame_and_trial_times
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'wb') as f:
            pkl.dump(output, f)


    return
