
from os.path import sep
import pickle as pkl

def get_metadata(data_path, metadata_file,
                 scanimage, frame_rate_Hz, n_sessions, sessions_to_process,
                 raw_tiff_image_path, trial_tiff_image_path,
                 daq_file_paths, bpod_file_path, bpod_trial_numbers,
                 vcam_trig_in_channel, vcam_trig_out_channel,
                 trial_start_trig_channel, led_trig_channel,
                 daq_sample_rate = 20000, # NI DAQ records triggers at this speed
                 min_iti_sec = 0.4, # Inter trial interval should not be less than this value
                 um_per_px = 2.32, # If 20x objective and 0.55x camera tube are used
                 h = None, w = None, # May not be specified when creating the metadata file, but will be specified later
                 dark_pixel_value = 1600, photons_per_pixel = 0.5, # Properties of camera
                 led_lag_ms = 10, # Lag between Bpod signal to LED and LED response

                ):

    try:
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)
    except:
        a = input('Metadata not found. Create new file with defaults? y/n')
        if a == 'y':
            metadata = {}
        else:
            print('Stopping')
            return

    metadata['scanimage'] =                    scanimage
    metadata['frame_rate_Hz'] =                frame_rate_Hz
    metadata['n_sessions'] =                   n_sessions
    metadata['sessions_to_process'] =          sessions_to_process
    metadata['raw_tiff_image_path'] =          raw_tiff_image_path
    metadata['trial_tiff_image_path'] =        trial_tiff_image_path
    metadata['daq_file_paths'] =               daq_file_paths
    metadata['bpod_file_path'] =               bpod_file_path
    metadata['bpod_trial_numbers'] =           bpod_trial_numbers
    metadata['daq_sample_rate'] =              daq_sample_rate
    metadata['vcam_trig_in_channel'] =         vcam_trig_in_channel
    metadata['vcam_trig_out_channel'] =        vcam_trig_out_channel
    metadata['trial_start_trig_channel'] =     trial_start_trig_channel
    metadata['led_trig_channel'] =             led_trig_channel
    metadata['min_iti_sec'] =                  min_iti_sec
    metadata['um_per_px'] =                    um_per_px
    metadata['dark_pixel_value'] =             dark_pixel_value
    metadata['photons_per_pixel'] =            photons_per_pixel
    metadata['led_lag_samples'] =              int(led_lag_ms*daq_sample_rate/1000)

    frame_times_file = 'frame_times.pkl'
    metadata['frame_times_file'] = frame_times_file
    try:
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
    except:
        print('Overwriting {0}'.format(frame_times_file))
        output = {}
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'wb') as f:
            pkl.dump(output, f)

    metadata['daq_data_file'] =                    'daq_data.pkl'
    metadata['tiff_timestamps_file'] =             'tiff_timestamps.pkl'
    metadata['dark_frames_file'] =                 'dark_frames.pkl'
    metadata['roi_file'] =                         'rois.pkl'
    metadata['roi_array_file'] =                   'roi_array.pkl'
    metadata['mean_px_val_file'] =                 'mean_px_vals.pkl'
    metadata['mean_photon_val_file'] =             'mean_photon_vals.pkl'
    metadata['bpod_data_file'] =                   'bpod_data.pkl'
    metadata['bpod_data_mat_file'] =               'bpod_data.mat'
    metadata['volpy_results_file'] =               'volpy_results.pkl'
    metadata['usable_trials_file'] =               'usable_trials.pkl'
    metadata['go_resp_file'] =                     'go_resp.pkl'
    metadata['isi_data_file'] =                    'isi_data.pkl'
    metadata['plots_path'] =                       '{0}{1}Plots'.format(data_path, sep)

    if not h == None:
        metadata['h'] = h
    if not w == None:
        metadata['w'] = w

    try:
        vpy_results_files = metadata['vpy_results_files']
    except KeyError:
        metadata['vpy_results_files'] = []

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'wb') as f:
        pkl.dump(metadata, f)

    return metadata
