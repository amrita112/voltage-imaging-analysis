from os.path import sep
import os
import pickle as pkl
from natsort import natsorted
#from skimage import io
from imread import imread_multi
from imread import imsave_multi
from tqdm import tqdm
import exifread
import numpy as np
import bisect

def main(data_path, metadata_file, daq_data, overwrite = False, n_trials_process = 0):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    # Check if tiff file metadata is already stored
    try:
        tiff_metadata = output['tiff_metadata']
        print('Tiff metadata loaded')
        frame_and_trial_times = output['frame_and_trial_times']
        print('Frame and trial times loaded')
    except:
        overwrite = True
        print('Could not load tiff metadata from {0}. Overwriting'.format(frame_times_file))

    if overwrite:

        trial_start_samples = daq_data['trial_start_samples']
        frame_samples = daq_data['frame_samples']
        led_trig = daq_data['led_trig']
        daq_sample_rate = metadata['daq_sample_rate']

        raw_tiff_image_path = metadata['raw_tiff_image_path']
        sessions_to_process = metadata['sessions_to_process']

        tiff_time_from_start_sec = {}
        bad_timestamps = {}
        n_tiff_files = {session: 0 for session in sessions_to_process}
        raw_tiff_files = {}
        trial_start_frames = {}
        trial_nos_imaged = {}
        n_frames_per_trial = {}
        frame_numbers = {}
        frame_times = {}
        frame_rate = {}
        frame_times_concat = []

        # Get mean pixel values and mean photon values
        mean_px_val_file = metadata['mean_px_val_file']
        try:
            with open('{0}{1}{2}'.format(data_path, sep, mean_px_val_file), 'rb') as f:
                mean_px_vals = pkl.load(f)
        except:
            print('Could not find mean pixel values. Rewriting')
            mean_px_vals = {session: {} for session in sessions_to_process}

        mean_photon_val_file = metadata['mean_photon_val_file']
        try:
            with open('{0}{1}{2}'.format(data_path, sep, mean_photon_val_file), 'rb') as f:
                mean_photon_vals = pkl.load(f)
        except:
            print('Could not find mean pixel values. Rewriting')
            mean_photon_vals = {session: {} for session in sessions_to_process}

        dark_pixel_value =  metadata['dark_pixel_value']
        photons_per_pixel = metadata['photons_per_pixel']
        trial_tiff_image_path = metadata['trial_tiff_image_path']

        for session in sessions_to_process:

            # Order tiff files in ascending order
            raw_tiff_files[session] = os.listdir(raw_tiff_image_path[session])
            raw_tiff_files[session] = ['{0}{1}{2}'.format(raw_tiff_image_path[session], sep, file) for file in raw_tiff_files[session] if file.endswith('.tif')]
            n_tiff_files[session] = len(raw_tiff_files[session])
            raw_tiff_files_ordered = natsorted(raw_tiff_files[session])

            n_trials_total = len(trial_start_samples[session])
            if n_trials_process == 0:
                n_trials_process = n_trials_total
            s0 = trial_start_samples[session][0]
            assert(np.sum(np.linalg.norm(frame_samples[session] - np.sort(frame_samples[session]))) == 0) # Frame samples are in ascending order
            fs = frame_samples[session]
            f0 = 0
            file_no = -1

            trial_nos_imaged[session] = []
            n_frames_per_trial[session] = []
            frame_numbers[session] = []
            trial_start_frames[session] = []

            n_samples = len(led_trig[session])
            tvec = np.linspace(0, n_samples/daq_sample_rate, n_samples)

            for trial in tqdm(range(n_trials_process)):

                f0 = bisect.bisect_left(fs, s0, lo = f0) # Index of first frame trigger after trial start trigger
                if trial < n_trials_total - 1:
                    s1 = trial_start_samples[session][trial + 1]
                else:
                    s1 = n_samples

                if f0 == len(fs):
                    continue

                if fs[f0] < s1: # Frame trigger occurs in trial, so trial was imaged

                    file_no += 1
                    f1 = bisect.bisect_left(fs, s1, lo = f0) # Index of last frame trigger before next trial start trigger

                    frame_numbers_trial = []
                    for frame_number in range(f0, f1):
                        if led_trig[session][fs[frame_number]] > 2:
                            frame_numbers_trial = np.append(frame_numbers_trial, frame_number)

                    if len(frame_numbers_trial) < 1: # LED trigger was not on in any frame in trial - i.e. trial is useless
                        s0 = s1
                        continue

                    n_triggers_trial = f1 - f0

                    im = imread_multi(raw_tiff_files_ordered[file_no])
                    n_frames = len(im)
                    if not n_triggers_trial == n_frames:
                        print('     Trial {0} will be skipped'.format(trial + 1))
                        print('         Tiff file {0}'.format(file_no))
                        print('         {0} frame triggers'.format(n_triggers_trial))
                        print('         {0} frames'.format(n_frames))
                        s0 = s1
                        continue

                    trial_nos_imaged[session] = np.append(trial_nos_imaged[session], trial)

                    frame_numbers_trial = frame_numbers_trial.astype(int)
                    n_frames_per_trial[session] = np.append(n_frames_per_trial[session], len(frame_numbers_trial))
                    trial_start_frames[session] = np.append(trial_start_frames[session], len(frame_numbers[session]))
                    frame_numbers[session] = np.append(frame_numbers[session], frame_numbers_trial)

                    pixel_array = np.array([im[frame] for frame in frame_numbers_trial - f0])
                    mean_px_vals[session][trial] = np.mean(np.mean(pixel_array, axis = 1), axis = 1)

                    # Deal with pixels that have values less than dark pixel value
                    sub_dark_pixels = np.where(pixel_array < dark_pixel_value)
                    pixel_array[sub_dark_pixels] = np.ones(len(sub_dark_pixels[0]))*dark_pixel_value

                    #photon_array = (pixel_array - dark_pixel_value)*photons_per_pixel
                    photon_array = pixel_array - dark_pixel_value
                    mean_photon_vals[session][trial] = np.mean(np.mean(photon_array, axis = 1), axis = 1)*photons_per_pixel

                    file_path_save = '{0}{1}Trial{2}.tif'.format(trial_tiff_image_path[session], sep, trial + 1)
                    if os.path.isfile(file_path_save):
                        print('     Trial {0} tiff stack exists'.format(trial + 1))
                    else:
                        imsave_multi(file_path_save, photon_array, opts = {'compress': False})

                s0 = s1

            tiff_time_from_start_sec[session] = np.zeros(n_tiff_files[session])
            bad_timestamps[session] = np.zeros(n_tiff_files[session])

            frame_numbers[session] = frame_numbers[session].astype(int)
            frame_times[session] = tvec[frame_samples[session][frame_numbers[session]]]

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

        # Save tiff metadata and frame and trial times
        tiff_metadata = {'n_tiff_files': n_tiff_files,
                         'raw_tiff_files': raw_tiff_files,
                         'tiff_time_from_start_sec': tiff_time_from_start_sec,
                         'bad_timestamps': bad_timestamps
                         }
        output['tiff_metadata'] = tiff_metadata

        frame_and_trial_times = {'frame_numbers': frame_numbers,
                                 'frame_times': frame_times,
                                 'trial_start_frames': trial_start_frames,
                                 'frame_rate': frame_rate,
                                 'frame_times_concat': frame_times_concat,

                                 # Specific to scanimage sessions
                                 'trial_nos_imaged': trial_nos_imaged,
                                 'n_frames_per_trial': n_frames_per_trial
        }

        output['frame_and_trial_times'] = frame_and_trial_times

        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'wb') as f:
            pkl.dump(output, f)

        # Save mean pixel values and mean photon values
        mean_px_val_file = metadata['mean_px_val_file']
        with open('{0}{1}{2}'.format(data_path, sep, mean_px_val_file), 'wb') as f:
            pkl.dump(mean_px_vals, f)

        mean_photon_val_file = metadata['mean_photon_val_file']
        with open('{0}{1}{2}'.format(data_path, sep, mean_photon_val_file), 'wb') as f:
            pkl.dump(mean_photon_vals, f)
