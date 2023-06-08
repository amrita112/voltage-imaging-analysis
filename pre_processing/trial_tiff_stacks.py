import os
from os.path import sep
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from natsort import natsorted
from imread import imread_multi
import time
from PIL import Image
from tqdm import tqdm

def get_n_frames_per_trial(data_path, metadata_file, overwrite = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    try:
        n_frames_per_trial = output['frame_and_trial_times']['n_frames_per_trial']
    except:
        print('Could not load number of frames per trial')
        overwrite = True
        n_frames_per_trial = {}

    if overwrite:
        print('overwriting number of frames per trial')
        trial_tiff_image_path = metadata['trial_tiff_image_path']
        sessions_to_process = metadata['sessions_to_process']
        for session in sessions_to_process:

            print('Session {0}'.format(session))
            fnames = os.listdir(trial_tiff_image_path[session])
            fnames = natsorted(fnames)
            fnames = ['{0}{1}{2}'.format(trial_tiff_image_path[session], sep, fname) for fname in fnames if fname.endswith('.tif')]

            n_frames_per_trial[session] = np.zeros(len(fnames))
            for trial in tqdm(range(len(fnames))):
                try:
                    im = Image.open(fnames[trial])
                    n_frames_per_trial[session][trial] = im.n_frames
                except:
                    im = imread_multi(fnames[trial])
                    n_frames_per_trial[session][trial] = len(im)

        output['frame_and_trial_times']['n_frames_per_trial'] = n_frames_per_trial
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'wb') as f:
            pkl.dump(output, f)

    return n_frames_per_trial

def trial_tiff_stacks(data_path, metadata_file, overwrite_all = False):

    print(' ')
    print('Writing tiff stacks')
    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']

    # Get tiff metadata, frame numbers and trial start frames
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_numbers = output['frame_and_trial_times']['frame_numbers']
    trial_start_frames = output['frame_and_trial_times']['trial_start_frames']
    raw_tiff_files = output['tiff_metadata']['raw_tiff_files']
    raw_tiff_image_path = metadata['raw_tiff_image_path']

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

    n_frames_per_trial = {}

    # For each trial, load tiff files, convert pixel values to photoelectrons, remove dark frames and write tiff stack
    for session in sessions_to_process:

        if not os.path.isdir(trial_tiff_image_path[session]):
            os.mkdir(trial_tiff_image_path[session])

        n_trials = len(trial_start_frames[session])
        print('Session {0}: {1} trials'.format(session, n_trials))
        raw_tiff_files_ordered = natsorted(raw_tiff_files[session])
        n_frames_per_trial[session] = np.zeros(n_trials)

        t0 = time.time()

        for trial in range(n_trials):

            file_path_save = '{0}{1}Trial_{2}.tif'.format(trial_tiff_image_path[session], sep, trial + 1)
            if os.path.isfile(file_path_save):
                print('Trial {0}: {1} seconds. Tiff stack exists'.format(trial + 1, np.round(time.time() - t0)))
                im = Image.open(file_path_save)
                n_frames_per_trial[session][trial] = im.n_frames
                overwrite_trial = False
            else:
                overwrite_trial = True

            if np.logical_or(overwrite_trial, overwrite_all):
                img_list = []
                if trial == n_trials - 1:
                    frames = frame_numbers[session][frame_numbers[session] >= trial_start_frames[session][trial]]
                else:
                    frames = frame_numbers[session][np.logical_and(frame_numbers[session] >= trial_start_frames[session][trial],
                                                                    frame_numbers[session] < trial_start_frames[session][trial + 1])]
                n_frames = len(frames)
                mean_px_vals[session][trial] = np.zeros(n_frames)
                mean_photon_vals[session][trial] = np.zeros(n_frames)
                n_frames_per_trial[session][trial] = n_frames
                frame_id = 0
                for frame in frames:

                    file_path = '{0}{1}{2}'.format(raw_tiff_image_path[session], sep, raw_tiff_files_ordered[frame])
                    img = Image.open(file_path, mode = 'r')
                    img_array = np.array(img)
                    mean_px_vals[session][trial][frame_id] = np.mean(img_array)

                    # Deal with pixels that have values less than dark pixel value
                    sub_dark_pixels = np.where(img_array < dark_pixel_value)
                    img_array[sub_dark_pixels] = np.ones(len(sub_dark_pixels[0]))*dark_pixel_value

                    photon_img_array = (img_array - dark_pixel_value)*photons_per_pixel
                    mean_photon_vals[session][trial][frame_id] = np.mean(photon_img_array)

                    img_list.append(Image.fromarray(photon_img_array))
                    frame_id += 1


                img_list[0].save(file_path_save, save_all = True, append_images = img_list[1:])

                print('Trial {0}: {2} frames; {1} seconds'.format(trial + 1, np.round(time.time() - t0), n_frames))


    # Save mean pixel values and mean photon values
    mean_px_val_file = metadata['mean_px_val_file']
    with open('{0}{1}{2}'.format(data_path, sep, mean_px_val_file), 'wb') as f:
        pkl.dump(mean_px_vals, f)

    mean_photon_val_file = metadata['mean_photon_val_file']
    with open('{0}{1}{2}'.format(data_path, sep, mean_photon_val_file), 'wb') as f:
        pkl.dump(mean_photon_vals, f)
