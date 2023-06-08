from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
from PIL import Image
import caiman as cm
from tqdm import tqdm

# Purpose: Check video for specific time points, view along with dF/F to get a sense of whether something is real activity or an artefact
def main(data_path, metadata_file, make_dff_fig = False, make_raw_movie = False, make_video = False,
         write_raw_movie_to_tiff = False, write_dff_movie_to_tiff = False):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    batch_data = metadata['batch_data']
    mmap_filenames = metadata['mmap_filenames']

    # Load frame times
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    tvec = output['frame_and_trial_times']['frame_times_concat']
    frame_rate = output['frame_and_trial_times']['frame_rate']
    frame_rate = np.mean(list(frame_rate.values()))

    # Load volpy results
    volpy_results_file = metadata['volpy_results_file']
    with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
            volpy_results = pkl.load(f)
    combined_data = volpy_results['combined_data']
    dFF = combined_data['dFF']
    dFF_sub = combined_data['dFF_sub']
    spike_times = combined_data['spike_times']
    spike_frames = combined_data['spike_frames']
    trial_start_frames_concat = combined_data['trial_start_frames_concat']

    # Get QC data
    good_cells = volpy_results['good_cells']
    cells = np.zeros(good_cells[sessions_to_process[0]].shape[0])
    for session in sessions_to_process:
        for batch in range(batch_data[session]['n_batches']):
            cells = cells + 1 - good_cells[session][:, batch]
    cells = np.where(cells == 0)[0] # Cells that are in good cells for all sessions, all batches
    n_cells = len(cells)

    if make_dff_fig:
        dff_fig(data_path, metadata_file, tvec, n_cells, cells, dFF, dFF_sub, spike_times, spike_frames, trial_start_frames_concat, sessions_to_process, batch_data,)

    if make_raw_movie:
        timepoints = get_timepoints()
        trials = get_trials(timepoints, data_path, metadata_file, tvec, trial_start_frames_concat)
        raw_movie = get_raw_movie(trials, data_path, metadata_file, sessions_to_process, batch_data,
                                    mmap_filenames, trial_start_frames_concat,
                                    write_raw_movie_to_tiff = write_raw_movie_to_tiff)
        dff_movie = get_dff_movie(raw_movie, data_path, metadata_file, trials, frame_rate,
                                  write_dff_movie_to_tiff = write_dff_movie_to_tiff)

    if make_video:
        traces_movie = get_traces_movie(timepoints, data_path, metadata_file)
        make_full_movie(timepoints, raw_movie, dff_movie, traces_movie)


def dff_fig(data_path, metadata_file, tvec, n_cells, cells, dFF, dFF_sub, spike_times, spike_frames, trial_start_frames_concat, sessions_to_process, batch_data, dff_scalebar_height = 0.1, scalebar_width = 1):

    # Plot dF/F
    n_frames_total = len(tvec)
    plt.figure(figsize = [n_frames_total/5000, n_cells*2])
    levels = np.zeros(n_cells)
    for idx in range(n_cells):
        cell = cells[idx]
        plt.plot(tvec, dFF[cell] + levels[idx], color = 'k', linewidth = 0.8, alpha = 1)
        plt.plot(tvec, dFF_sub[cell] + levels[idx], color = 'gray', linewidth = 2)
        plt.scatter(spike_times[cell], levels[idx] + dFF[cell][np.array(spike_frames[cell]).astype(int)], color = 'r', marker = '.')
        if idx < n_cells - 1:
            levels[idx + 1] = levels[idx] + np.max(dFF[cell]) - np.min(dFF[cells[idx + 1]])

    # Scalebar for dF/F
    [x0, x1] = plt.gca().get_xlim()
    dff_bottom = levels[0]
    dff_right = x1 - 10
    dff_left = dff_right - scalebar_width
    dff_scalebar = patches.Rectangle([dff_left, dff_bottom], scalebar_width, dff_scalebar_height, color = 'k',)
    plt.gca().add_patch(dff_scalebar)
    plt.text(dff_right + 1, dff_bottom + dff_scalebar_height/2, '-{0}% \ndF/F'.format(int(dff_scalebar_height*100)))

    # Plot trial starts
    [y0, y1] = plt.gca().get_ylim()
    for trial_start_frame in trial_start_frames_concat:
        trial_start_time = tvec[int(trial_start_frame)]
        plt.plot(np.ones(10)*trial_start_time, np.linspace(y0, y1, 10),
                    color = 'gray', linestyle = '--', linewidth = 0.8)

    # Plot batch starts
    for session in sessions_to_process:
        for batch in range(batch_data[session]['n_batches']):
            first_trial = int(batch_data[session]['first_trials'][batch])
            batch_start_time = tvec[int(trial_start_frames_concat[first_trial])]
            plt.plot(np.ones(10)*batch_start_time, np.linspace(y0, y1, 10),
                    color = 'k', linestyle = '--', linewidth = 2)

    plt.xlabel('Time (s)')
    plt.ylabel('Cell # ')
    plt.yticks(ticks = levels, labels = cells + 1)

def get_timepoints():

    t1 = input('Start time (s):')
    t1 = int(t1)
    t2 = input('End time (s):')
    t2 = int(t2)

    return [t1, t2]

def get_trials(timepoints, data_path, metadata_file, tvec, trial_start_frames_concat):

    f1 = np.where(tvec > timepoints[0])[0][0]
    f2 = np.where(tvec > timepoints[1])[0][0]
    print('Frame {0} to frame {1}'.format(f1, f2))

    trial1 = np.where(trial_start_frames_concat < f1)[0][-1]
    trial2 = np.where(trial_start_frames_concat > f2)[0][0]

    print('Making movie from trial {0} to trial {1}'.format(trial1, trial2))
    return list(range(trial1, trial2))

def get_raw_movie(trials, data_path, metadata_file, sessions_to_process, batch_data, mmap_filenames, trial_start_frames_concat, write_raw_movie_to_tiff = False):

    for session in sessions_to_process:
        for batch in range(batch_data[session]['n_batches']):
            if batch_data[session]['first_trials'][batch] < trials[0]:
                if batch_data[session]['last_trials'][batch] >= trials[0]:
                    session = session
                    batch = batch
                    break
    mmap_filename = mmap_filenames[session][batch]

    trial1_within_batch = trials[0] - batch_data[session]['first_trials'][batch]
    trial2_within_batch = trials[-1] - batch_data[session]['first_trials'][batch]
    if trial2_within_batch > batch_data[session]['last_trials'][batch]:
        trial2_within_batch = batch_data[session]['last_trials'][batch]
        trials = trials[:trial2_within_batch - trial1_within_batch]

    frames = np.array(range(trial_start_frames_concat[trials[0]], trial_start_frames_concat[trials[-1] + 1]))
    frames = frames - trial_start_frames_concat[int(batch_data[session]['first_trials'][batch])]
    assert(frames[-1] < batch_data[session]['n_frames_per_batch'])
    frames = frames.astype(int)
    n_frames = len(frames)

    i = mmap_filename.find('d2_')
    j = mmap_filename.find('_', i + 3)
    w = int(mmap_filename[i + 3:j])
    i = mmap_filename.find('d1_')
    j = mmap_filename.find('_', i + 3)
    h = int(mmap_filename[i + 3:j])

    print('Loading movie for batch {0}. This takes upto 5 minutes'.format(batch))
    try:
        imgs = cm.mmapping.load_memmap(mmap_filename)
    except FileNotFoundError:
        print('Mmap file name not correct')
        print(mmap_filename)
        dir = input('Correct directory letter:')
        mmap_filename = '{0}{1}'.format(dir, mmap_filename[1:])
        imgs = cm.mmapping.load_memmap(mmap_filename)
    batch_img_array = np.array(imgs[0])
    img_array = np.reshape(batch_img_array[:, frames], [w, h, n_frames])
    print('Raw movie loaded')

    if write_raw_movie_to_tiff:
        print('Writing raw movie to tiff')
        img_list = []
        file_path_save = '{0}{1}Registered_movies{1}Trial{2}-{3}.tif'.format(data_path, sep, trials[0], trials[-1])
        if not os.path.isdir('{0}{1}Registered_movies'.format(data_path, sep)):
            os.mkdir('{0}{1}Registered_movies'.format(data_path, sep))
        for frame in range(n_frames):

            img_frame = np.transpose(img_array[:, :, frame])
            img_list.append(Image.fromarray(img_frame))

        img_list[0].save(file_path_save, save_all = True, append_images = img_list[1:])

    return img_array

def get_dff_movie(raw_movie, data_path, metadata_file, trials, frame_rate, hp_freq_pb = 0.1, write_dff_movie_to_tiff = False):

    print('Calculating dFF movie')
    w = raw_movie.shape[0]
    h = raw_movie.shape[1]
    raw_movie = np.reshape(raw_movie, (raw_movie.shape[-1], -1))
    f0_movie = signal_filter(raw_movie.T, hp_freq_pb, frame_rate).T
    dF_movie = np.subtract(raw_movie, f0_movie)
    dFF_movie = np.divide(dF_movie, f0_movie)
    dFF_movie = np.reshape(dFF_movie, (w, h, -1))

    if write_dff_movie_to_tiff:
        print('Writing dFF movie to tiff')

        file_path_save = '{0}{1}Registered_movies{1}Trial{2}-{3}_dFF.tif'.format(data_path, sep, trials[0], trials[-1])
        if not os.path.isdir('{0}{1}Registered_movies'.format(data_path, sep)):
            os.mkdir('{0}{1}Registered_movies'.format(data_path, sep))

        img_list = []
        n_frames = dFF_movie.shape[-1]
        for frame in range(n_frames):

            img_frame = dFF_movie[:, :, frame]
            img_list.append(Image.fromarray(img_frame))

        img_list[0].save(file_path_save, save_all = True, append_images = img_list[1:])

    return dFF_movie

#def get_traces_movie(timepoines, data_path, metadata_file):



def signal_filter(sg, freq, fr, order=3, mode='low'):
    """
    Function for high/low passing the signal with butterworth filter

    Args:
        sg: 1-d array
            input signal

        freq: float
            cutoff frequency

        order: int
            order of the filter

        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering

    Returns:
        sg: 1-d array
            signal after filtering
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg
