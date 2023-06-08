from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import csd

def correlation_spectrum(data_path, ephys_data, volpy_results, sub_ids, cells, movies, sd_accuracy, make_plots = False, overwrite = False):

    corr_spctrm = get_corr_spctrm(data_path, ephys_data, volpy_results, sub_ids, cells, movies, sd_accuracy, overwrite = overwrite)

    if make_plots:
        plot_corr_spctrm(corr_spctrm)

    return corr_spctrm

def get_corr_spctrm(data_path, ephys_data, volpy_results, sub_ids, cells, movies, sd_accuracy, overwrite = False):

    try:
        with open('{0}{2}csd_data.pkl'.format(data_path, sep), 'rb') as f:
            csd_data = pkl.load(f)
    except:
        overwrite = True

    if overwrite:
        csd_data = {sid: {} for sid in sub_ids}
        for sid in sub_ids:
            for cell in cells[sid]:
                csd_data[sid][cell] = {}
                movie_idx = 0
                for movie in movies[sid][cell]:
                    movie_idx += 1
                    if not movie in list(sd_accuracy[sid][cell].keys()):
                        continue
                    if sd_accuracy[sid][cell][movie]['true_pos'] == None:
                        continue

                    dff = volpy_results[sid][cell][movie]['dFF']
                    trace = ephys_data[sid]['traces'][cell][movie]

                    im_frame_times = np.load('{0}{1}{2}{1}{3}{1}frame_times.npy'.format(data_path, sep, cell_folders[sid][cell], movie))
                    dff_fs = 1/np.mean(np.diff(im_frame_times))

                    ephys_sample_times = ephys_data[sid]['timings'][cell][movie]
                    ephys_fs = 1/min(np.diff(ephys_sample_times)) # Ephys is not necessarily continuous, so take minimum instead of average

                    #ephys_trace_down_sampled =

                    csd_data[sid][cell][movie] = csd(dff, ephys_trace_down_sampled, fs = dff_fs)

        with open('{0}{2}csd_data.pkl'.format(data_path, sep), 'wb') as f:
            pkl.dump(csd_data, f)

    return csd_data
