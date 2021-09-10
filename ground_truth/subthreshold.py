from scipy import signal

from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltpath
import time
import json

from ground_truth import ephys_vs_imaging
from ground_truth import volpy

def power_spectra(data_path, sub_ids, cells, movies, cell_folders, make_plots = False, n_rows = 3):

    ps = {}
    ephys_data = ephys_vs_imaging.get_ephys_data(data_path, sub_ids, cells, movies, cell_folders)
    volpy_results = volpy.main(data_path, sub_ids, cells, cell_folders, movies)
    for sid in sub_ids:
        try:
            with open('{0}{1}ANM{2}_power_spectra.pkl'.format(data_path, sep, sid), 'rb') as f:
                ps[sid] = pkl.load(f)
            print('ANM {0} power spectra loaded'.format(sid))
        except:
            print('ANM {0} power spectra could not be loaded'.format(sid))
            ps[sid] = {cell: {} for cell in cells[sid]}
            for cell in cells[sid]:
                print(' Cell {0}: {1} movies'.format(cell, len(movies[sid][cell])))
                for movie in movies[sid][cell]:
                    print('     Movie {0}'.format(movie))

                    frame_times = np.load('{0}{1}{2}{1}{3}{1}frame_times.npy'.format(data_path, sep, cell_folders[sid][cell], movie))
                    frame_rate = 1/np.mean(np.diff(frame_times))
                    dff = np.reshape(volpy_results[sid][cell][movie]['dFF'], [-1])

                    ephys_times = ephys_data[sid]['timings'][cell][movie]
                    ephys_sampling_rate = 1/min(np.diff(ephys_times)) # Ephys is not necessarily continuous, so take minimum instead of average
                    ephys_trace = ephys_data[sid]['traces'][cell][movie]

                    ps[sid][cell][movie] = {'im': signal.welch(dff, frame_rate), 'ephys': signal.welch(ephys_trace, ephys_sampling_rate, nperseg = 10000)}
            with open('{0}{1}ANM{2}_power_spectra.pkl'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(ps[sid], f)

    if make_plots:
        n_cells = sum([len(cells[sid]) for sid in sub_ids])
        n_cols = int(np.ceil(n_cells/n_rows))
        fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, constrained_layout = True, figsize = [n_cols*2, n_rows*2])
        cell_no = 0
        for sid in sub_ids:
            for cell in cells[sid]:

                row = int(np.floor(cell_no/n_cols))
                col = int(np.mod(cell_no, n_cols))
                ax_plot = ax[row, col]

                for movie in movies[sid][cell]:
                    im_psd = np.reshape(ps[sid][cell][movie]['im'][1]/max(ps[sid][cell][movie]['im'][1]), [-1])
                    ephys_psd = np.reshape(ps[sid][cell][movie]['ephys'][1]/max(ps[sid][cell][movie]['ephys'][1]), [-1])
                    im_line = ax_plot.plot(ps[sid][cell][movie]['im'][0], im_psd, color = 'b', alpha = 0.6,)
                    ephys_line = ax_plot.plot(ps[sid][cell][movie]['ephys'][0], ephys_psd, color = 'r', alpha = 0.6, )

                ax_plot.set_xlim([0, 100])
                #ax_plot.legend([im_line, ephys_line], ['Imaging', 'Ephys'])
                ax_plot.set_ylabel('Power')
                ax_plot.set_xlabel('Frequency (Hz)')
                cell_no += 1
        fig.savefig('{0}{1}Ephys_im_spectra.png'.format(data_path, sep))
