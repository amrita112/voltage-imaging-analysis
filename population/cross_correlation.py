import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from os.path import sep
import pickle as pkl

from volpy import run_volpy

def cross_correlation(population_data_path, data_paths, metadata_file,
           bin_size_ms = [10, 20, 50, 100, 200, 400], overwrite = False):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        cells = qc_results['cells'][float(qc_results['snr_cutoff'])]
        blocks = qc_results['blocks'][float(qc_results['snr_cutoff'])]
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}cross_corr.py'.format(population_data_path, sep), 'rb') as f:
            cross_corr = pkl.load(f)
            corr = cross_corr['corr']
    except:
        overwrite = True

    if overwrite:

        corr = {movie: [] for movie in movies}
        tvec = {}

        cell_pair_idx = 0
        for movie in movies:
            print('Movie {0}'.format(movie))
            data_path = data_paths[movie]
            volpy_results = run_volpy.run_volpy(data_path, metadata_file)
            tvec[movie] = volpy_results['combined_data']['tvec']

            n_frames_per_block = []
            for session in list(volpy_results.keys()):
                if type(session) is int:
                    for block in list(volpy_results[session].keys()):
                        n_frames_per_block = np.append(n_frames_per_block,
                                                        len(volpy_results[session][block]['vpy']['dFF'][0]))
            cum_frames_per_block = np.cumsum(n_frames_per_block)

            blocks_movie = blocks[movie].astype(int)
            n_good_frames = int(cum_frames_per_block[np.min(blocks_movie)])

            bin_edges_ms = np.arange(0, tvec[movie][n_good_frames - 1]*1000 + 3*bin_size_ms, bin_size_ms)
            n_bins = len(bin_edges_ms) - 1
            tvec[movie] = (bin_edges_ms[1:] + bin_edges_ms[:-1])/2000

            fr_cells[movie] = np.zeros([len(cells[movie]), n_bins])

            cell_idx  = 0
            for cell in cells[movie]:

                spike_times = volpy_results['combined_data']['spike_times'][cell]
                spike_frames = volpy_results['combined_data']['spike_frames'][cell]
                spike_times = spike_times[spike_frames < n_good_frames]
                fr_cells[movie][cell_idx, :] = np.histogram(spike_times*1000, bin_edges_ms)[0]
                cell_idx += 1

            fr_movie[movie] = np.mean(fr_cells[movie], axis = 0)
            assert(len(fr_movie[movie]) == n_bins)

            for cell_idx in range(len(cells[movie])):
                fr_movie_without_cell = (fr_movie[movie]*len(cells[movie]) - fr_cells[movie][cell_idx, :])/(len(cells[movie]) - 1)
                corr[movie] = np.append(corr[movie], np.corrcoef(fr_movie_without_cell, fr_cells[movie][cell_idx, :])[0, 1])

        cross_corr = {'corr': corr,
                      'fr_cells': fr_cells,
                      'fr_movie': fr_movie,
                      'tvec': tvec
                      }

        with open('{0}{1}cross_corr.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(cross_corr, f)

    plt.figure(constrained_layout = True)
    corr_all = np.concatenate(list(corr.values()))
    assert(len(corr_all) == total_cells)
    plt.hist(corr_all, 30, color = 'k')
    plt.xlabel('Pearson\'s correlation of cell firing rate to \naverage firing rate in field of view')
    plt.ylabel('# cells')

    return cross_corr
