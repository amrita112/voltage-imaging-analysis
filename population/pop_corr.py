import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from os.path import sep
import pickle as pkl

from volpy import run_volpy

def spikes_vs_subth(population_data_path, data_paths, metadata_file,
                    plot_fov_traces = False, add_cell_nos_to_scatter = False):

    spike_corr = spikes(population_data_path, data_paths, metadata_file)

    subth_corr = subth(population_data_path, data_paths, metadata_file)

    plt.figure(constrained_layout = True)

    fov_idx = 1
    movies = list(data_paths.keys())
    for movie in movies:
        plt.scatter(spike_corr['corr'][movie], subth_corr['corr'][movie],
                        label = 'FOV {0}'.format(fov_idx), marker = '.')

        fov_idx += 1

    plt.xlabel('Spikes')
    plt.ylabel('Subthreshold')
    plt.title('Correlation to population mean')
    plt.legend()

    if plot_fov_traces:
        cmap_fr = cm.get_cmap('autumn')
        cmap_subth = cm.get_cmap('winter')

        for movie in movies:

            fr_cells = spike_corr['fr_cells'][movie]
            fr_movie = spike_corr['fr_movie'][movie]
            subth_cells = subth_corr['subth_cells'][movie]
            subth_movie = subth_corr['subth_movie'][movie]
            min_spike_corr = np.min(spike_corr['corr'][movie])
            max_spike_corr = np.max(spike_corr['corr'][movie])
            min_subth_corr = np.min(subth_corr['corr'][movie])
            max_subth_corr = np.max(subth_corr['corr'][movie])
            tvec_fr = spike_corr['tvec'][movie]
            tvec_subth = subth_corr['tvec'][movie]
            n_cells = fr_cells.shape[0]

            fig, ax = plt.subplots(nrows = n_cells + 1, ncols = 1,
                                    figsize = [20, 15], sharex = True, sharey = True)
            for cell in range(n_cells):
                val = (spike_corr['corr'][movie][cell] - min_spike_corr)/(max_spike_corr - min_spike_corr)
                ax[cell].plot(tvec_fr, fr_cells[cell, :],
                              color = cmap_fr(val))
                ax_subth = ax[cell].twinx()
                val = (subth_corr['corr'][movie][cell] - min_subth_corr)/(max_subth_corr - min_subth_corr)
                ax_subth.plot(tvec_subth, subth_cells[cell, :], color = cmap_subth(val))
                #ax[cell].set_ylabel('FR (Hz)')
                #ax_subth.set_ylabel('% dF/F')

            ax[-1].plot(tvec_fr, fr_movie, color = 'k')
            ax[-1].set_ylabel('Pop FR(Hz)')
            ax[-1].set_xlabel('Time (s)')
            ax_subth = ax[-1].twinx()
            ax_subth.tick_params(axis = 'y', labelcolor = 'gray')
            ax_subth.plot(tvec_subth, subth_movie, color = 'gray')
            ax_subth.set_ylabel('Pop subth \n(% dF/F)')

            fig.subplots_adjust(right = 0.8)
            cbar_fr_ax = fig.add_axes([0.85, 0.2, 0.02, 0.7])
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_fr, norm = colors.Normalize(vmin = min_spike_corr, vmax = max_spike_corr)),
                            cax = cbar_fr_ax, label = 'Firing rate correlation')
            cbar_subth_ax = fig.add_axes([0.93, 0.2, 0.02, 0.7])
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_subth, norm = colors.Normalize(vmin = min_subth_corr, vmax = max_subth_corr)),
                            cax = cbar_subth_ax, label = 'Subthreshold correlation')

            plt.savefig('{0}{1}Movie_{2}.png'.format(population_data_path, sep, movie))

            #fig.tight_layout()




def spikes(population_data_path, data_paths, metadata_file,
            bin_size_ms = 200, plot_spike_corr = True, overwrite = False):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        cells = qc_results['cells'][float(qc_results['snr_cutoff'])]
        blocks = qc_results['blocks'][float(qc_results['snr_cutoff'])]
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}spike_corr.py'.format(population_data_path, sep), 'rb') as f:
            spike_corr = pkl.load(f)
            corr = spike_corr['corr']
            fr_cells = spike_corr['fr_cells']
            fr_movie = spike_corr['fr_movie']

    except:
        overwrite = True

    if overwrite:

        corr = {movie: [] for movie in movies}
        fr_cells = {}
        fr_movie = {}
        tvec = {}

        cell_idx = 0
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

        spike_corr = {'corr': corr,
                      'fr_cells': fr_cells,
                      'fr_movie': fr_movie,
                      'tvec': tvec
                      }

        with open('{0}{1}spike_corr.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(spike_corr, f)

    plt.figure(constrained_layout = True)
    corr_all = np.concatenate(list(corr.values()))
    assert(len(corr_all) == total_cells)
    plt.hist(corr_all, 30, color = 'k')
    plt.xlabel('Pearson\'s correlation of cell firing rate to \naverage firing rate in field of view')
    plt.ylabel('# cells')

    return spike_corr

def subth(population_data_path, data_paths, metadata_file,
            overwrite = False):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        cells = qc_results['cells'][float(qc_results['snr_cutoff'])]
        blocks = qc_results['blocks'][float(qc_results['snr_cutoff'])]
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}subth_corr.py'.format(population_data_path, sep), 'rb') as f:
            subth_corr = pkl.load(f)
            corr = subth_corr['corr']
            subth_cells = subth_corr['subth_cells']
            subth_movie = subth_corr['subth_movie']

    except:
        overwrite = True

    if overwrite:

        corr = {movie: [] for movie in movies}
        subth_cells = {}
        subth_movie = {}
        tvec = {}

        cell_idx = 0
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
            tvec[movie] = tvec[movie][:n_good_frames]

            subth_cells[movie] = np.zeros([len(cells[movie]), n_good_frames])

            cell_idx  = 0
            for cell in cells[movie]:

                subth_cells[movie][cell_idx, :] = volpy_results['combined_data']['dFF_sub'][cell][:n_good_frames]
                cell_idx += 1

            subth_movie[movie] = np.mean(subth_cells[movie], axis = 0)
            assert(len(subth_movie[movie]) == n_good_frames)

            for cell_idx in range(len(cells[movie])):
                subth_movie_without_cell = (subth_movie[movie]*len(cells[movie]) - subth_cells[movie][cell_idx, :])/(len(cells[movie]) - 1)
                corr[movie] = np.append(corr[movie], np.corrcoef(subth_movie_without_cell, subth_cells[movie][cell_idx, :])[0, 1])

        subth_corr = {'corr': corr,
                      'subth_cells': subth_cells,
                      'subth_movie': subth_movie,
                      'tvec': tvec,
                      }

        with open('{0}{1}subth_corr.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(subth_corr, f)

    plt.figure(constrained_layout = True)
    corr_all = np.concatenate(list(corr.values()))
    assert(len(corr_all) == total_cells)
    plt.hist(corr_all, 30, color = 'k')
    plt.xlabel('Pearson\'s correlation of subthreshold membrane potential to \naverage subthreshold membrane potential in field of view')
    plt.ylabel('# cells')

    return subth_corr
