import numpy as np
import matplotlib.pyplot as plt
from os.path import sep
import pickle as pkl

from volpy import run_volpy

def burstiness(population_data_path, data_paths, metadata_file,
                plot_isi_dist = True, n_rows = 5, max_cells_per_fig = 50,
                max_isi_s = 0.025, n_bins = 10, burst_isi_ms = [5, 10, 20, 30],
                overwrite = False):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        cells = qc_results['cells'][float(qc_results['snr_cutoff'])]
        blocks = qc_results['blocks'][float(qc_results['snr_cutoff'])]
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}isi_dist.py'.format(population_data_path, sep), 'rb') as f:
            isi_dist = pkl.load(f)
            isi = isi_dist['isi']
            cv_isi = isi_dist['cv_isi']
            frac_isi_burst = isi_dist['frac_isi_burst']

    except:
        overwrite = True

    if overwrite:
        isi = {movie: {} for movie in movies}
        cv_isi = []
        frac_isi_burst = {b_isi: np.zeros(total_cells) for b_isi in burst_isi_ms}

        cell_idx = 0
        for movie in movies:

            print('Movie {0}'.format(movie))
            data_path = data_paths[movie]
            volpy_results = run_volpy.run_volpy(data_path, metadata_file)

            n_frames_per_block = []
            for session in list(volpy_results.keys()):
                if type(session) is int:
                    for block in list(volpy_results[session].keys()):
                        n_frames_per_block = np.append(n_frames_per_block,
                                                        len(volpy_results[session][block]['vpy']['dFF'][0]))
            cum_frames_per_block = np.cumsum(n_frames_per_block)

            blocks_movie = blocks[movie].astype(int)

            for cell in cells[movie]:

                cell = int(cell)
                last_good_frame = cum_frames_per_block[blocks_movie[np.where(cells[movie] == cell)[0][0]] - 1]

                spike_times = volpy_results['combined_data']['spike_times'][cell]
                spike_frames = volpy_results['combined_data']['spike_frames'][cell]
                spike_times = spike_times[spike_frames < last_good_frame]

                isis = np.diff(spike_times)*1000
                short_isis = isis[isis < max_isi_s*1000]
                isi[movie][cell] = short_isis
                cv_isi = np.append(cv_isi, np.std(isis)/np.mean(isis))
                for b_isi in burst_isi_ms:
                    frac_isi_burst[b_isi][cell_idx] = np.sum(isis <= b_isi)/len(isis)
                cell_idx += 1

        isi_dist = {'isi': isi,
                    'cv_isi': cv_isi,
                    'frac_isi_burst': frac_isi_burst}
        with open('{0}{1}isi_dist.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(isi_dist, f)

    if plot_isi_dist:

        n_cols = int(np.ceil(np.min([total_cells, max_cells_per_fig])/n_rows))
        n_figs = int(np.ceil(total_cells/max_cells_per_fig))
        figs = {}
        axs = {}
        for fig_no in range(n_figs):
            figs[fig_no], axs[fig_no] = plt.subplots(nrows = n_rows, ncols = n_cols,
                                                     sharex = True,
                                                    constrained_layout = True, figsize = [20, 10])
        cell_idx = 0
        for movie in movies:
            for cell in cells[movie]:
                fig_no = int(np.floor(cell_idx/max_cells_per_fig))
                row = int(np.floor((cell_idx - fig_no*max_cells_per_fig)/n_cols))
                col = int(np.mod((cell_idx - fig_no*max_cells_per_fig), n_cols))
                axs[fig_no][row, col].hist(isi[movie][cell], n_bins, color = 'k')
                axs[fig_no][row, col].set_title('Movie {0} Cell {1}'.format(movie, cell))
                if row == n_rows - 1:
                    axs[fig_no][row, col].set_xlabel('ISI (ms)')
                if col == 0:
                    axs[fig_no][row, col].set_ylabel('# occurences')
            cell_idx += 1

    plt.figure()
    plt.hist(cv_isi, 50, color = 'k')
    plt.xlabel('CV of ISI')
    plt.ylabel('# cells')
    plt.savefig('{0}{1}CV_ISI.png'.format(population_data_path, sep))

    plt.figure()
    for b_isi in burst_isi_ms:
        plt.hist(frac_isi_burst[b_isi], 50, label = '{0} ms'.format(b_isi), alpha = 0.5)
    plt.xlabel('Fraction of ISIs <= X ms')
    plt.yscale('log')
    plt.ylabel('# cells')
    plt.legend()

    return isi_dist
