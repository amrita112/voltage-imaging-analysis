import numpy as np
import matplotlib.pyplot as plt
from os.path import sep
import pickle as pkl

from volpy import run_volpy
from volpy import quality_control
from volpy import combine_sessions

def quality_control(population_data_path, data_paths, metadata_file,
                        snr_cutoffs = [3, 3.5, 4, 4.5, 5, 5.5]):

    try:
        with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
            qc_results = pkl.load(f)
        print('Selected SNR cutoff: {0}'.format(qc_results['snr_cutoff']))

    except:
        movies = list(data_paths.keys())
        cells = {snr: {movie: [] for movie in movies} for snr in snr_cutoffs}
        blocks = {snr: {movie: [] for movie in movies} for snr in snr_cutoffs}

        for movie in movies:
            print('Movie {0}'.format(movie))
            data_path = data_paths[movie]
            volpy_results = run_volpy.run_volpy(data_path, metadata_file)
            try:
                snr_movie = volpy_results['combined_data']['snr']
                cells_movie = volpy_results['combined_data']['cells']
                print('     Combined data loaded')
            except KeyError:
                print('     Combined data not found, compiling')
                combine_sessions.combine_sessions(data_path, metadata_file, volpy_results)
                volpy_results = run_volpy.run_volpy(data_path, metadata_file)
                snr_movie = volpy_results['combined_data']['snr']
                cells_movie = volpy_results['combined_data']['cells']

            for snr in snr_cutoffs:
                for cell in cells_movie:
                    blocks_cell = np.where(snr_movie[np.where(cells_movie == cell)[0][0], :] >= snr)[0]
                    if len(blocks_cell > 0):
                        cells[snr][movie] = np.append(cells[snr][movie], cell)
                        blocks[snr][movie] = np.append(blocks[snr][movie], len(blocks_cell))

        plt.figure()
        blocks_all = {snr: [] for snr in snr_cutoffs}
        for snr in snr_cutoffs:
            for movie in movies:
                blocks_all[snr] = np.append(blocks_all[snr], blocks[snr][movie])
            blocks_all[snr] = np.flip(np.sort(blocks_all[snr]))
            plt.plot(blocks_all[snr], label = snr)
        plt.legend(title = 'SNR cutoff')
        plt.title('Quality control')
        plt.xlabel('Number of cells')
        plt.ylabel('Number of blocks with SNR > cutoff')

        selected_cutoff = float(input('Select SNR cutoff: '))

        qc_results = {  'snr_cutoff': selected_cutoff,
                        'cells': cells,
                        'blocks':blocks,
                        }

        with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(qc_results, f)

    return qc_results
