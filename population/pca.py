
from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA

def pca_session(data_path, metadata_file, cell_ids, dFF, spike_frames, tvec, frame_rate, pca_file = 'pca.pkl', spike_bin_ms = 20, n_components = 0, overwrite = False, make_plots = False, save_plots = False, save_path = None):

    n_cells = len(cell_ids)
    if n_components == 0:
        n_components = n_cells

    try:
        with open('{0}{1}{2}-component_{3}'.format(data_path, sep, n_components, pca_file), 'rb') as f:
            pca_dict = pkl.load(f)
        dFF_pca = pca_dict['dFF']
        spike_pca = pca_dict['spike']

    except:
        overwrite = True

    spike_bin_frames = int(spike_bin_ms*frame_rate/1000)

    if overwrite:

        dFF_pca = {}
        spike_pca = {}
        print('dF/F PCA, keeping {0} components'.format(n_components))
        assert(len(cell_ids) == dFF.shape[0])
        assert(len(tvec) == dFF.shape[1])

        mean_dFF = np.mean(dFF, axis = 1)
        assert(len(mean_dFF) == n_cells)
        mean_dFF = np.reshape(mean_dFF, [n_cells, 1])
        dFF = dFF - mean_dFF
        pca = PCA(n_components = n_components)
        pca.fit(dFF)
        dFF_pca['components'] = pca.components_
        dFF_pca['expl_var'] = pca.explained_variance_ratio_

        assert(len(list(spike_frames.keys())) == len(cell_ids))
        n_frames = len(tvec)
        spike_vectors = np.zeros([n_cells, n_frames])
        print('           Binning spikes for {0} cells'.format(n_cells))
        for cell in tqdm(range(n_cells)):
            spike_frames_cell = spike_frames[cell_ids[cell]]
            for frame in range(n_frames):
                first_frame = np.max([0, frame - spike_bin_frames])
                spike_frames_causal = spike_frames_cell[spike_frames_cell < frame]
                spike_vectors[cell, frame] = np.sum(spike_frames_causal > first_frame)
            spike_vectors[cell, :] = spike_vectors[cell, :]*1000/spike_bin_ms
            spike_vectors[cell, :] -= np.mean(spike_vectors[cell, :])

        pca = PCA(n_components = n_components)
        pca.fit(spike_vectors)
        spike_pca['components'] = pca.components_
        spike_pca['expl_var'] = pca.explained_variance_ratio_

        pca_dict = {'dFF': dFF_pca, 'spike': spike_pca}
        with open('{0}{1}{2}-component_{3}'.format(data_path, sep, n_components, pca_file), 'wb') as f:
            pkl.dump(pca_dict, f)

    if make_plots:

        fig, ax = plt.subplots(nrows = 1, ncols = 2, constrained_layout = True)

        ax[0].plot(list(range(1, n_components + 1)), np.cumsum(spike_pca['expl_var']), color = 'k')
        ax[0].set_xlabel('Number of components')
        ax[0].set_ylabel('Percentage of variance explained')
        ax[0].set_title('Spike rate PCA')

        ax[1].plot(list(range(1, n_components + 1)), np.cumsum(dFF_pca['expl_var']), color = 'k')
        ax[1].set_xlabel('Number of components')
        ax[1].set_ylabel('Percentage of variance explained')
        ax[1].set_title('dF/F PCA')

        if save_plots:
            if save_path == None:
                save_path = data_path
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            plt.savefig('{0}{1}PCA_expl_var.png'.format(save_path, sep))

    return pca_dict
