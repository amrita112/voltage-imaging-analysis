from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy

from population import population_psth

from segmentation import get_roi_arrays


def hier_clust(population_data_path, data_paths, metadata_file, input = 'Spike psth', n_clusters = 4):

    print('TO BE CHECKED: Trial type order matches number of frames per trial ')
    print('But there are some \'correct\' trials with less frames than go cue')
    print('Padding zeros to dF/F for trials with less frames than max trial length.')
    if input == 'Spike psth':
        pop_psth = population_psth.get_population_psth(population_data_path, data_paths, metadata_file, plot_psths = False)
        vectors = pop_psth['spikes']
        tvec = pop_psth['tvec']
        ticks = [0, pop_psth['sample_end_bin_left'], pop_psth['go_cue_bin_left'],
                            pop_psth['sample_start_bin_right'], pop_psth['sample_end_bin_right'], pop_psth['go_cue_bin_right']]

    else:
        if input == 'dFF psth':
            pop_psth = population_psth.get_population_psth(population_data_path, data_paths, metadata_file, plot_psths = False)
            vectors = pop_psth['dFF']
            tvec = pop_psth['dFF_tvec']
            ticks = [0, pop_psth['sample_end_frame_left'], pop_psth['go_cue_frame_left'],
                                pop_psth['sample_start_frame_right'], pop_psth['sample_end_frame_right'], pop_psth['go_cue_frame_right']]

        else:
            print('\'input\' must be \'Spike psth\' or \'dFF psth\'')

    norm_input = norm_vectors(vectors)
    dist_matrix = distance.pdist(norm_input, metric = 'correlation')
    Z = hierarchy.linkage(dist_matrix, method = 'ward')

    plt.figure()
    hierarchy.dendrogram(Z)
    plt.xlabel('Neuron #')
    plt.title('{0}-based clustering'.format(input))
    plt.savefig('{0}{1}{2}_dendrogram'.format(population_data_path, sep, input))

    cluster_identities = hierarchy.cut_tree(Z, n_clusters = n_clusters)
    cluster_identities = np.reshape(cluster_identities, [-1])
    order = np.argsort(cluster_identities)

    dif = np.diff(np.sort(cluster_identities))
    cluster_boundaries = np.where(dif == 1)[0]

    if input == 'Spike psth':
        colorbar_label = 'Firing rate'
        unit = '(Hz)'
        figsize = [8, 6]
    else:
        if input == 'dFF psth':
            colorbar_label = '-dF/F'
            unit = ''
            figsize = [8, 6]

    plot_pop_psth(pop_psth, vectors, list(range(vectors.shape[0])), [], tvec, ticks, ylabel = 'Neuron #',
                    save_path = '{0}{1}{2}_population.png'.format(population_data_path, sep, input),
                    colorbar_label = '{0} {1}'.format(colorbar_label, unit), figsize = figsize)
    plot_pop_psth(pop_psth, norm_input, list(range(vectors.shape[0])), [], tvec, ticks, ylabel = 'Neuron #',
                        colorbar_label = 'Z-scored {0}'.format(colorbar_label), figsize = figsize,
                    save_path = '{0}{1}Normalized_{2}_population.png'.format(population_data_path, sep, input))
    plot_pop_psth(pop_psth, norm_input, order, cluster_boundaries, tvec, ticks,
                        colorbar_label = 'Z-scored {0}'.format(colorbar_label), figsize = figsize,
                    save_path = '{0}{1}Normalized_{2}_clusters.png'.format(population_data_path, sep, input))
    plot_pop_psth(pop_psth, vectors, order, cluster_boundaries, tvec, ticks,
                    colorbar_label = '{0} {1}'.format(colorbar_label, unit), figsize = figsize,
                    save_path = '{0}{1}{2}_clusters.png'.format(population_data_path, sep, input))

    plot_cluster_wise_psth(pop_psth, vectors, n_clusters, cluster_identities, tvec, ticks,
                            input = input, ylabel = '{0} {1}'.format(colorbar_label, unit),
                            save_path = population_data_path)

def snr_by_cluster(population_data_path, data_paths, metadata_file, n_clusters = 3, input = 'spike_psth',
                        overwrite = False):

    movies = list(data_paths.keys())
    with open('{0}{1}qc_results.py'.format(population_data_path, sep), 'rb') as f:
        qc_results = pkl.load(f)
        snr_cutoff = qc_results['snr_cutoff']
        cells = qc_results['cells'][float(snr_cutoff)]
        blocks = qc_results['blocks'][float(snr_cutoff)]
    total_cells = np.sum([len(cells[movie]) for movie in movies])

    try:
        with open('{0}{1}snr_by_cluster.pkl', 'rb') as f:
            snr_by_cluster = pkl.load(f)
            clusters = list(snr_by_cluster.keys())

    except:
        overwrite = True

    if overwrite:

        print('Getting SNR distribution for each cluster')

        if input == 'spike_psth':
            pop_psth = population_psth.get_population_psth(population_data_path, data_paths, metadata_file, plot_psths = False)
            vectors = pop_psth['spikes']

        else:
            print('\'input\' must be \'spike_psth\'')

        norm_input = norm_vectors(vectors)
        dist_matrix_full = distance.pdist(norm_input, metric = 'correlation')
        Z = hierarchy.linkage(dist_matrix_full, method = 'ward')
        cluster_identities = hierarchy.cut_tree(Z, n_clusters = n_clusters)
        cluster_identities = np.reshape(cluster_identities, [-1])
        clusters = list(range(np.max(cluster_identities) + 1))
        snr_by_cluster = {cluster: [] for cluster in clusters}
        snr_all = []

        for movie in movies:

            print('Movie {0}'.format(movie + 1))

            good_cells = list(cells[movie].astype(int))
            good_blocks = blocks[movie]
            snr_movie = np.zeros(len(good_cells))

            with open('{0}{1}{2}'.format(data_paths[movie], sep, metadata_file), 'rb') as f:
                metadata = pkl.load(f)
            sessions_to_process = metadata['sessions_to_process']
            batch_data = metadata['batch_data']

            roi_arrays = get_roi_arrays.get_roi_arrays(data_paths[movie], metadata_file)
            n_cells = roi_arrays[sessions_to_process[0]].shape[0]

            volpy_results_file = metadata['volpy_results_file']
            with open('{0}{1}{2}'.format(data_paths[movie], sep, volpy_results_file), 'rb') as f:
                volpy_results = pkl.load(f)

            for cell in range(n_cells):
                if cell in good_cells:
                    for session in sessions_to_process:

                        n_batches = batch_data[session]['n_batches']
                        for batch in range(n_batches):
                            if batch > good_blocks[good_cells.index(cell)]:
                                snr_movie[good_cells.index(cell)] /= batch
                                continue
                            else:
                                if batch == n_batches - 1:
                                    snr_movie[good_cells.index(cell)] /= batch
                                else:
                                    estimates = volpy_results[session][batch]['vpy']
                                    snr = estimates['snr']
                                    snr_movie[good_cells.index(cell)] += snr[cell]
            snr_all = np.append(snr_all, snr_movie, axis = 0)

        print(snr_all.shape)
        for cluster in clusters:
            snr_by_cluster[cluster] = snr_all[np.where(cluster_identities == cluster)[0]]

        with open('{0}{1}snr_by_cluster.pkl'.format(population_data_path, sep), 'wb') as f:
            pkl.dump(snr_by_cluster, f)

    fig, ax = plt.subplots(nrows = len(clusters), ncols = 1, constrained_layout = True, figsize = (4, 8))
    for cluster in clusters:
        ax[cluster].hist(snr_by_cluster[cluster])
        ax[cluster].set_title('Cluster {0}'.format(cluster + 1))
        ax[cluster].set_xlabel('SNR')
        ax[cluster].set_ylabel('Number of neurons')

    plt.savefig('{0}{1}SNR_by_cluster.png'.format(population_data_path, sep))



def validate_clustering(population_data_path, data_paths, metadata_file, n_clusters = 3,
                input = 'Spike psth',
                        overwrite_spike_psth = False, overwrite_dFF_psth = False,):

    pop_psth_all = population_psth.get_population_psth(population_data_path, data_paths, metadata_file,
                                                    plot_psths = False)
    try:
        with open('{0}{1}hier_clust_validation.py'.format(population_data_path, sep), 'rb') as f:
            hc_validation = pkl.load(f)
            spike_psth_validation = hc_validation['spike_psth'] # Matrix of number of neurons X number of bins

    except:
        overwrite_spike_psth = True

    try:
        with open('{0}{1}hier_clust_validation.py'.format(population_data_path, sep), 'rb') as f:
            hc_validation = pkl.load(f)
            dFF_psth_validation = hc_validation['dFF_psth'] # Matrix of number of neurons X number of frames
    except:
        overwrite_dFF_psth = True

    if overwrite_dFF_psth or overwrite_spike_psth:
        hc_validation = {}

    if overwrite_dFF_psth:

        hc_validation['dFF_psth'] = {}
        vectors = pop_psth_all['dFF']

        norm_input = norm_vectors(vectors)
        dist_matrix_full = distance.pdist(norm_input, metric = 'correlation')
        Z = hierarchy.linkage(dist_matrix_full, method = 'ward')
        cluster_identities = hierarchy.cut_tree(Z, n_clusters = n_clusters)
        cluster_identities = np.reshape(cluster_identities, [-1])
        cluster_order = np.argsort(cluster_identities)
        dif = np.diff(np.sort(cluster_identities))
        cluster_boundaries = np.where(dif == 1)[0]

        n_neurons = norm_input.shape[0]
        clust_diff = np.zeros(n_neurons)
        corr_matrix = np.zeros([n_neurons, n_neurons])

        for n in range(n_neurons):

            if np.mod(n, 10) == 0:
                print('Neuron {0}'.format(n + 1))

            for m in range(n):
                index = int(n_neurons*m + n - (m + 2)*(m + 1)/2)
                corr_matrix[m, n] = dist_matrix_full[index]
                corr_matrix[n, m] = corr_matrix[m, n]

            X = np.zeros([n_neurons - 1, norm_input.shape[1]])
            X[:n, :] = norm_input[:n, :]
            X[n:, :] = norm_input[(n + 1):, :]
            dist_matrix = distance.pdist(X, metric = 'correlation')
            Z = hierarchy.linkage(dist_matrix, method = 'ward')
            new_cluster_identities = hierarchy.cut_tree(Z, n_clusters = n_clusters)
            new_cluster_identities = np.reshape(new_cluster_identities, [-1])

            old_cluster_identities = np.zeros(n_neurons - 1)
            old_cluster_identities[:n] = cluster_identities[:n]
            old_cluster_identities[n:] = cluster_identities[(n + 1):]
            if not hierarchy.is_isomorphic(new_cluster_identities, old_cluster_identities):

                #print('     Original clusters: {0}'.format(old_cluster_identities))
                #print('     New clusters: {0}'.format(new_cluster_identities))
                diff = (n_neurons - np.sum(old_cluster_identities == new_cluster_identities))/n_neurons
                print('     Neuron {0}: {1}% difference'.format(n + 1, np.round(diff*100, 2)))

            cluster_means = np.zeros([n_clusters, norm_input.shape[1]])
            max_corr = 0
            new_cluster = 0
            for cluster in range(n_clusters):
                cluster_means[cluster, :] = np.mean(X[np.where(new_cluster_identities == cluster)[0], :], axis = 0)
                corr =  np.corrcoef(norm_input[n, :], cluster_means[cluster, :])[0, 0]
                if corr > max_corr:
                    max_corr = corr
                    new_cluster = cluster

            if new_cluster == cluster_identities[n]:
                clust_diff[n] = 0
            else:
                clust_diff[n] = 1

            hc_validation['dFF_psth']['clust_diff'] = clust_diff
            hc_validation['dFF_psth']['n_neurons'] = n_neurons
            hc_validation['dFF_psth']['corr_matrix'] = corr_matrix
            hc_validation['dFF_psth']['cluster_boundaries'] = cluster_boundaries
            print('HC based on dFF PSTH')
            print('Fraction of neurons with different cluster after leaving out = {0}'.format(np.sum(clust_diff)/n_neurons))

    if overwrite_spike_psth:

        hc_validation['spike_psth'] = {}
        vectors = pop_psth_all['spikes']

        norm_input = norm_vectors(vectors)
        dist_matrix_full = distance.pdist(norm_input, metric = 'correlation')
        Z = hierarchy.linkage(dist_matrix_full, method = 'ward')
        cluster_identities = hierarchy.cut_tree(Z, n_clusters = n_clusters)
        cluster_identities = np.reshape(cluster_identities, [-1])
        cluster_order = np.argsort(cluster_identities)
        dif = np.diff(np.sort(cluster_identities))
        cluster_boundaries = np.where(dif == 1)[0]

        n_neurons = norm_input.shape[0]
        clust_diff = np.zeros(n_neurons)
        corr_matrix = np.zeros([n_neurons, n_neurons])

        for n in range(n_neurons):

            if np.mod(n, 10) == 0:
                print('Neuron {0}'.format(n + 1))

            for m in range(n):
                index = int(n_neurons*m + n - (m + 2)*(m + 1)/2)
                corr_matrix[m, n] = dist_matrix_full[index]
                corr_matrix[n, m] = corr_matrix[m, n]

            X = np.zeros([n_neurons - 1, norm_input.shape[1]])
            X[:n, :] = norm_input[:n, :]
            X[n:, :] = norm_input[(n + 1):, :]
            dist_matrix = distance.pdist(X, metric = 'correlation')
            Z = hierarchy.linkage(dist_matrix, method = 'ward')
            new_cluster_identities = hierarchy.cut_tree(Z, n_clusters = n_clusters)
            new_cluster_identities = np.reshape(new_cluster_identities, [-1])

            old_cluster_identities = np.zeros(n_neurons - 1)
            old_cluster_identities[:n] = cluster_identities[:n]
            old_cluster_identities[n:] = cluster_identities[(n + 1):]
            if not hierarchy.is_isomorphic(new_cluster_identities, old_cluster_identities):

                #print('     Original clusters: {0}'.format(old_cluster_identities))
                #print('     New clusters: {0}'.format(new_cluster_identities))
                diff = (n_neurons - np.sum(old_cluster_identities == new_cluster_identities))/n_neurons
                print('     Neuron {0}: {1}% difference'.format(n + 1, np.round(diff*100, 2)))

            cluster_means = np.zeros([n_clusters, norm_input.shape[1]])
            max_corr = 0
            new_cluster = 0
            for cluster in range(n_clusters):
                cluster_means[cluster, :] = np.mean(X[np.where(new_cluster_identities == cluster)[0], :], axis = 0)
                corr =  np.corrcoef(norm_input[n, :], cluster_means[cluster, :])[0, 0]
                if corr > max_corr:
                    max_corr = corr
                    new_cluster = cluster

            if new_cluster == cluster_identities[n]:
                clust_diff[n] = 0
            else:
                clust_diff[n] = 1

            hc_validation['spike_psth']['clust_diff'] = clust_diff
            hc_validation['spike_psth']['n_neurons'] = n_neurons
            hc_validation['spike_psth']['corr_matrix'] = corr_matrix
            hc_validation['spike_psth']['cluster_boundaries'] = cluster_boundaries

            print('HC based on spike PSTH')
            print('Fraction of neurons with different cluster after leaving out = {0}'.format(np.sum(clust_diff)/n_neurons))


        with open('{0}{1}hier_clust_validation.pkl', 'wb') as f:
            pkl.dump(hc_validation, f)

    corr_matrix = hc_validation['spike_psth']['corr_matrix']
    cluster_boundaries = hc_validation['spike_psth']['cluster_boundaries']
    n_neurons = hc_validation['spike_psth']['n_neurons']

    plt.figure()
    plt.imshow(corr_matrix)
    for cb in cluster_boundaries:
        plt.plot(np.ones(n_neurons)*cb, list(range(n_neurons)), color = 'w', linestyle = '--', linewidth = 2)
        plt.plot(list(range(n_neurons)), np.ones(n_neurons)*cb, color = 'w', linestyle = '--', linewidth = 2)
    plt.xlabel('Neuron # ordered by cluster')
    plt.ylabel('Neuron # ordered by cluster')
    plt.title('Correlation distance for Spike PSTH')
    plt.colorbar(label = 'Correlation distance')
    plt.savefig('{0}{1}Spike_psthcorrelation.png'.format(population_data_path, sep))

    corr_matrix = hc_validation['dFF_psth']['corr_matrix']
    cluster_boundaries = hc_validation['dFF_psth']['cluster_boundaries']
    n_neurons = hc_validation['dFF_psth']['n_neurons']

    plt.figure()
    plt.imshow(corr_matrix)
    for cb in cluster_boundaries:
        plt.plot(np.ones(n_neurons)*cb, list(range(n_neurons)), color = 'w', linestyle = '--', linewidth = 2)
        plt.plot(list(range(n_neurons)), np.ones(n_neurons)*cb, color = 'w', linestyle = '--', linewidth = 2)
    plt.xlabel('Neuron # ordered by cluster')
    plt.ylabel('Neuron # ordered by cluster')
    plt.colorbar(label = 'Correlation distance')
    plt.title('Correlation distance for dFF PSTH')
    plt.savefig('{0}{1}dFF_psthcorrelation.png'.format(population_data_path, sep))



def norm_vectors(vectors, method = 'z_score'):

    norm_vectors = np.zeros(vectors.shape)

    for row in range(vectors.shape[0]):
        if method == 'z_score':
            norm_vectors[row, :] = (vectors[row, :] - np.mean(vectors[row, :]))/np.std(vectors[row, :])
        else:
            if method == 'max_divide':
                norm_vectors[row, :] = vectors[row, :]/np.max(vectors[row, :])
            else:
                print('\'method\' for norm_vectors should be either \'z_score\' or \'max_divide\'')

    return norm_vectors

def plot_cluster_wise_psth(pop_psth, vectors, n_clusters, cluster_identities, tvec, ticks,
                            figsize = [20, 6], n_rows = 3, input = 'Spike PSTH', ylabel = 'Firing rate (Hz)',
                            save_figs = True, save_path = None):

    n_bins = int(vectors.shape[1]/2)
    fig_avg, ax_avg = plt.subplots(nrows = 1, ncols = n_clusters, constrained_layout = True,
                                    figsize = [20, 4])

    print('Plotting PSTHs for each cluster')

    for cluster in range(n_clusters):

        print('Cluster {0}'.format(cluster + 1))

        neuron_ids = list(np.where(cluster_identities == cluster)[0])
        n_neurons = len(neuron_ids)

        avg_psth_left = np.mean(vectors[neuron_ids, :n_bins], axis = 0)
        avg_psth_right = np.mean(vectors[neuron_ids, n_bins:], axis = 0)
        ax_avg[cluster].plot(avg_psth_left, color = 'r', linewidth = 5)
        ax_avg[cluster].plot(avg_psth_right, color = 'b', linewidth = 5)
        ax_avg[cluster].set_title('Cluster {0}'.format(cluster + 1))

        n_cols = int(np.ceil(n_neurons/n_rows))
        fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = figsize, constrained_layout = True)
        for n in range(n_neurons):

            row = int(np.floor(n/n_cols))
            col = int(np.mod(n, n_cols))
            ax_n = ax[row, col]

            psth_left = vectors[neuron_ids[n]][:n_bins]
            psth_right = vectors[neuron_ids[n]][n_bins:]

            ax_n.plot(psth_left, color = 'r', linewidth = 2)
            ax_n.plot(psth_right, color = 'b', linewidth = 2)

            ax_avg[cluster].plot(psth_left, color = 'r', alpha = 0.1, linewidth = 3)
            ax_avg[cluster].plot(psth_right, color = 'b', alpha = 0.1, linewidth = 3)

            y0 = ax_n.get_ylim()[0]
            y1 = ax_n.get_ylim()[1]

            ax_n.plot(np.ones(10)*ticks[2], np.linspace(y0, y1, 10), color = 'k', linewidth = 0.9, linestyle = '--')
            ax_n.plot(np.ones(10)*ticks[1], np.linspace(y0, y1, 10), color = 'k', linewidth = 0.9, linestyle = '--')

            labels = tvec[ticks[:3]]
            ax_n.set_xticks(ticks[:3])
            ax_n.set_xticklabels(np.round(labels, 2))
            ax_n.set_xlabel('Time from go cue (s)')

            x_s1 = 0.2*(ticks[1])
            x_d1 = 0.4*(ticks[1] + ticks[2])
            x_r1 = 0.4*(ticks[2] + ticks[3])

            ax_n.text(x_s1, y0 + 0.9*(y1 - y0), 'S')
            ax_n.text(x_d1, y0 + 0.9*(y1 - y0), 'D')
            ax_n.text(x_r1, y0 + 0.9*(y1 - y0), 'R')

            ax_n.set_ylabel(ylabel)

        y0 = ax_avg[cluster].get_ylim()[0]
        y1 = ax_avg[cluster].get_ylim()[1]

        ax_avg[cluster].plot(np.ones(10)*ticks[2], np.linspace(y0, y1, 10), color = 'k', linewidth = 0.9, linestyle = '--')
        ax_avg[cluster].plot(np.ones(10)*ticks[1], np.linspace(y0, y1, 10), color = 'k', linewidth = 0.9, linestyle = '--')

        labels = tvec[ticks[:3]]
        ax_avg[cluster].set_xticks(ticks[:3])
        ax_avg[cluster].set_xticklabels(np.round(labels, 2))
        ax_avg[cluster].set_xlabel('Time from go cue (s)')

        x_s1 = 0.2*(ticks[1])
        x_d1 = 0.4*(ticks[1] + ticks[2])
        x_r1 = 0.4*(ticks[2] + ticks[3])

        ax_avg[cluster].text(x_s1, y0 + 0.9*(y1 - y0), 'Sample')
        ax_avg[cluster].text(x_d1, y0 + 0.9*(y1 - y0), 'Delay')
        ax_avg[cluster].text(x_r1, y0 + 0.9*(y1 - y0), 'Response')

        ax_avg[cluster].set_ylabel(ylabel)

        if save_figs:
            fig.savefig('{0}{1}Cluster{2}_{3}.png'.format(save_path, sep, cluster + 1, input))

    if save_figs:
        fig_avg.savefig('{0}{1}Cluster_mean_{2}.png'.format(save_path, sep, input))

def plot_pop_psth(pop_psth, vectors, neuron_order, cluster_boundaries, tvec, ticks,
                    colorbar_label = 'Firing rate (Hz)', figsize = [8, 6],
                    ylabel = 'Neuron # (ordered by cluster)',
                    save_fig = True, save_path = 'pop_psth.png'):

    plt.figure(constrained_layout = True, figsize = figsize)
    plt.imshow(vectors[neuron_order, :], aspect = 'auto')
    for cb in cluster_boundaries:
        plt.plot(list(range(vectors.shape[1])), np.ones(vectors.shape[1])*cb, color = 'white', linestyle = '--', linewidth = 2)
    plt.colorbar(label = colorbar_label)
    plt.ylabel(ylabel, fontsize = 15)

    n_neurons = vectors.shape[0]
    n_bins = int(vectors.shape[1]/2)

    for i in range(1, 6):
        plt.plot(np.ones(n_neurons)*ticks[i], list(range(n_neurons)), color = 'white', linewidth = 2)


    labels = tvec[ticks]
    plt.xticks(ticks = ticks, labels = np.round(labels, 2))
    plt.xlabel('Time from go cue (s)', fontsize = 15)

    plt.text(n_bins/3, -5, 'Lick left trials', fontsize = 15)
    plt.text(n_bins + n_bins/3, -5, 'Lick right trials', fontsize = 15)

    x_s1 = 0.2*(ticks[1])
    x_s2 = x_s1 + n_bins
    x_d1 = 0.4*(ticks[1] + ticks[2])
    x_d2 = x_d1 + n_bins
    x_r1 = 0.4*(ticks[2] + ticks[3])
    x_r2 = x_r1 + n_bins

    plt.text(x_s1, -2, 'Sample')
    plt.text(x_s2, -2, 'Sample')
    plt.text(x_d1, -2, 'Delay')
    plt.text(x_d2, -2, 'Delay')
    plt.text(x_r1, -2, 'Response')
    plt.text(x_r2, -2, 'Response')

    if save_fig:
        plt.savefig(save_path)
