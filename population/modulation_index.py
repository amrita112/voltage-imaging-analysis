from population import population_psth
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import mannwhitneyu

def modulation_index(population_data_path, metadata_file, movies, data_paths, genotype,
                     psth_type = 'spike_rate', metric = 'trial_type_selectivity', method = 'mann_whiteney',
                     plot_example_neurons = 'False'):

    psth = population_psth.get_population_psth(population_data_path, movies, data_paths, metadata_file, genotype,
                                                 make_plot_spike_psth = False, make_plot_dFF_psth = False, plot_psths = False)
    if psth_type == 'spike_rate':
        psth = psth['spikes']
        left_trials_activity = psth['spike_count_trials_left']
        right_trials_activity = psth['spike_count_trials_right']
    else:
        if psth_type == 'dFF':
            psth = psth['dFF']
        else:
            print('\'psth_type\' must be either \'spike_rate\' or \'dFF\'')
            return

    n_cells = len(left_trials_activity.keys())
    assert(len(right_trials_activity.keys()) == n_cells)

    if metric == 'trial_type_selectivity':
        mod_ind = np.zeros([n_cells, 4])
        for cell in range(n_cells):
            for i in range(4):
                    mod_ind[cell, i] = compare_activity(left_trials_activity[cell][:, i], right_trials_activity[cell][:, i],
                                            method)
    else:
        if metric == 'epoch_selectivity':
            mod_ind = np.zeros([n_cells, 4])
            for cell in range(n_cells):
                for i in range(1, 4):
                    pre_sample_vec = np.concatenate([left_trials_activity[cell][:, 0],
                                                    right_trials_activity[cell][:, 0]], axis = 0)
                    assert(len(pre_sample_vec) == left_trials_activity[cell].shape[0] + right_trials_activity[cell].shape[0])
                    epoch_vec = np.concatenate([left_trials_activity[cell][:, i],
                                                    right_trials_activity[cell][:, i]], axis = 0)
                    assert(len(epoch_vec) == left_trials_activity[cell].shape[0] + right_trials_activity[cell].shape[0])
                    mod_ind[cell, i] = compare_activity(epoch_vec, pre_sample_vec, method)
        else:
            if metric == 'tuning_curve_variance':
                mod_ind = np.zeros([n_cells])
                for cell in range(n_cells):
                    tuning_curve = np.zeros(8)
                    tuning_curve[:4] = np.mean(left_trials_activity[cell], axis = 0)
                    tuning_curve[4:] = np.mean(right_trials_activity[cell], axis = 0)
                    mod_ind[cell] = np.var(tuning_curve)
                if plot_example_neurons:
                    fig, ax = plt.subplots(nrows = 2, ncols = 1, constrained_layout = True, figsize = [5, 4])
                    #neuron_nos = [np.argmax(mod_ind), np.argmin(mod_ind)]
                    order = np.argsort(mod_ind)
                    neuron_nos = [order[np.random.randint(0, high = int(n_cells/5))],
                                    order[np.random.randint(int(4*n_cells/5), high = n_cells)]]
                    for i in range(2):
                        mean = np.zeros(8)
                        sem = np.zeros(8)
                        mean[:4] = np.mean(left_trials_activity[neuron_nos[i]], axis = 0)
                        mean[4:] = np.mean(right_trials_activity[neuron_nos[i]], axis = 0)
                        sem[:4] = np.std(left_trials_activity[neuron_nos[i]], axis = 0)/np.sqrt(left_trials_activity[neuron_nos[i]].shape[0])
                        sem[4:] = np.std(right_trials_activity[neuron_nos[i]], axis = 0)/np.sqrt(right_trials_activity[neuron_nos[i]].shape[0])
                        ax[i].errorbar(list(range(8)), mean, sem, color = 'gray')
                        ax[i].set_xticks(list(range(8)))
                        ax[i].set_ylabel('Mean spike count', fontsize = 10)
                        ax[i].set_title('Tuning curve variance = {0}'.format(np.round(mod_ind[neuron_nos[i]], 1)),
                                            fontsize = 12, pad = 5)
                        ax[i].set_yticklabels(np.round(ax[i].get_yticks(), 1), fontsize = 8)
                    ax[0].set_xticklabels([])
                    ax[1].set_xticklabels(['Pre-sample', 'S', 'D', 'R', 'Pre-sample', 'S', 'D', 'R'], fontsize = 10)
                    #ax[1].text(1, -2, 'Left trials', fontsize = 15)
                    #ax[1].text(5, -2, 'Right trials', fontsize = 15)
            else:
                print('\'metric\' must be one of the following:\n\'trial_type_selectivity\'\n\'epoch_selectivity\'\n\'tuning_curve_variance\'')
    return mod_ind

def compare_activity(left, right, method = 'mann_whitney', pval_threshold = 0.05):

    if method == 'mann_whiteney':
        try:
            pval = mannwhitneyu(left, right, alternative = 'two-sided')[1]
        except ValueError as e:
            print(e)
            pval = 1
        return pval < pval_threshold
    else:
        if method == 'd-prime':
            if np.std(left) + np.std(right) == 0:
                if np.mean(left) == np.mean(right):
                    d_prime = 0
                else:
                    print('Unequal means with zero variance')
            else:
                d_prime = np.divide(np.mean(left) - np.mean(right), (np.std(left) + np.std(right))/2)
            return d_prime
        else:
            print('\'method\' must be \'mann_whitney\' or \'d-prime\'')
        return
