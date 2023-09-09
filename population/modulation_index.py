from population import population_psth
import numpy as np
from matplotlib import pyplot as plt
from os.path import sep

from scipy.stats import mannwhitneyu

def modulation_index(activity_dict, epoch_start_timepoints, epoch_end_timepoints,
                     metric = 'trial_type_selectivity', method = 'mann_whiteney',
                     trial_types = ['Left', 'Right'], trial_epochs = ['Pre-sample', 'Sample', 'Delay', 'Response'],
                     baseline_epoch = 'Pre-sample', colors = {'Left': 'r', 'Right': 'b'},
                     plot_example_neurons = 'False', figsize = [8, 4], savefig = False, save_path = None):

    """ Calculate and plot the trial type-selectivity or trial epoch-selectivity index over all neurons in the population.

        Inputs:
        activity_dict: Dictionary with a key-value pair for each trial type, itself containing a dictionary with keys 0, 1, 2... n_cells
                       with a n_trials X n_timepoints activity matrix for each cell
        epoch_start_timepoints: list of starting timepoint for each epoch. Should be equal in length to trial_epochs
        epoch_end_timepoints: list of ending timepoint for each epoch. Should be equal in length to trial_epochs
        metric: String, 'trial_type_selectivity' (default), 'trial_epoch_selectivity' or 'tuning_curve_variance'.
        method: String, 'mann_whiteney' (default) or 'd-prime' - method for comparing activity in trial types/epochs.
        trial_types: list of trial types, should match activity_dict.keys(). Currently, only two trial types are supported.
        trial_epochs: list of trial_epochs. Should be equal in length to epoch_start_timepoints
        baseline_epoch: String, one of the elements of trial_epochs. Used as a comparison for calculating trial epoch modulation index

        Output:
        mod_ind: Dictionary with a key-value pair for each trial epoch, itself containing:
                 - if metric is 'trial_type_selectivity': n_cells X 1 vector with modulation index for each cell
                 - if metric is 'trial_epoch_selectivity': n_cells X len(trial_types) matrix with modulation index for each cell and each trial type.
                                                           In this case, mod_ind[baseline_epoch] is a matrix of zeros.

    """
    n_epochs = len(trial_epochs)
    assert(len(epoch_start_timepoints) == n_epochs)
    assert(len(epoch_end_timepoints) == n_epochs)

    n_types = len(trial_types)
    assert(n_types == 2)
    assert(np.all(list(activity_dict.keys()) == trial_types))

    n_cells = len(activity_dict[trial_types[0]].keys())
    for type in trial_types:
        assert(len(activity_dict[type].keys()) == n_cells)

    n_timepoints = activity_dict[type][0].shape[1]
    assert(n_timepoints > np.max(epoch_start_timepoints))

    assert(baseline_epoch in trial_epochs)
    baseline_epoch_no = trial_epochs.index(baseline_epoch)

    if metric == 'trial_type_selectivity':
        mod_ind = {epoch: np.zeros(n_cells) for epoch in trial_epochs}
        for cell in range(n_cells):
            for e in range(n_epochs):

                    t1 = epoch_start_timepoints[e]
                    t2 = epoch_end_timepoints[e]

                    v1 = np.mean(activity_dict[trial_types[0]][cell][:, t1:t2], axis = 1)
                    assert(len(v1) == activity_dict[trial_types[0]][cell].shape[0])
                    v2 = np.mean(activity_dict[trial_types[1]][cell][:, t1:t2], axis = 1)
                    assert(len(v2) == activity_dict[trial_types[1]][cell].shape[0])

                    mod_ind[trial_epochs[e]][cell] = compare_activity(v1, v2, method)

        plt.figure(constrained_layout = True, figsize = figsize)
        for e in range(n_epochs):
            plt.boxplot(mod_ind[trial_epochs[e]], positions = [e + 1])

        plt.xticks(list(range(1, n_epochs + 1)), labels = trial_epochs)
        plt.ylabel('left-right selectivity\n({0})'.format(method))
        if savefig:
            plt.savefig('{0}{1}{2}_{3}.png'.format(save_path, sep, metric, method))

    else:
        if metric == 'trial_epoch_selectivity':
            mod_ind = {epoch: np.zeros([n_cells, n_types]) for epoch in trial_epochs}
            for cell in range(n_cells):

                # Baseline epoch activity for trial type 1
                vb1 = np.mean(activity_dict[trial_types[0]][cell][:, epoch_start_timepoints[baseline_epoch_no]:epoch_end_timepoints[baseline_epoch_no]], axis = 1)
                assert(len(vb1) == activity_dict[trial_types[0]][cell].shape[0])

                # Baseline epoch activity for trial type 2
                vb2 = np.mean(activity_dict[trial_types[1]][cell][:, epoch_start_timepoints[baseline_epoch_no]:epoch_end_timepoints[baseline_epoch_no]], axis = 1)
                assert(len(vb2) == activity_dict[trial_types[1]][cell].shape[0])

                for e in range(n_epochs):

                    if not e == baseline_epoch_no:

                        t1 = epoch_start_timepoints[e]
                        t2 = epoch_end_timepoints[e]

                        # Modulation index for trial type 1
                        v1 = np.mean(activity_dict[trial_types[0]][cell][:, t1:t2], axis = 1)
                        assert(len(v1) == activity_dict[trial_types[0]][cell].shape[0])

                        mod_ind[trial_epochs[e]][cell, 0] = compare_activity(v1, vb1, method)

                        # Modulation index for trial type 2
                        v2 = np.mean(activity_dict[trial_types[1]][cell][:, t1:t2], axis = 1)
                        assert(len(v2) == activity_dict[trial_types[1]][cell].shape[0])

                        mod_ind[trial_epochs[e]][cell, 1] = compare_activity(v2, vb2, method)

            plt.figure(constrained_layout = True, figsize = figsize)
            for e in range(n_epochs):
                if not e == baseline_epoch_no:

                    c = colors[trial_types[0]]
                    plt.boxplot(mod_ind[trial_epochs[e]][:, 0], positions = [e + 0.25], patch_artist=True,
                                    boxprops=dict(facecolor=c, color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    flierprops=dict(color=c, markeredgecolor=c),
                                    medianprops=dict(color=c))

                    c = colors[trial_types[1]]
                    plt.boxplot(mod_ind[trial_epochs[e]][:, 1], positions = [e + 0.5], patch_artist=True,
                                    boxprops=dict(facecolor=c, color=c),
                                    capprops=dict(color=c),
                                    whiskerprops=dict(color=c),
                                    flierprops=dict(color=c, markeredgecolor=c),
                                    medianprops=dict(color=c))

            plt.xticks([e + 0.375 for e in range(n_epochs) if not e == baseline_epoch_no],
                        labels = [e for e in trial_epochs if not e == baseline_epoch])
            plt.ylabel('Modulation in trial epoch\ncompared to baseline ({0})'.format(method))
            if savefig:
                plt.savefig('{0}{1}{2}_{3}.png'.format(save_path, sep, metric, method))

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

def compare_activity(v1, v2, method = 'mann_whiteney', pval_threshold = 0.05):
    """ Compare two activity vectors v1 or v2 using mann-mann_whiteney U test or d prime.
    """

    if method == 'mann_whiteney':
        try:
            pval = mannwhitneyu(v1, v2, alternative = 'two-sided')[1]
        except ValueError as e:
            print(e)
            pval = 1
        return pval < pval_threshold
    else:
        if method == 'd-prime':
            if np.std(v1) + np.std(v2) == 0:
                if np.mean(v1) == np.mean(v2):
                    d_prime = 0
                else:
                    print('Unequal means with zero variance')
            else:
                d_prime = np.divide(np.mean(v1) - np.mean(v2), (np.std(v1) + np.std(v2))/2)
            return d_prime
        else:
            print('\'method\' must be \'mann_whitney\' or \'d-prime\'')
        return
