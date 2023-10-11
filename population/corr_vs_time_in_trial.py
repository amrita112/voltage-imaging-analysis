import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import sep

from population import clustering
from behavior_responses import utils

def plot_pop_corr(activity_dict, epoch_start_timepoints, epoch_end_timepoints, tvec,
                  trial_types = ['Left', 'Right'], divide_sd = False,
                  plot_activity_matrices = False,
                  figsize = [10, 10], savefig = False, save_path = None):
    """Plot a heatmap showing the correlation of the population activity vector for each timepoint in a trial with each other timepoint.
       Inputs:
       activity_dict: Dictionary with a key-value pair for each trial type, itself containing a dictionary with keys 0, 1, 2... n_cells
                      with a n_trials X n_timepoints activity matrix for each cell
       epoch_start_timepoints: list of starting timepoint for each epoch. Should be equal in length to trial_epochs
       trial_types: list of trial types, should match activity_dict.keys(). Currently, only two trial types are supported.
    """
    n_types = len(trial_types)
    assert(n_types == 2)
    assert(np.all(list(activity_dict.keys()) == trial_types))

    n_cells = len(activity_dict[trial_types[0]].keys())
    for type in trial_types:
        assert(len(activity_dict[type].keys()) == n_cells)

    n_timepoints = activity_dict[type][0].shape[1]
    assert(n_timepoints > np.max(epoch_start_timepoints))
    assert(len(tvec) == n_timepoints)
    sdr_loc = -0.1*n_timepoints
    trial_type_loc = -0.2*n_timepoints

    # Build two population activity matrices, with the two trial types concatenated, using a randomly selectd half of trials for each.
    activity_matrix1 = np.zeros([n_cells, 2*n_timepoints])
    activity_matrix2 = np.zeros([n_cells, 2*n_timepoints])

    sd_matrix1 = np.zeros([n_cells, 2*n_timepoints])
    sd_matrix2 = np.zeros([n_cells, 2*n_timepoints])

    print('Building population activity matrices')
    for cell in tqdm(range(n_cells)):

        # Trial type 1
        n_trials = activity_dict[trial_types[0]][cell].shape[0]

        trial_set1 = np.random.choice(list(range(n_trials)), int(n_trials/2), replace = False)
        trial_set2 = [trial for trial in list(range(n_trials)) if not trial in trial_set1]
        assert(len(np.intersect1d(trial_set1, trial_set2, assume_unique = True)) == 0)

        activity_matrix1[cell, :n_timepoints] = np.mean(activity_dict[trial_types[0]][cell][trial_set1, :], axis = 0)
        activity_matrix2[cell, :n_timepoints] = np.mean(activity_dict[trial_types[0]][cell][trial_set2, :], axis = 0)
        sd_matrix1[cell, :n_timepoints] = np.std(activity_dict[trial_types[0]][cell][trial_set1, :], axis = 0)
        sd_matrix2[cell, :n_timepoints] = np.std(activity_dict[trial_types[0]][cell][trial_set2, :], axis = 0)

        # Trial type 2
        n_trials = activity_dict[trial_types[1]][cell].shape[0]

        trial_set1 = np.random.choice(list(range(n_trials)), int(n_trials/2), replace = False)
        trial_set2 = [trial for trial in list(range(n_trials)) if not trial in trial_set1]
        assert(len(np.intersect1d(trial_set1, trial_set2, assume_unique = True)) == 0)

        activity_matrix1[cell, n_timepoints:] = np.mean(activity_dict[trial_types[1]][cell][trial_set1, :], axis = 0)
        activity_matrix2[cell, n_timepoints:] = np.mean(activity_dict[trial_types[1]][cell][trial_set2, :], axis = 0)
        sd_matrix1[cell, n_timepoints:] = np.std(activity_dict[trial_types[1]][cell][trial_set1, :], axis = 0)
        sd_matrix2[cell, n_timepoints:] = np.std(activity_dict[trial_types[1]][cell][trial_set2, :], axis = 0)

    # Plot the activity matrices
    if plot_activity_matrices:

        ticks = np.concatenate([epoch_start_timepoints[1:], [t + n_timepoints for t in epoch_start_timepoints[1:]]])
        labels = np.concatenate([np.round([tvec[t] for t in epoch_start_timepoints[1:]], 2),
                                    np.round([tvec[t] for t in epoch_start_timepoints[1:]], 2)])

        fig1, ax1 = plt.subplots(nrows = 2, ncols = 1, figsize = figsize, constrained_layout = True, sharex = True)

        plot = ax1[0].imshow(activity_matrix1, aspect = 'auto', cmap = 'bwr', norm = utils.get_two_slope_norm(activity_matrix1, 1))
        plt.colorbar(mappable = plot, ax = ax1[0], label = 'Firing rate (Hz)', shrink = 1)

        plot = ax1[1].imshow(sd_matrix1, aspect = 'auto', cmap = 'bwr', norm = utils.get_two_slope_norm(sd_matrix1, 1))
        plt.colorbar(mappable = plot, ax = ax1[1], label = 'Standard deviation of firing rate (Hz)', shrink = 1)

        for t in epoch_start_timepoints:
            ax1[0].plot([t, t], [0, n_cells], color = 'k')
            ax1[0].plot([t + n_timepoints, t + n_timepoints], [0, n_cells], color = 'k')
            ax1[1].plot([t, t], [0, n_cells], color = 'k')
            ax1[1].plot([t + n_timepoints, t + n_timepoints], [0, n_cells], color = 'k')

        ax1[1].set_xticks(ticks)
        ax1[1].set_xticks(ticks)
        ax1[1].set_xticklabels(labels)
        ax1[1].set_xlabel('Time from go cue (s)')
        ax1[0].set_ylabel('Cell #')
        ax1[0].set_title('Mean across trials')
        ax1[1].set_ylabel('Cell #')
        ax1[1].set_title('Standard deviation across trials')

        fig2, ax2 = plt.subplots(nrows = 2, ncols = 1, figsize = figsize, constrained_layout = True, sharex = True)

        plot = ax2[0].imshow(activity_matrix2, aspect = 'auto', cmap = 'bwr', norm = utils.get_two_slope_norm(activity_matrix2, 1))
        plt.colorbar(mappable = plot, ax = ax2[0], label = 'Firing rate (Hz)', shrink = 1)

        plot = ax2[1].imshow(sd_matrix2, aspect = 'auto', cmap = 'bwr', norm = utils.get_two_slope_norm(sd_matrix2, 1))
        plt.colorbar(mappable = plot, ax = ax2[1], label = 'Standard deviation of firing rate (Hz)', shrink = 1)

        for t in epoch_start_timepoints:
            ax2[0].plot([t, t], [0, n_cells], color = 'k')
            ax2[0].plot([t + n_timepoints, t + n_timepoints], [0, n_cells], color = 'k')
            ax2[1].plot([t, t], [0, n_cells], color = 'k')
            ax2[1].plot([t + n_timepoints, t + n_timepoints], [0, n_cells], color = 'k')

        ax2[1].set_xticks(ticks)
        ax2[1].set_xticks(ticks)
        ax2[1].set_xticklabels(labels)
        ax2[1].set_xlabel('Time from go cue (s)')
        ax2[0].set_ylabel('Cell #')
        ax2[0].set_title('Mean across trials')
        ax2[1].set_ylabel('Cell #')
        ax2[1].set_title('Standard deviation across trials')


    # Calculate Pearson's correlation of activity_matrix1 with activity_matrix2
    print('Calculating correlation')
    correlation = np.corrcoef(activity_matrix1, activity_matrix2, rowvar = False)
    correlation = correlation[:2*n_timepoints, 2*n_timepoints:]

    assert(correlation.shape[0] == 2*n_timepoints)
    assert(correlation.shape[1] == 2*n_timepoints)

    # Plot correlation as a heat map
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, constrained_layout = True)
    plot = ax.imshow(correlation, cmap = 'jet', vmin = 0, vmax = 1)

    # Indicate epoch starts
    for t in epoch_start_timepoints:

        if t == 0:
            ls = '-'
        else:
            ls = '--'

        # Vertical lines
        plt.plot([t, t], [0, 2*n_timepoints], color = 'k', linestyle = ls)
        plt.plot([t + n_timepoints, t + n_timepoints], [0, 2*n_timepoints], color = 'k', linestyle = ls)

        # Horizontal lines
        plt.plot([0, 2*n_timepoints], [t, t], color = 'k', linestyle = ls)
        plt.plot([0, 2*n_timepoints], [t + n_timepoints, t + n_timepoints], color = 'k', linestyle = ls)

    ticks = np.concatenate([epoch_start_timepoints[1:], [t + n_timepoints for t in epoch_start_timepoints[1:]]])
    labels = np.concatenate([np.round([tvec[t] for t in epoch_start_timepoints[1:]], 2),
                                np.round([tvec[t] for t in epoch_start_timepoints[1:]], 2)])
    plt.xticks(ticks, labels = labels)
    plt.yticks(ticks, labels = labels)

    text_locs_left = [(epoch_start_timepoints[i] + epoch_end_timepoints[i])/2 for i in [1, 2, 3]]
    text_locs_right = [l + n_timepoints for l in text_locs_left]

    for [s, d, r] in [text_locs_left, text_locs_right]:
        plt.text(s, sdr_loc, 'S')
        #plt.text(sdr_loc, s, '')
        plt.text(d, sdr_loc, 'D')
        #plt.text(sdr_loc, d, 'D')
        plt.text(r, sdr_loc, 'R')
        #plt.text(sdr_loc, r, 'R')

    plt.text(text_locs_left[0], trial_type_loc, 'Left Trials')
    #plt.text(trial_type_loc, text_locs_left[0], 'Left Trials')
    plt.text(text_locs_right[0], trial_type_loc, 'Right Trials')
    #plt.text(trial_type_loc, text_locs_right[0], 'Right Trials')

    plt.xlabel('Time from go cue (s)')
    plt.ylabel('Time from go cue (s)')

    plt.xlim([0, 2*n_timepoints])
    plt.ylim([2*n_timepoints, 0])

    plt.colorbar(mappable = plot, ax = ax, label = 'Pearson\'s correlation of\npopulation activity vector', shrink = 0.5)

    if savefig:
        plt.savefig('{0}{1}correlation_vs_time.png'.format(save_path, sep))
