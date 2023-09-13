import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import sep

from population import clustering

def plot_pop_corr(activity_dict, epoch_start_timepoints,
                  trial_types = ['Left', 'Right'], figsize = [10, 10], savefig = False, save_path = None):
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

    # Build two population activity matrices, with the two trial types concatenated, using a randomly selectd half of trials for each.
    activity_matrix1 = np.zeros([n_cells, 2*n_timepoints])
    activity_matrix2 = np.zeros([n_cells, 2*n_timepoints])

    print('Building population activity matrices')
    for cell in tqdm(range(n_cells)):

        # Trial type 1
        n_trials = activity_dict[trial_types[0]][cell].shape[0]

        trial_set1 = np.random.choice(list(range(n_trials)), int(n_trials/2), replace = False)
        trial_set2 = [trial for trial in list(range(n_trials)) if not trial in trial_set1]
        assert(len(np.intersect1d(trial_set1, trial_set2, assume_unique = True)) == 0)

        activity_matrix1[cell, :n_timepoints] = np.mean(activity_dict[trial_types[0]][cell][trial_set1, :], axis = 0)
        activity_matrix2[cell, :n_timepoints] = np.mean(activity_dict[trial_types[0]][cell][trial_set2, :], axis = 0)

        # Trial type 2
        n_trials = activity_dict[trial_types[1]][cell].shape[0]

        trial_set1 = np.random.choice(list(range(n_trials)), int(n_trials/2), replace = False)
        trial_set2 = [trial for trial in list(range(n_trials)) if not trial in trial_set1]
        assert(len(np.intersect1d(trial_set1, trial_set2, assume_unique = True)) == 0)

        activity_matrix1[cell, n_timepoints:] = np.mean(activity_dict[trial_types[1]][cell][trial_set1, :], axis = 0)
        activity_matrix2[cell, n_timepoints:] = np.mean(activity_dict[trial_types[1]][cell][trial_set2, :], axis = 0)


    plt.figure(figsize = [10, 10])
    plt.imshow(activity_matrix1, aspect = 'auto', cmap = 'bwr', vmax = 15)
    for t in epoch_start_timepoints:
        plt.plot([t, t], plt.ylim(), color = 'k', linestyle = '--')
        plt.plot([t + n_timepoints, t + n_timepoints], plt.ylim(), color = 'k', linestyle = '--')
    plt.title('Activity matrix 1')
    plt.colorbar()

    plt.figure(figsize = [10, 10])
    plt.imshow(activity_matrix2, aspect = 'auto', cmap = 'bwr', vmax = 15)
    for t in epoch_start_timepoints:
        plt.plot([t, t], plt.ylim(), color = 'k', linestyle = '--')
        plt.plot([t + n_timepoints, t + n_timepoints], plt.ylim(), color = 'k', linestyle = '--')
    plt.title('Activity matrix 2')
    plt.colorbar()

    plt.figure(figsize = [10, 10])
    plt.imshow(activity_matrix2 - activity_matrix1, aspect = 'auto', cmap = 'bwr')
    for t in epoch_start_timepoints:
        plt.plot([t, t], plt.ylim(), color = 'k', linestyle = '--')
        plt.plot([t + n_timepoints, t + n_timepoints], plt.ylim(), color = 'k', linestyle = '--')
    plt.title('Activity matrix 2 - 1')
    plt.colorbar()

    # Calculate Pearson's correlation of activity_matrix1 with activity_matrix2
    print('Calculating correlation')

    mean1 = np.mean(activity_matrix1, axis=0)
    mean2 = np.mean(activity_matrix2, axis=0)

    # Subtract the means from the matrices
    centered_matrix1 = activity_matrix1 - mean1
    centered_matrix2 = activity_matrix2 - mean2

    # Calculate the sum of squares of the centered matrices
    sum_of_squares1 = np.sum(centered_matrix1 ** 2, axis=0)
    sum_of_squares2 = np.sum(centered_matrix2 ** 2, axis=0)

    # Calculate the dot product of the centered matrices
    #dot_product = np.dot(centered_matrix1.T, centered_matrix2)
    dot_product = np.matmul(centered_matrix1.T, centered_matrix2)

    # Calculate Pearson's correlation coefficient
    correlation = dot_product / np.sqrt(sum_of_squares1 * sum_of_squares2)

    correlation = np.corrcoef(activity_matrix1, activity_matrix2, rowvar = False)
    correlation = correlation[:2*n_timepoints, :2*n_timepoints]

    assert(correlation.shape[0] == 2*n_timepoints)
    assert(correlation.shape[1] == 2*n_timepoints)

    # Plot correlation as a heat map
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, constrained_layout = True)
    plot = ax.imshow(correlation, cmap = 'jet')
    plt.colorbar(mappable = plot, ax = ax, label = 'Pearson\'s correlation of\npopulation activity vector')
