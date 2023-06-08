from population import population_psth
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as linregress

# Test how well trial type and trial epoch are encoded by the population

def main(population_data_path, metadata_file, movies, data_paths, genotype, plot_coeffs = False, min_trials_train = 10, min_trials_test = 5):

    psth = population_psth.get_population_psth(population_data_path, movies, data_paths, metadata_file, genotype,
                                                 make_plot_spike_psth = False, make_plot_dFF_psth = False, plot_psths = False)
    spike_counts = {'left': psth['spikes']['spike_count_trials_left'],
                    'right': psth['spikes']['spike_count_trials_right']}
    cells = list(spike_counts['left'].keys())
    n_trials = {'left': {cell: spike_counts['left'][cell].shape[0] for cell in cells},
                'right': {cell: spike_counts['right'][cell].shape[0] for cell in cells}}

    test_train_set = get_test_train(n_trials, min_trials_train = min_trials_train, min_trials_test = min_trials_test)
    model = build_model(test_train_set, spike_counts, make_plots = plot_coeffs)
    accuracy = pred_accuracy(model, test_train_set, spike_counts)
    print(accuracy)

def get_test_train(n_trials, min_trials_train = 10, min_trials_test = 5):

    cell_ids = list(n_trials['left'].keys())
    test_train_set = {'left': {'test': {}, 'train': {}},
                      'right': {'test': {}, 'train': {}}}

    n_trials_left = np.array(list(n_trials['left'].values()))
    min_left = np.min(n_trials_left)
    if min_left < min_trials_train + min_trials_test:
        min_left = min_trials_train + min_trials_test
        cells_keep = np.where(n_trials_left >= min_left)[0]
    else:
        cells_keep = list(range(len(cell_ids)))
    test_train_set['n_train_left'] = int(min_left*2/3)
    test_train_set['n_test_left'] = min_left - test_train_set['n_train_left']

    n_trials_right = np.array(list(n_trials['right'].values()))
    min_right = np.min(n_trials_right)
    if min_right < min_trials_train + min_trials_test:
        min_right = min_trials_train + min_trials_test
        cells_keep = [cell for cell in cells_keep if n_trials['right'][cell_ids[cell]] >= min_right]
    test_train_set['n_train_right'] = int(min_right*2/3)
    test_train_set['n_test_right'] = min_right - test_train_set['n_train_right']

    print('Building model using {0} left trials and {1} right trials for {2} cells'.format(test_train_set['n_train_left'], test_train_set['n_train_right'], len(cells_keep)))
    print('Testing model using {0} left trials and {1} right trials for {2} cells'.format(test_train_set['n_test_left'], test_train_set['n_test_right'], len(cells_keep)))

    for cell in cells_keep:

        cell_id = cell_ids[cell]
        n_left_trials = n_trials['left'][cell_id]
        train = np.random.choice(list(range(n_left_trials)), test_train_set['n_train_left'], replace = False)
        test = [trial for trial in list(range(n_left_trials)) if not trial in train]
        assert(len(train) + len(test) == n_left_trials)

        test = np.random.choice(test, test_train_set['n_test_left'], replace = False)
        test_train_set['left']['test'][cell_id] = test
        test_train_set['left']['train'][cell_id] = train

        n_right_trials = n_trials['right'][cell_id]
        train = np.random.choice(list(range(n_right_trials)), test_train_set['n_train_right'], replace = False)
        test = [trial for trial in list(range(n_right_trials)) if not trial in train]
        assert(len(test) + len(train) == n_right_trials)

        test = np.random.choice(test, test_train_set['n_test_right'], replace = False)
        test_train_set['right']['test'][cell_id] = test
        test_train_set['right']['train'][cell_id] = train

    return test_train_set

def build_model(test_train_set, spike_counts, make_plots = False):

    n_left_samples = 4*test_train_set['n_train_left']
    n_right_samples = 4*test_train_set['n_train_right']
    n_samples = n_left_samples + n_right_samples
    cell_ids = list(test_train_set['left']['train'].keys())
    n_cells = len(cell_ids)

    X = np.zeros([n_samples, n_cells]) # Training data
    y = np.zeros(n_samples) # Training labels (trial type + epoch)

    for trial_no in range(test_train_set['n_train_left']):
        for cell in range(n_cells):

            cell_id = cell_ids[cell]
            trial_id = test_train_set['left']['train'][cell_id][trial_no]

            for i in range(4):
                sample_no = trial_no*4 + i
                X[sample_no, cell] = spike_counts['left'][cell_id][trial_id, i]
                y[sample_no] = i

    for trial_no in range(test_train_set['n_train_right']):
        for cell in range(n_cells):

            cell_id = cell_ids[cell]
            trial_id = test_train_set['right']['train'][cell_id][trial_no]

            for i in range(4):
                sample_no = n_left_samples + trial_no*4 + i
                X[sample_no, cell] = spike_counts['right'][cell_id][trial_id, i]
                y[sample_no] = i + 4

    model = linregress()
    model.fit(X, y)

    coefficients = model.coef_
    if make_plots:

        plt.figure()
        plt.plot(coefficients, color = 'k')
        plt.xlabel('Neuron #')
        plt.ylabel('Regression coefficient')

    return model

def pred_accuracy(model, test_train_set, spike_counts, trial_average = True):

    n_left_samples = 4*test_train_set['n_test_left']
    n_right_samples = 4*test_train_set['n_test_right']
    n_samples = n_left_samples + n_right_samples
    cell_ids = list(test_train_set['left']['test'].keys())
    n_cells = len(cell_ids)

    X = np.zeros([n_samples, n_cells]) # Training data
    y = np.zeros(n_samples) # Training labels (trial type + epoch)

    for trial_no in range(test_train_set['n_test_left']):
        for cell in range(n_cells):

            cell_id = cell_ids[cell]
            trial_id = test_train_set['left']['test'][cell_id][trial_no]

            for i in range(4):
                sample_no = trial_no*4 + i
                X[sample_no, cell] = spike_counts['left'][cell_id][trial_id, i]
                y[sample_no] = i

    for trial_no in range(test_train_set['n_test_right']):
        for cell in range(n_cells):

            cell_id = cell_ids[cell]
            trial_id = test_train_set['right']['test'][cell_id][trial_no]

            for i in range(4):
                sample_no = n_left_samples + trial_no*4 + i
                X[sample_no, cell] = spike_counts['right'][cell_id][trial_id, i]
                y[sample_no] = i + 4

    accuracy = model.score(X, y)
    return accuracy
