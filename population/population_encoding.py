from population import population_psth
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as logregress

def main(activity_dict, tvec, trial_types, trial_epochs, epoch_start_timepoints, epoch_end_timepoints, min_trials_train = 10, min_trials_test = 5, test_fraction = 2/3, penalty = 'l2', reg_strength = 1, random_state = None, max_iter = 100, plot_train_set = False, plot_reg_coeffs = False):
    """Test how well trial type and trial epoch are encoded by the population
    """
    trial_types = list(activity_dict.keys())
    n_cells = len(activity_dict[trial_types[0]].keys())

    # Divide trials into test and train set
    n_trials = {type: {cell: activity_dict[type][cell].shape[0] for cell in list(activity_dict[type].keys())} for type in trial_types}
    (test_train_set, cells_keep) = get_test_train(n_trials, trial_types,
                                                    min_trials_train = min_trials_train, min_trials_test = min_trials_test,
                                                    test_fraction = test_fraction)

    # Train model
    print('Training model')
    model = train_model(test_train_set, activity_dict, tvec, cells_keep, trial_types, trial_epochs,
                        epoch_start_timepoints, epoch_end_timepoints,
                        penalty = penalty, reg_strength = reg_strength,
                        random_state = random_state, max_iter = max_iter,
                        plot_reg_coeffs = plot_reg_coeffs, plot_train_set = plot_train_set)

    #accuracy = pred_accuracy(model, test_train_set, spike_counts)
    #print(accuracy)

def get_test_train(n_trials, trial_types, min_trials_train = 10, min_trials_test = 5, test_fraction = 2/3):

    test_train_set = {type: {'test': {}, 'train': {}} for type in trial_types}
    cell_ids = list(n_trials[trial_types[0]].keys())

    for type in trial_types:
        n_trials_type = np.array(list(n_trials[type].values()))
        min_type = np.min(n_trials_type)
        if min_type > min_trials_train + min_trials_test:
            # Every cell has enough trials
            test_train_set[type]['n_trials'] = min_type
        else:
            # Some cells have too few trials
            test_train_set[type]['n_trials'] = min_trials_train + min_trials_test

        test_train_set[type]['n_train'] = int(test_train_set[type]['n_trials']*test_fraction)
        test_train_set[type]['n_test'] = test_train_set[type]['n_trials'] - test_train_set[type]['n_train']

    cells_keep = [cell for cell in cell_ids if np.all([n_trials[type][cell] >= test_train_set[type]['n_trials'] for type in trial_types])]

    print('Training model using')
    for type in trial_types:
        print('     {0} {1} trials'.format(test_train_set[type]['n_train'], type))

    print('Testing model using')
    for type in trial_types:
        print('     {0} {1} trials'.format(test_train_set[type]['n_test'], type))

    print('for {0} cells'.format(len(cells_keep)))

    for cell in cells_keep:

        for type in trial_types:
            # Choose trials for training set
            n_trials_type = n_trials[type][cell]
            train = np.random.choice(list(range(n_trials_type)), test_train_set[type]['n_train'], replace = False)
            test = [trial for trial in list(range(n_trials_type)) if not trial in train]
            assert(len(train) + len(test) == n_trials_type)

            # Choose trials for test set
            test = np.random.choice(test, test_train_set[type]['n_test'], replace = False)

            test_train_set[type]['test'][cell] = test
            test_train_set[type]['train'][cell] = train

    return (test_train_set, cells_keep)

def train_model(test_train_set, activity_dict, tvec, cells_keep, trial_types, trial_epochs, epoch_start_timepoints, epoch_end_timepoints, penalty = 'l2', dual = False, solver = 'lbfgs', reg_strength = 1, random_state = None, max_iter = 100, plot_reg_coeffs = False, plot_train_set = False,):

    cell_ids = list(test_train_set[trial_types[0]]['train'].keys())
    n_cells = len(cell_ids)
    n_features = n_cells
    n_timepoints = activity_dict[trial_types[0]][cell_ids[0]].shape[1]
    n_samples = np.sum([test_train_set[type]['n_train'] for type in trial_types])*n_timepoints # Number of samples for single trial type
    assert(len(tvec) == n_timepoints)
    n_types = len(trial_types)
    n_epochs = len(epoch_start_timepoints)

    labels = np.zeros(n_timepoints)
    for tp in range(n_timepoints):
        e = np.where(np.array(epoch_start_timepoints) <= tp)[0][-1]
        assert(epoch_end_timepoints[e] > tp)
        labels[tp] = e

    X = np.zeros([n_samples, n_features]) # Training data
    y = np.zeros(n_samples) # Training labels (trial type + epoch)

    total_trials = 0
    for type_no in range(n_types):

        type = trial_types[type_no]

        for trial_no in range(test_train_set[type]['n_train']):
            sample1 = total_trials*n_timepoints
            sample2 = (total_trials + 1)*n_timepoints

            # Training labels
            y[sample1:sample2] = labels + type_no*n_epochs

            # Training data
            for cell_no in range(n_cells):
                cell = cell_ids[cell_no]
                X[sample1:sample2, cell_no] = activity_dict[type][cell][test_train_set[type]['train'][cell][trial_no], :]

            total_trials += 1

    model = logregress(penalty = penalty, dual = dual, C = reg_strength, random_state = random_state, multi_class = 'multinomial', solver = solver, max_iter = max_iter)
    model.fit(X, y)

    if plot_train_set:
        fig, ax = plt.subplots(nrows = 2, ncols = 1, constrained_layout = True, sharex = True)
        ax[0].plot(y)
        ax[0].set_ylabel('Training label')
        ax[1].imshow(np.transpose(X), aspect = 'auto')
        ax[1].set_xlabel('Sample #')
        ax[1].set_ylabel('Cell #')
        ax[1].set_ylim([n_cells, 0])

    if plot_reg_coeffs:
        plt.figure(constrained_layout = True, figsize = [8, 3])
        plt.imshow(model.coef_, aspect = 'auto', cmap = 'bwr')
        for type_no in range(n_types):
            for e in range(n_epochs):
                plt.plot([0, n_cells], [e + type_no*n_epochs - 0.5, e + type_no*n_epochs - 0.5], color = 'k', linewidth = 0.5)
        plt.colorbar(label = 'Coefficients')
        plt.xlabel('Neuron #')
        plt.ylabel('Class #')

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
