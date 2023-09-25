from population import population_psth
from behavior_responses import utils

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as logregress
from sklearn.cluster import KMeans

def main(activity_dict, tvec, trial_types, trial_epochs, epoch_start_timepoints, epoch_end_timepoints, min_trials_train = 10, min_trials_test = 5, test_fraction = 2/3, penalty = 'l2', reg_strength = 1, random_state = None, max_iter = 100, plot_train_set = False, plot_reg_coeffs = False, plot_confusion_matrix = False):
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

    conf_matrix = confusion_matrix(model, test_train_set, activity_dict,
                                   trial_types, trial_epochs, epoch_start_timepoints, epoch_end_timepoints,
                                   make_plot = plot_confusion_matrix)

    #accuracy = pred_accuracy(model, test_train_set, spike_counts)
    #print(accuracy)
    return model

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
        ax[1].imshow(np.transpose(X), aspect = 'auto', cmap = 'bwr')
        ax[1].set_xlabel('Sample #')
        ax[1].set_ylabel('Cell #')
        ax[1].set_ylim([n_cells, 0])

    if plot_reg_coeffs:
        plt.figure(constrained_layout = True, figsize = [8, 2])
        coefs = model.coef_
        kmeans = KMeans(n_clusters = n_types*n_epochs).fit(np.transpose(coefs))
        plt.imshow(coefs[:, np.argsort(kmeans.labels_)], aspect = 'auto', cmap = 'bwr', norm = utils.get_two_slope_norm(coefs, 1, medium_value = 0))
        #plt.imshow(coefs, aspect = 'auto', cmap = 'bwr', norm = utils.get_two_slope_norm(coefs, 1, medium_value = 0))
        for type_no in range(n_types):
            for e in range(n_epochs):
                plt.plot([0, n_cells], [e + type_no*n_epochs - 0.5, e + type_no*n_epochs - 0.5], color = 'k', linewidth = 0.5)
        plt.colorbar(label = 'Coefficients')
        plt.xlabel('Neuron #')
        plt.ylabel('Class #')

    return model

def confusion_matrix(model, test_train_set, activity_dict, trial_types, trial_epochs, epoch_start_timepoints, epoch_end_timepoints, make_plot = False):

    cell_ids = list(test_train_set[trial_types[0]]['train'].keys())
    n_cells = len(cell_ids)
    n_features = n_cells
    n_timepoints = activity_dict[trial_types[0]][cell_ids[0]].shape[1]
    n_samples = np.sum([test_train_set[type]['n_test'] for type in trial_types])*n_timepoints # Number of samples for single trial type
    n_types = len(trial_types)
    n_epochs = len(trial_epochs)

    labels = np.zeros(n_timepoints)
    for tp in range(n_timepoints):
        e = np.where(np.array(epoch_start_timepoints) <= tp)[0][-1]
        assert(epoch_end_timepoints[e] > tp)
        labels[tp] = e

    X = np.zeros([n_samples, n_features]) # Testing data
    true_labels = np.zeros(n_samples) # True labels

    total_trials = 0
    for type_no in range(n_types):

        type = trial_types[type_no]

        for trial_no in range(test_train_set[type]['n_test']):
            sample1 = total_trials*n_timepoints
            sample2 = (total_trials + 1)*n_timepoints

            true_labels[sample1:sample2] = labels + type_no*n_epochs

            for cell_no in range(n_cells):
                cell = cell_ids[cell_no]
                X[sample1:sample2, cell_no] = activity_dict[type][cell][test_train_set[type]['test'][cell][trial_no], :]

            total_trials += 1

    pred_labels = model.predict(X)
    assert(len(pred_labels) == n_samples)

    pred_labels = pred_labels.astype(int)
    true_labels = true_labels.astype(int)

    n_classes = n_types*n_epochs
    conf_matrix = np.zeros([n_classes, n_classes])
    for sample in range(n_samples):
        true_label = true_labels[sample]
        pred_label = pred_labels[sample]
        conf_matrix[true_label, pred_label] += 1

    freq_true = [np.sum(true_labels == label) for label in range(n_classes)]
    for label in range(n_classes):
        conf_matrix[label, :] = conf_matrix[label, :]/freq_true[label]

    if make_plot:
        plt.figure(constrained_layout = True)
        plt.imshow(conf_matrix)
        plt.colorbar(label = 'Fraction labeled')
        plt.xlabel('True label')
        plt.ylabel('Predicted label')
        plt.xticks(list(range(n_classes)), labels = np.concatenate([trial_epochs, trial_epochs]))
        plt.yticks(list(range(n_classes)), labels = np.concatenate([trial_epochs, trial_epochs]))
        plt.gca().tick_params(axis='both', which='major', labelsize = 8)
        for type in range(n_types):
            #plt.text(-3, (type + 1)*n_epochs, trial_types[type], rotation = 'vertical')
            plt.text(type*n_epochs, -1, trial_types[type])
            plt.plot([0, n_classes], [(type + 1)*n_epochs, (type + 1)*n_epochs], color = 'w')
            plt.plot([(type + 1)*n_epochs, (type + 1)*n_epochs], [0, n_classes], color = 'w')
    return conf_matrix
