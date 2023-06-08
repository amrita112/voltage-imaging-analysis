from os.path import sep
import pickle as pkl
import numpy as np

from segmentation import get_roi_arrays

def perform_quality_control(data_path, metadata_file, volpy_results, volpy_results_file_user = '', overwrite = False):

    if not 'good_cells' in list(volpy_results.keys()):
        print('Could not find quality control results')
        overwrite = True
    else:
        good_cells = volpy_results['good_cells']
        print('Found quality control results')

    if overwrite:
        print('Performing quality control')
        # Select cells with good spatial filters
        good_spatial_filters = get_good_spatial_filters(data_path, metadata_file, volpy_results)

        # Select cells with good temporal filters
        print('   Quality control based on spike template to be implemented')
        # Idea: get ideal spike shape (maybe spike shape of highest SNR neuron in recording or something) and set a threshold on np.linalg.norm(spike shape - ideal spike shape)

        # Select cells with good voltage trace
        print('   Quality control based on voltage trace to be implemented')

        # Get overall good cells (to be edited after adding more quality control)
        good_cells = good_spatial_filters

        # Save QC data
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
            metadata = pkl.load(f)

        volpy_results['good_cells'] = good_cells
        if len(volpy_results_file_user) > 0:
            volpy_results_file = volpy_results_file_user
        else:
            volpy_results_file = metadata['volpy_results_file']
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'wb') as f:
            pkl.dump(volpy_results, f)

    return good_cells


def get_good_spatial_filters(data_path, metadata_file, volpy_results):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    batch_data = metadata['batch_data']
    roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)

    volpy_results['good_spatial_filters'] = {}

    sessions_to_process = metadata['sessions_to_process']
    for session in sessions_to_process:
        print('     Session {0}'.format(session))

        n_batches = batch_data[session]['n_batches']
        n_cells = roi_arrays[session].shape[0]
        volpy_results['good_spatial_filters'][session] = np.zeros([n_cells, n_batches])

        for batch in range(n_batches):
            estimates = volpy_results[session][batch]['vpy']
            volpy_results['good_spatial_filters'][session][:, batch] = estimates['locality']
            n_good_cells = np.sum(volpy_results['good_spatial_filters'][session][:, batch])
            print('         Batch {0}: {1} cells have good spatial filters'.format(batch + 1, n_good_cells))

    volpy_results_file = metadata['volpy_results_file']
    with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'wb') as f:
        pkl.dump(volpy_results, f)

    return volpy_results['good_spatial_filters']
