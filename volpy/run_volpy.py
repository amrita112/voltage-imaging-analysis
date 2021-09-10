from os.path import sep
import pickle as pkl
import numpy as np
import warnings

from segmentation import get_roi_arrays

import caiman as cm
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY

def run_volpy(data_path, metadata_file, overwrite = False,
                hp_freq_pb = 0.1,
                reuse_spatial_filters = True, batch_reuse = 0,
                ):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    volpy_results_file = metadata['volpy_results_file']
    try:
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
            volpy_results = pkl.load(f)
        print('Volpy results loaded')
    except:
        overwrite = True
        volpy_results = {}

    if overwrite:

        print('Overwriting volpy results')

        # Get estimated frame rate
        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_rate = output['frame_and_trial_times']['frame_rate']

        # Get roi arrays
        roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)

        # Get registered frames info
        mmap_filenames = metadata['mmap_filenames']
        batch_data = metadata['batch_data']

        # Suppress warnings
        warnings.filterwarnings(action = 'ignore')

        sessions_to_process = metadata['sessions_to_process']
        for session in sessions_to_process:

            print('Session {0}'.format(session))
            volpy_results[session] = {}
            n_batches = batch_data[session]['n_batches']

            for batch in range(n_batches):

                print('Batch {0} of {1}'.format(batch + 1, n_batches))
                if np.logical_and(batch > 0, reuse_spatial_filters):
                    print('Reusing spatial filters from session {0} batch {1}'.format(session, batch_reuse))
                    weights = []
                    estimates = volpy_results[session][batch_reuse]['vpy']
                    for cell in range(roi_arrays[session].shape[0]):
                        indices = np.where(estimates['weights'][cell] != 0)
                        xmin = indices[0].min()
                        xmax = indices[0].max()
                        ymin = indices[1].min()
                        ymax = indices[1].max()
                        weights.append(estimates['weights'][cell][xmin:xmax + 1, ymin:ymax + 1])
                else:
                    print('Calculating new spatial filters for batch')
                    weights = None

                opts_dict = {
                    'fnames': mmap_filenames[session][batch],
                    'fr': frame_rate[session],
                    'index': list(range(roi_arrays[session].shape[0])),
                    'ROIs': roi_arrays[session],
                    'weights': weights,
                    'pw_rigid': False,
                    'max_shifts': (5, 5),
                    'gSig_filt': (3, 3),
                    'strides': (48, 48),
                    'overlaps': (24, 24),
                    'max_deviation_rigid': 3,
                    'border_nan': 'copy',
                    'method': 'SpikePursuit'
                }

                opts = volparams(params_dict=opts_dict)
                c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)
                vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts,
                                hp_freq_pb = hp_freq_pb
                                )
                vpy.fit(n_processes=n_processes, dview=dview)
                dview.terminate()

                print('Saving')
                first_trial = int(batch_data[session]['first_trials'][batch])
                last_trial = int(batch_data[session]['last_trials'][batch])
                volpy_results[session][batch] = {
                    'vpy': vpy.estimates,
                    'trials': list(range(first_trial, last_trial)),
                    'hp_freq_pb': hp_freq_pb
                }


        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'wb') as f:
            pkl.dump(volpy_results, f)

    return volpy_results
