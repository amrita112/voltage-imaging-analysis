import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams

from os.path import sep
import os
import pickle as pkl
import numpy as np
from natsort import natsorted

from pre_processing import trial_tiff_stacks

def register(data_path, metadata_file, overwrite = False, batch_size_frames = 50000):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    try:
        mmap_filenames = metadata['mmap_filenames']
        print('Found registered mmap filenames')
    except:
        print('Could not find registered mmap filenames. Overwriting')
        overwrite = True

    if overwrite:

        mmap_filenames = {}
        batch_data = {}

        # Get estimated frame rate
        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        frame_rate = output['frame_and_trial_times']['frame_rate']

        # Get tiff stack info
        trial_tiff_image_path = metadata['trial_tiff_image_path']
        n_frames_per_trial = trial_tiff_stacks.get_n_frames_per_trial(data_path, metadata_file)

        sessions_to_process = metadata['sessions_to_process']
        for session in sessions_to_process:

            print('Session {0}'.format(session))
            mmap_filenames[session] = {}

            # Setup parameters for data and motion correction
            fnames = os.listdir(trial_tiff_image_path[session])
            fnames = natsorted(fnames)
            fnames = ['{0}{1}{2}'.format(trial_tiff_image_path[session], sep, fname) for fname in fnames if fname.endswith('.tif')]

            # Create batches of trials
            nf_cum = np.cumsum(n_frames_per_trial[session])
            nf_tot = nf_cum[-1]
            n_batches = int(np.round(nf_tot/batch_size_frames, decimals = 0))
            if n_batches < 1:
                n_batches = 1
            n_frames_per_batch = int(np.ceil(nf_tot/n_batches))
            first_trials = np.zeros(n_batches)
            last_trials = np.zeros(n_batches)
            for batch in range(n_batches - 1):
                last_trials[batch] = np.where(nf_cum >= n_frames_per_batch*(batch + 1))[0][0]
                first_trials[batch + 1] = last_trials[batch]
            last_trials[n_batches - 1] = len(n_frames_per_trial[session])

            for batch in range(n_batches):
                print('Batch {0} of {1}'.format(batch + 1, n_batches))
                opts_dict = {
                    'fnames': fnames[int(first_trials[batch]):int(last_trials[batch])],
                    'fr': frame_rate[session],                      # sample rate of the movie
                    'index': None,
                    'ROIs': None,
                    'weights': None,
                    'pw_rigid': False,                              # flag for pw-rigid motion correction
                    'max_shifts': (5, 5),                           # maximum allowed rigid shift
                    'gSig_filt': (3, 3),                            # size of filter, in general gSig (see below),
                    'strides': (48, 48),                            # start a new patch for pw-rigid motion correction every x pixels
                    'overlaps': (24, 24),                           # overlap between pathes (size of patch strides+overlaps)
                    'max_deviation_rigid': 3,                       # maximum deviation allowed for patch with respect to rigid shifts
                    'border_nan': 'copy'
                }
                opts = volparams(params_dict=opts_dict)

                # Motion correction
                try:
                    dview.terminate()
                except:
                    pass
                try:
                    c, dview, n_processes = cm.cluster.setup_cluster(
                        backend='local', n_processes=None, single_thread=False)
                except:
                    c, dview, n_processes = cm.cluster.setup_cluster(
                        backend='local', n_processes=None, single_thread=False)

                # Create a motion correction object with the specified parameters
                mc = MotionCorrect(fnames[int(first_trials[batch]):int(last_trials[batch])],
                                        dview=dview, **opts.get_group('motion'))

                # Run piecewise rigid motion correction
                mc.motion_correct(save_movie=True)
                dview.terminate()

                # Memory mapping
                try:
                    dview.terminate()
                except:
                    pass

                c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False)

                border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
                mmap_filenames[session][batch] = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                                           add_to_mov=border_to_0, dview=dview, n_chunks=10)

            batch_data[session] = {
                'n_batches': n_batches,
                'n_frames_per_batch': n_frames_per_batch,
                'first_trials': first_trials,
                'last_trials': last_trials
            }

        metadata['mmap_filenames'] = mmap_filenames
        metadata['batch_data'] = batch_data
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'wb') as f:
            pkl.dump(metadata, f)

    return mmap_filenames
