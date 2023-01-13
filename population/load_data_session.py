from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data_session(data_path, metadata_file):

    print('     {0}'.format(data_path))

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    n_frames = 0
    try:

        roi_file = metadata['roi_file']
        with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
            rois = pkl.load(f)
        n_rois = len(rois.keys())

        try:
            usable_trials_file = metadata['usable_trials_file']
            with open('{0}{1}{2}'.format(data_path, sep, usable_trials_file), 'rb') as f:
                data = pkl.load(f)
            good_cells = data['good_cells']
            no_usable_trials_file = False
        except:
            no_usable_trials_file = True
            good_cells = np.zeros(n_rois)

        volpy_results_files = metadata['vpy_results_files']
        for file in volpy_results_files:
            with open(file, 'rb') as f:
                data = pkl.load(f)
            estimates = data['vpy'].estimates
            try:
                dFF_cell = estimates['dFF'][0]
            except:
                dFF_cell = estimates['trace'][0]
            try:
                f0_cell = estimates['F0'][0]
                no_f0 = False
            except:
                no_f0 = True
                f0 = None
            n_frames += len(dFF_cell)
            if no_usable_trials_file:
                good_cells += [int(p) for p in estimates['passedLocalityTest']]

        try:
            trial_start_file = metadata['trial_start_file']
            with open('{0}{1}{2}'.format(data_path, sep, trial_start_file), 'rb') as f:
                output = pkl.load(f)
            frame_rate = output['frame_rate']
            frame_times = output['frame_times_led_on']
            no_frame_times_file = False
        except:
            frame_times = []
            frame_rate = metadata['frame_rate']
            no_frame_times_file = True

        if no_frame_times_file:
            frame_times = np.linspace(0, n_frames/frame_rate, n_frames)
        if no_usable_trials_file:
            good_cells = list(np.where(good_cells == len(volpy_results_files))[0])
        n_cells = len(good_cells)
        snr = np.zeros([ n_cells, len(volpy_results_files)])
        dFF = np.zeros([n_cells, n_frames])
        if not no_f0:
            f0 = np.zeros([n_cells, n_frames])
        spike_frames = {cell: [] for cell in good_cells}
        tvec = frame_times[:n_frames]
        dur = frame_times[n_frames - 1]

        last_frame_prev = 0
        file_no = 0
        for file in volpy_results_files:
            with open(file, 'rb') as f:
                data = pkl.load(f)
            estimates = data['vpy'].estimates
            try:
                dFF_cell = estimates['dFF'][0]
            except:
                dFF_cell = estimates['trace'][0]
            if not no_f0:
                f0_cell = estimates['F0'][0]
            last_frame = last_frame_prev + len(dFF_cell)
            first_frame = last_frame_prev

            for cell in good_cells:
                snr[good_cells.index(cell), file_no] = estimates['snr'][cell]
                if not no_f0:
                    f0[good_cells.index(cell), first_frame:last_frame] = estimates['F0'][cell]
                try:
                    spike_frames[cell] = np.append(spike_frames[cell],
                                                            last_frame_prev + estimates['spikes'][cell])
                    dFF[good_cells.index(cell), first_frame:last_frame] = estimates['dFF'][cell]

                except:
                    spike_frames[cell] = np.append(spike_frames[cell],
                                                            last_frame_prev + estimates['spikeTimes'][cell])
                    try:
                        dFF[good_cells.index(cell), first_frame:last_frame] = -estimates['dFF'][cell]
                    except:
                        dFF[good_cells.index(cell), first_frame:last_frame] = estimates['trace'][cell]
            file_no += 1
            last_frame_prev = last_frame


    except:
        sessions_to_process = metadata['sessions_to_process']

        frame_times_file = metadata['frame_times_file']
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
        try:
            frame_times = output['frame_and_trial_times']['frame_times_concat']
        except:
            frame_times = []
            last_frame_time_prev = 0
            for s in sessions_to_process[:-1]:
                frame_times_session = output['frame_and_trial_times']['frame_times'][s]
                frame_times = np.append(frame_times,
                                            last_frame_time_prev + frame_times_session)
                frame_times_next_session = output['frame_and_trial_times']['frame_times'][s + 1]
                last_frame_time_prev = frame_times[-1] - frame_times_next_session[0]
            frame_times = np.append(frame_times,
                                            last_frame_time_prev + frame_times_next_session)

        frame_rate = output['frame_and_trial_times']['frame_rate']
        frame_rate = np.mean(list(frame_rate.values()))
        n_frames = len(frame_times)
        tvec = frame_times
        dur = tvec[-1]

        roi_file = metadata['roi_file']
        try:
            with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
                rois = pkl.load(f)
            n_rois = len(rois[sessions_to_process[0]].keys())
        except:
            with open('{0}{1}roi_array.pkl'.format(data_path, sep), 'rb') as f:
                roi_array = pkl.load(f)
            n_rois = roi_array[sessions_to_process[0]].shape[0]

        volpy_results_file = metadata['volpy_results_file']
        with open('{0}{1}{2}'.format(data_path, sep, volpy_results_file), 'rb') as f:
            volpy_results = pkl.load(f)

        try:

            combined_data = volpy_results['combined_data']
            good_cells = list(combined_data['cells'])
            n_cells = len(good_cells)
            dFF = np.zeros([n_cells, n_frames])
            f0 = np.zeros([n_cells, n_frames])
            for cell in good_cells:
                dFF[good_cells.index(cell), :] = combined_data['dFF'][cell]
                f0[good_cells.index(cell), :] = combined_data['F0'][cell]
            spike_frames = combined_data['spike_frames']
            snr = combined_data['snr']
            tvec = combined_data['tvec']
            dur = tvec[-1]
            print('     Combined data loaded')

        except:

            print('     Combined data could not be loaded')

            batch_data = metadata['batch_data']
            total_batches = np.sum([dict['n_batches'] for dict in list(batch_data.values())])

            good_cells_session = volpy_results['good_cells']
            good_cells = np.zeros(good_cells_session[sessions_to_process[0]].shape[0])
            for s in sessions_to_process:
                for batch in range(batch_data[s]['n_batches']):
                    good_cells = good_cells + 1 - good_cells_session[s][:, batch]
            good_cells = np.where(good_cells == 0)[0] # Cells that are in good cells for all sessions, all batches
            n_cells = len(good_cells)

            snr = np.zeros([n_cells, total_batches])
            dFF = np.zeros([n_cells, n_frames])
            f0 = np.zeros([n_cells, n_frames])
            spike_frames = {cell: [] for cell in good_cells}
            n_frames_total = 0
            n_batches_total = 0

            for s in sessions_to_process:
                n_batches = batch_data[s]['n_batches']
                n_frames_s = 0
                for batch in range(n_batches):
                    estimates = volpy_results[s][batch]['vpy']
                    n_frames_batch = len(estimates['dFF'][0])
                    cell_idx = 0
                    for cell in good_cells:
                        spike_frames_batch = estimates['spikes'][cell]
                        spike_frames[cell] = np.append(spike_frames[cell], spike_frames_batch + n_frames_total + n_frames_s)

                        dFF[cell_idx, n_frames_s + n_frames_total:n_frames_batch + n_frames_s + n_frames_total] = estimates['dFF'][cell]
                        f0[cell_idx, n_frames_s + n_frames_total:n_frames_batch + n_frames_s + n_frames_total] = estimates['F0'][cell]
                        snr[cell_idx, n_batches_total] = estimates['snr'][cell]
                        cell_idx += 1

                    n_batches_total += 1
                    n_frames_s += len(estimates['dFF'][0])
                n_frames_total += n_frames_s
    print('     Average SNR for session = {0}'.format(np.mean(snr)))

    session_data = {}
    session_data['n_cells'] = n_cells
    session_data['good_cells'] = good_cells
    session_data['n_rois'] = n_rois
    session_data['dFF'] = dFF
    session_data['f0'] = f0
    session_data['tvec'] = tvec
    session_data['spike_frames'] = spike_frames
    session_data['n_frames'] = n_frames
    session_data['snr'] = snr
    session_data['dur'] = dur
    session_data['frame_rate'] = frame_rate

    return session_data
