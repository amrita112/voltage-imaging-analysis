import os
from os.path import sep
import pickle as pkl
from pywavesurfer import ws
import numpy as np
import matplotlib.pyplot as plt

def get_daq_data(data_path, metadata_file, overwrite = False):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)

    # Check if daq data is already stored
    try:
        daq_data = output['daq_data']
        trial_start_samples = daq_data['trial_start_samples']
        frame_samples = daq_data['frame_samples']
        led_trig = daq_data['led_trig']
        print('DAQ data loaded')
    except:
        overwrite = True
        print('Could not load daq data from {0}. Overwriting'.format(frame_times_file))

    if overwrite:
        daq_data_file = metadata['daq_data_file']
        sessions_to_process = metadata['sessions_to_process'].copy()
        n_sessions = len(sessions_to_process)
        daq_file_paths = metadata['daq_file_paths']
        #daq_sample_rate = metadata['daq_sample_rate']
        trial_start_trig_channel = metadata['trial_start_trig_channel']
        vcam_trig_out_channel = metadata['vcam_trig_out_channel']
        vcam_trig_in_channel = metadata['vcam_trig_in_channel']
        led_trig_channel = metadata['led_trig_channel']
        led_lag_samples = metadata['led_lag_samples']

        trial_start_samples = {}
        frame_samples = {}
        led_trig = {}

        if n_sessions > 1:
            fig, ax = plt.subplots(nrows = n_sessions, ncols = 1)
        else:
            fig, ax = plt.subplots(nrows = 2, ncols = 1)

        for session_no in range(n_sessions):

            session = sessions_to_process[session_no]
            print('Session {0}'.format(session))
            daq_file_path = daq_file_paths[session]
            daq_data_all = ws.loadDataFile('{0}'.format(daq_file_path))

            keys = list(daq_data_all.keys())
            daq_metadata = daq_data_all['header']
            sweeps = [key for key in keys if not key == 'header']
            print('Found DAQ data: {1} sweeps. Using first sweep.\n{0}\n'.format(daq_file_path, len(sweeps)))
            daq_data = daq_data_all[sweeps[0]]

            # Get trial start samples
            if trial_start_trig_channel == 'Analog4':
                trial_start_trigger = daq_data['analogScans'][3, :]
            else:
                print('\'trial_start_trig_channel\' is not Analog4. Change')
            med = (np.max(trial_start_trigger) + np.min(trial_start_trigger))/2
            trial_start_trigger[trial_start_trigger <= med] = np.zeros(np.sum(trial_start_trigger <= med))
            trial_start_trigger[trial_start_trigger > med] = np.ones(np.sum(trial_start_trigger > med))
            d = np.diff(trial_start_trigger)
            trial_start_samples[session] = np.where(d == 1)[0] + 1
            print('Session {0}: {1} trials'.format(session, len(trial_start_samples[session])))
            ax[session_no].plot(trial_start_trigger, label = 'Trial start trigger')

            # Get DAQ samples corresponding to camera frames
            if vcam_trig_out_channel == 'Analog1':
                vcam_trig_out = daq_data['analogScans'][0, :]
            else:
                print('\'vcam_trig_out_channel\' is not Analog1. Change')
            med = np.min(vcam_trig_out) + (np.max(vcam_trig_out) - np.min(vcam_trig_out))*0.8
            vcam_trig_out[vcam_trig_out <= med] = np.zeros(np.sum(vcam_trig_out <= med))
            vcam_trig_out[vcam_trig_out > med] = np.ones(np.sum(vcam_trig_out > med))
            d = np.diff(vcam_trig_out)
            frame_samples[session] = (np.where(d == 1)[0] + 1).astype(int)
            ax[session_no].plot(vcam_trig_out, label = 'Vcam trig out')
            print('Session {0}: {1} camera out triggers'.format(session, len(frame_samples[session])))

            if len(frame_samples[session]) == 0:
                metadata['sessions_to_process'].remove(session)
                with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'wb') as f:
                    pkl.dump(metadata, f)
                print('SESSION {0} WILL NOT BE PROCESSED \nCHANGE SESSIONS TO PROCESS IN METADATA INITIALIZATION CELL'.format(session))

            #if vcam_trig_in_channel == 'Analog3':
            #    vcam_trig_in = daq_data['analogScans'][2, :]
            #else:
            #    print('\'vcam_trig_out_channel\' is not Analog1. Change')
            #med = np.min(vcam_trig_in) + (np.max(vcam_trig_in) - np.min(vcam_trig_in))*0.2
            #vcam_trig_in[vcam_trig_in <= med] = np.zeros(np.sum(vcam_trig_in <= med))
            #vcam_trig_in[vcam_trig_in > med] = np.ones(np.sum(vcam_trig_in > med))
            #d = np.diff(vcam_trig_in)
            #ax[session].plot(vcam_trig_in, label = 'Vcam trig in')

            #if len(frame_samples[session]) == 0:
            #    print('Problem with camera out trigger. Using camera in trigger to calculate frame times')
            #    frame_samples[session] = (np.where(d == 1)[0] + 1).astype(int)


            # Get LED on samples
            if led_trig_channel == 'Digital1':
                led_trig[session] = daq_data['digitalScans'][0, :]
                # Remove first 10ms after onset of LED trigger, and last 10ms before LED offset, to compensate for delay in LED illumination
                diff = np.diff(np.array(led_trig[session]).astype(int))
                onset_samples = np.where(diff == 1)[0]
                for sample in onset_samples:
                    led_trig[session][sample + 1:sample + 1 + led_lag_samples] = np.zeros(led_lag_samples)
                offset_samples = np.where(diff == -1)[0]
                for sample in offset_samples:
                    led_trig[session][sample + 1 - 2*led_lag_samples:sample + 1] = np.zeros(2*led_lag_samples)
            else:
                print('\'led_trig_channel\' is not Digital1. Change')
            ax[session_no].plot(led_trig[session], label = 'Led trigger')

            #n_samples = len(trial_start_trigger)
            #tvec = np.linspace(0, n_samples/daq_sample_rate, n_samples)
            #frame_times[session] = tvec[frame_samples[session]]

            ax[session_no].legend()

        daq_data = {    'trial_start_samples': trial_start_samples,
                        'frame_samples': frame_samples,
                        'led_trig': led_trig,
            }
        output['daq_data'] = daq_data
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'wb') as f:
            pkl.dump(output, f)

    return daq_data
