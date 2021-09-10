import os
from os.path import sep
import pickle as pkl

from pre_processing import get_tiff_metadata
from pre_processing import get_daq_data
from pre_processing import get_frame_times

# Get frame numbers and corresponding times, trial start times and frames - not including dark frames
def frame_and_trial_times(data_path, metadata_file, overwrite = False, overwrite_daq_data = False, overwrite_tiff_metadata = False,):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    frame_times_file = metadata['frame_times_file']

    # Check if frame and trial times are already saved
    try:
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
            output = pkl.load(f)
            tiff_metadata = output['tiff_metadata']
            daq_data = output['daq_data']
            frame_and_trial_times = output['frame_and_trial_times']
            print('Frame times and trial start frames loaded')
    except:
        print('Could not find frame and trial times, calculating')
        overwrite = True

    if overwrite:

        # Get frame times, inter-frame-interval and trial starts from DAQ data
        daq_data = get_daq_data.get_daq_data(data_path, metadata_file, overwrite = overwrite_daq_data)

        # Get number of tiff files and tiff file timestamps
        tiff_metadata = get_tiff_metadata.get_tiff_metadata(data_path, metadata_file, overwrite = overwrite_tiff_metadata)

        # Check daq data vs tiff metadata and get final frame times, trial start times and trial start frames (not including dark frames)
        frame_times = get_frame_times.get_frame_times(data_path, metadata_file, tiff_metadata, daq_data, overwrite = overwrite)
