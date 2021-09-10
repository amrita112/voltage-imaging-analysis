import numpy as np

def get_jitter(tiff_timestamps, daq_frame_times):

    n_tiff_files = len(tiff_timestamps)
    n_daq_frames = len(daq_frame_times)
    final_first_tiff_file = 0
    final_last_tiff_file = n_tiff_files
    final_first_frame_trigger = 0
    final_last_frame_trigger = n_daq_frames
    min_jitter = np.inf

    if n_tiff_files > n_daq_frames:

        # More tiff files than camera frame times: select subset of tiff files
        first_tiff_file = 0
        last_tiff_file = n_daq_frames

        while last_tiff_file < n_tiff_files:
            temp_tiff_timestamps = tiff_timestamps[first_tiff_file:last_tiff_file]
            jitter = np.mean(np.abs(temp_tiff_timestamps - daq_frame_times) - np.abs(temp_tiff_timestamps[0] - daq_frame_times[0]))
            if jitter < min_jitter:
                final_first_tiff_file = first_tiff_file
                final_last_tiff_file = last_tiff_file
                min_jitter = jitter
            first_tiff_file += 1
            last_tiff_file += 1

    else:

        # More camera frame times than tiff files: select subset of camera frame triggers
        first_frame_trigger = 0
        last_frame_trigger = n_tiff_files

        min_jitter = np.inf
        while last_frame_trigger < n_daq_frames:
            temp_daq_frame_times = daq_frame_times[first_frame_trigger:last_frame_trigger]
            jitter = np.mean(np.abs(temp_daq_frame_times - tiff_timestamps)- np.abs(temp_daq_frame_times[0] - tiff_timestamps[0]))
            if jitter < min_jitter:
                final_first_frame_trigger = first_frame_trigger
                final_last_frame_trigger = last_frame_trigger
                min_jitter = jitter
            first_frame_trigger += 1
            last_frame_trigger += 1

    output = {'first_frame_trigger': final_first_frame_trigger,
              'first_tiff_file': final_first_tiff_file,
              'last_frame_trigger': final_last_frame_trigger,
              'last_tiff_file': final_last_tiff_file,
              'jitter': min_jitter}
    return output
