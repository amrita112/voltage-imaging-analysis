# This script calls a bunch of functions from the pre-processing module to get the behavior timings, trial types, and create tiff stacks
# from individual tiff files

from pre_processing import frame_and_trial_times
from pre_processing import trial_tiff_stacks
from pre_processing import view_mean_px_vals
from pre_processing import get_bpod_info

def main(data_path, metadata_file,
         overwrite_frame_and_trial_times = False, overwrite_daq_data = False, overwrite_tiff_metadata = False,
         overwrite_tiff_stacks = False, show_mean_px_vals = False, save_mean_px_val_fig = False,
         overwrite_bpod_data = False

        ):

    # Get frame numbers and corresponding times, trial start times and frames - not including dark frames
    frame_and_trial_times.frame_and_trial_times(data_path, metadata_file,
                                                         overwrite = overwrite_frame_and_trial_times,
                                                         overwrite_daq_data = overwrite_daq_data,
                                                         overwrite_tiff_metadata = overwrite_tiff_metadata)


    # Convert pixel values into photo-electrons and write a single tiff stack for each trial
    trial_tiff_stacks.trial_tiff_stacks(data_path, metadata_file, overwrite_all = overwrite_tiff_stacks)
    if show_mean_px_vals:
        view_mean_px_vals.view_mean_px_vals(data_path, metadata_file, save_fig = save_mean_px_val_fig)

    # Get trial types and within-trial timing data from Bpod (select correct trial numbers for session)
    bpod_data = get_bpod_info.get_bpod_info(data_path, metadata_file, overwrite = overwrite_bpod_data)
