from behavior_responses import spike_rasters

def main(data_path, metadata_file, bin_size_psth_ms = 50, snr_thresh = 5, suffix = ''):

    # Plot spike rasters for all cells - correct and incorrect, left and right trials
    spike_rasters.plot_spike_rasters(data_path, metadata_file, bin_size_ms = bin_size_psth_ms, snr_thresh = snr_thresh, suffix = suffix)
