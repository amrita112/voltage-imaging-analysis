# Goal: estimate dF/F to dVm ratio
# Idea: Find situation where dVm is known/comparable for two cells.
# Spike bursts: similar depolarization should be required for any cell to produce a burst of n spikes.
# Assumptions: 1. Resting Vm is same for all cells, and same over time. 2. hotobleaching is correctly estimated, such that dF/F = 0 means cell is at resting Vm
# If these assumptions are correct, then we can compare the dF/F before the first spike in a burst of n spikes

from subthreshold import isi_dist
from subthreshold import burst_subth

from os.path import sep
import pickle as pkl
import numpy as np

def get_subth_from_bursts(data_path, metadata_file,
            overwrite_isi_data = False,
            plot_isi_dist = False,
            plot_burst_thresh = False,
            overwrite_burst_data = False,
            overwrite_burst_dff = False,
            plot_dff_in_bursts = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
        
    # Plot ISI distribution of all good cells, select cells with bimodal ISI distribution
    # and identify ISI threshold for burst vs non-burst
    isi_data = isi_dist.get_isi_data(data_path, metadata_file, overwrite = overwrite_isi_data, make_plot = plot_isi_dist)
    if plot_burst_thresh:
        isi_dist.plot_burst_thresh(isi_data, metadata['plots_path'])

    # Assign each spike to a burst
    isi_data = burst_subth.get_bursts(data_path, metadata_file, isi_data, overwrite = overwrite_burst_data)

    # Get dF/F at start of burst of n spikes
    isi_data = burst_subth.get_burst_dff(data_path, metadata_file, isi_data, overwrite = overwrite_burst_dff, make_plots = plot_dff_in_bursts)

    if np.any([overwrite_isi_data, overwrite_burst_data, overwrite_burst_dff]):
        isi_data_file = metadata['isi_data_file']
        print('Saving ISI data')
        with open('{0}{1}{2}'.format(data_path, sep, isi_data_file), 'wb') as f:
            pkl.dump(isi_data, f)
