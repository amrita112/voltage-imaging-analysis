from volpy import run_volpy
from volpy import quality_control
from volpy import combine_sessions
from volpy import display_output
import numpy as np

def main(data_path, metadata_file,
            overwrite_volpy_results = False,
            overwrite_combined_data = False,
            calc_burst_snr = False,
            disp_combined_data = False,
            hp_freq_pb = 0.1, disp_output = True):

    # Run volpy for all sessions
    volpy_results = run_volpy.run_volpy(data_path, metadata_file, overwrite = overwrite_volpy_results, hp_freq_pb = hp_freq_pb)

    # Perform quality control on volpy results
    good_cells = quality_control.perform_quality_control(data_path, metadata_file, volpy_results)
    for session in good_cells.keys():
        print('Session {0}: {1} good cells'.format(session, np.sum([np.all(good_cells[session][i, :]) for i in range(good_cells[session].shape[0])])))

    # Combine data from sessions
    combine_sessions.combine_sessions(data_path, metadata_file, volpy_results, overwrite = overwrite_combined_data, make_plot = disp_combined_data, calc_burst_snr = calc_burst_snr)

    # Display volpy output summary
    if disp_output:
        display_output.display_output(data_path, metadata_file, volpy_results)
