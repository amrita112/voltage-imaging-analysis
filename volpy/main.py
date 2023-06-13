from volpy import run_volpy
from volpy import quality_control
from volpy import combine_sessions
from volpy import display_output
import numpy as np

def main(data_path, metadata_file,
            overwrite_volpy_results = False,
            overwrite_combined_data = False,
            calc_burst_snr = False,
            disp_combined_data = False, show_trial_starts = False,
            disp_spatial_filters = False,
            hp_freq_pb = 0.1, disp_output = False,
            context_size = 40, censor_size = 10, ridge_bg = 0.01,
            simple_bg_sub = True, bg_sub_factor = 0.25,
            exclude_cells_from_background = False, background_cell_censor_size = 8,
            visualize_ROI = False,
            filename_save = ''):

    # Run volpy for all sessions
    volpy_results = run_volpy.run_volpy(data_path, metadata_file, overwrite = overwrite_volpy_results, hp_freq_pb = hp_freq_pb,
                                        context_size = context_size, censor_size = censor_size, ridge_bg = ridge_bg,
                                        simple_bg_sub = simple_bg_sub, bg_sub_factor = bg_sub_factor,
                                        exclude_cells_from_background = exclude_cells_from_background, background_cell_censor_size = background_cell_censor_size,
                                        visualize_ROI = visualize_ROI,
                                        filename_save = filename_save)

    # Perform quality control on volpy results
    good_cells = quality_control.perform_quality_control(data_path, metadata_file, volpy_results, volpy_results_file_user = filename_save)
    for session in good_cells.keys():
        print('Session {0}: {1} good cells'.format(session, np.sum([np.all(good_cells[session][i, :]) for i in range(good_cells[session].shape[0])])))

    # Combine data from sessions
    combine_sessions.combine_sessions(data_path, metadata_file, volpy_results,
                                      overwrite = overwrite_combined_data, make_plot = disp_combined_data,
                                      calc_burst_snr = calc_burst_snr, show_trial_starts = show_trial_starts,
                                      filename_save = filename_save)

    # Display volpy output summary
    if disp_output:
        display_output.display_output(data_path, metadata_file, volpy_results, disp_spatial_filters = disp_spatial_filters)

    return volpy_results
