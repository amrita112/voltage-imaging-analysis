from segmentation import registration
from segmentation import draw_rois
from segmentation import display_rois
from segmentation import get_roi_arrays

import numpy as np

def main(data_path, metadata_file,
         overwrite_registered_mmap_filenames = False,
         overwrite_rois = False, overwrite_roi_arrays = False,
         save_fig = True, make_roi_fig = False,
         flip_vertical = False, flip_horizontal = False):

    # Perform motion correction
    registration.register(data_path, metadata_file, overwrite = overwrite_registered_mmap_filenames)

    # Present image for segmentation
    draw_rois.draw_rois(data_path, metadata_file, overwrite = overwrite_rois)

    # Display segmented cells
    if np.logical_and(make_roi_fig, not overwrite_rois):
        # Check that number of rois is equal for all sessions, otherwise throw error
        roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file, overwrite = overwrite_roi_arrays)
        n_cells = [roi_arrays[session].shape[0] for session in list(roi_arrays.keys())]
        if not len(np.unique(n_cells)) == 1:
            print('NUMBER OF CELLS IS NOT EQUAL IN ALL SESSIONS. RE-DRAW ROIS')

        display_rois.display_rois(data_path, metadata_file, save_fig = save_fig, flip_vertical = flip_vertical, flip_horizontal = flip_horizontal)

        return roi_arrays
