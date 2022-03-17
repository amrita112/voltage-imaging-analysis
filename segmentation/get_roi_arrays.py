from os.path import sep
import pickle as pkl
import numpy as np
import matplotlib.path as mpltpath

from segmentation import draw_rois

def get_roi_arrays(data_path, metadata_file, overwrite = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    roi_array_file = metadata['roi_array_file']
    try:
        with open('{0}{1}{2}'.format(data_path, sep, roi_array_file), 'rb') as f:
            roi_arrays = pkl.load(f)
        #print('ROI arrays loaded')
    except:
        overwrite = True

    if overwrite:
        print('Overwriting ROI arrays')

        # Load ROIs (as vector paths)
        roi_file = metadata['roi_file']
        with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
                rois = pkl.load(f)

        roi_arrays = {}
        seg_images = draw_rois.make_seg_images(data_path, metadata_file)

        sessions_to_process = metadata['sessions_to_process']
        for session in sessions_to_process:

            # Create meshgrid of points
            h = seg_images[session].shape[0]
            w = seg_images[session].shape[1]
            xv = range(w)
            yv = range(h)
            coord_array = np.array(np.meshgrid(xv, yv))
            points = np.zeros([h*w, 2])
            p = 0
            for i in range(h):
                for j in range(w):
                    points[p, 1] = coord_array[0, i, j]
                    points[p, 0] = coord_array[1, i, j]
                    p += 1

            # For each cell roi, find pixels inside boundary
            cell_ids = list(rois[session].keys())
            no_cells = len(cell_ids)
            roi_arrays[session] = np.zeros([no_cells, h, w])

            for cell in range(no_cells):
                cell_id = cell_ids[cell]
                cell_pixels = []
                roi = rois[session][cell_id]
                vertices = roi['mask']
                path = mpltpath.Path(vertices)
                mask = path.contains_points(points)
                mask = np.reshape(mask, [h, w])
                cell_pixels = np.array(np.where(mask))
                roi_arrays[session][cell, cell_pixels[0, :], cell_pixels[1, :]] = np.ones(cell_pixels.shape[1])

        with open('{0}{1}{2}'.format(data_path, sep, roi_array_file), 'wb') as f:
                    pkl.dump(roi_arrays, f)

    return roi_arrays
