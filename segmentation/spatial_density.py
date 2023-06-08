import numpy as np
from os.path import sep
import pickle as pkl

# Calculate density of ROIs in field of view
def main(data_path, metadata_file, roi_arrays, overwrite = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    um_per_px = metadata['um_per_px']
    sessions_to_process = metadata['sessions_to_process']

    try:
        max_x = metadata['max_x']
        max_y = metadata['max_y']
        min_x = metadata['min_x']
        min_y = metadata['min_y']
        density = metadata['density']
    except:
        overwrite = True

    if overwrite:
        print('Overwriting density')
        max_x = {}
        max_y = {}
        min_x = {}
        min_y = {}
        density = {}
        for session in sessions_to_process:
            max_x[session], max_y[session], min_x[session], min_y[session] = get_max_xy(roi_arrays[session])
            n_rois = roi_arrays[session].shape[0]
            density[session] = get_density(max_x[session], max_y[session], min_x[session], min_y[session], n_rois, um_per_px)

        metadata['max_x'] = max_x
        metadata['max_y'] = max_y
        metadata['min_x'] = min_x
        metadata['min_y'] = min_y
        metadata['density'] = density
        with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'wb') as f:
            pkl.dump(metadata, f)

    return (max_x, max_y, min_x, min_y, density)

def get_max_xy(roi_arrays):

    sum_array = np.sum(roi_arrays, axis = 0)

    sum_x = np.sum(sum_array, axis = 0)
    min_x = np.where(sum_x > 0)[0][0]
    max_x = np.where(sum_x > 0)[0][-1]

    sum_y = np.sum(sum_array, axis = 1)
    min_y = np.where(sum_y > 0)[0][0]
    max_y = np.where(sum_y > 0)[0][-1]

    return (max_x, max_y, min_x, min_y)

def get_density(max_x, max_y, min_x, min_y, n_rois, um_per_px):

    x_ext_um = (max_x - min_x)*um_per_px
    y_ext_um = (max_y - min_y)*um_per_px
    area = x_ext_um*y_ext_um
    density = (n_rois/area)*1000000 # In neurons/nm^2

    return density
