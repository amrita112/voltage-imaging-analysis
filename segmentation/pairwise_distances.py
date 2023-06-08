from os.path import sep
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def get_pairwise_distances(data_path, metadata_file, cell_ids = []):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    um_per_px = metadata['um_per_px']
    roi_file = metadata['roi_file']
    with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
        rois = pkl.load(f)

    try:
        sessions = metadata['sessions_to_process']
        rois = rois[sessions[0]]
    except:
        rois = rois

    if len(cell_ids) == 0:
        cell_ids = np.array(rois.keys()) - 1

    n_cells = len(cell_ids)
    x = np.zeros(n_cells)
    y = np.zeros(n_cells)

    for cell in range(n_cells):

        cell_id = int(cell_ids[cell] + 1) # Cell IDs start from 0, roi keys start from 1
        roi = rois[cell_id]
        vertices = roi['mask']
        x[cell] = np.mean(vertices[:, 1])
        y[cell] = np.mean(vertices[:, 0])

    n_pairs = int(n_cells*(n_cells - 1)/2)
    distances = np.zeros(n_pairs)

    pair_no = 0
    for i in range(n_cells):
        for j in range(i):
            distances[pair_no] = np.linalg.norm([x[i] - x[j], y[i] - y[j]])
            pair_no += 1

    distances = distances*um_per_px

    return distances
