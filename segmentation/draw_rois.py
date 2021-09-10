import caiman as cm

from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import napari

from segmentation import segmentation_utils

def draw_rois(data_path, metadata_file, overwrite = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    roi_file = metadata['roi_file']
    try:
        with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
            rois = pkl.load(f)
            print('Rois loaded')
    except:
        print('Could not find rois. Overwriting')
        overwrite = True

    if overwrite:
        # Create segmentation image
        seg_images = make_seg_images(data_path, metadata_file)

        # Create napari window, display segmentation image and enable user to draw rois
        draw_rois_napari(data_path, metadata_file, seg_images)


def make_seg_images(data_path, metadata_file, n_frames_reg = 1000, overwrite = False):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    seg_images = {}

    try:
        for session in sessions_to_process:
            im = Image.open('{0}{1}Session{2}_seg_image_registered.tif'.format(data_path, sep, session))
            seg_images[session] = np.array(im)
        print('Loaded segmentation images')

    except:
        overwrite = True

    if overwrite:
        print('Overwriting segmentation images')

        mmap_filenames = metadata['mmap_filenames']

        for session in sessions_to_process:
            print('Session {0}: Creating segmentation image'.format(session))

            # Load memory mapped file as images
            Yr, dims, T = cm.load_memmap(mmap_filenames[session][0])
            images = np.reshape(Yr.T, [T] + list(dims), order='F')

            seg_image = np.mean(images[:n_frames_reg, :, :], axis = 0)
            plt.figure()
            plt.imshow(seg_image)
            plt.title('Segmentation image for session {0}'.format(session))

            im = Image.fromarray(seg_image)
            im.save('{0}{1}Session{2}_seg_image_registered.tif'.format(data_path, sep, session))

            seg_images[session] = seg_image

    return seg_images


def draw_rois_napari(data_path, metadata_file, seg_images, cell_radius_px = 5):

    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    sessions_to_process = metadata['sessions_to_process']
    rois = {}
    roi_file = metadata['roi_file']
    viewer = napari.Viewer()
    image_layers = {}
    mask_layers = {}
    point_layers = {}
    cell_data = {session: {} for session in sessions_to_process}
    n_cells = {session: 0 for session in sessions_to_process}

    for session in sessions_to_process:

        image_layers[session] = viewer.add_image(seg_images[session], colormap = 'green', name = 'Session {0}_image'.format(session))
        mask_layers[session] = viewer.add_shapes(data = None, shape_type = 'polygon', opacity = 0.2,
                                                    face_color = 'white', edge_color = 'red', edge_width = 3, name = 'Session {0}_masks'.format(session))
        point_layers[session] = viewer.add_points(data = None, name = 'Session {0}_points'.format(session))

    @viewer.bind_key('n', overwrite = True)
    def new_cell(viewer):
        active_layer = viewer.active_layer
        name = active_layer.name
        session = int(name[8])
        cell_id = len(cell_data[session].keys()) + 1
        cell_data[session][cell_id] = {}
        cell_data[session][cell_id]['cell_id'] = cell_id
        cell_data[session][cell_id]['mask'] = None

    @viewer.bind_key('d', overwrite = True)
    def get_mask(viewer):
        active_layer = viewer.active_layer
        name = active_layer.name
        session = int(name[8])
        center = point_layers[session].data[-1] # Last drawn point
        z_plane = point_layers[session].data[-1, 0].astype(int)
        boundary = segmentation_utils.disc_roi(seg_images[session], center, cell_radius_px)
        mask_layers[session].add(boundary, shape_type= 'polygon')

    @viewer.bind_key('m', overwrite = True)
    def add_mask(viewer):
        active_layer = viewer.active_layer
        name = active_layer.name
        session = int(name[8])
        cell_id = len(cell_data[session].keys())
        cell_data[session][cell_id]['mask'] = mask_layers[session].data[-1]

    @viewer.bind_key('w', overwrite = True)
    def save_all(viewer):
        for session in sessions_to_process:
            rois[session] = cell_data[session]
        with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'wb') as f:
                    pkl.dump(rois, f)
