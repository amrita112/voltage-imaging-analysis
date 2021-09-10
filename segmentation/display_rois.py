from segmentation import draw_rois
from os.path import sep
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def display_rois(data_path, metadata_file,
                 flip_vertical = False, flip_horizontal = False,
                 scalebar_width_um = 100, scalebar_text = True,
                 roi_color = 'r', roi_width = 1.5,  roi_colors = [],
                 fig_width = 15,
                 save_fig = False, save_path = None, title = 'seg_image_annotated.tif',
                 cells = [], cell_labels = [],
                 show_rois = True,):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    um_per_px = metadata['um_per_px']
    roi_file = metadata['roi_file']
    try:
        with open('{0}{1}{2}'.format(data_path, sep, roi_file), 'rb') as f:
            rois = pkl.load(f)
    except:
        print('ROIs could not be loaded')
        show_rois = False
        save_fig = False

    mmap_filenames = metadata['mmap_filenames']
    seg_images = draw_rois.make_seg_images(data_path, metadata_file, mmap_filenames)
    sessions_to_process = metadata['sessions_to_process']

    for session in sessions_to_process:

        im_array = seg_images[session]
        h = im_array.shape[0]
        w = im_array.shape[1]
        if flip_vertical:
            im_array_temp = np.zeros([h, w])
            for row in range(h):
                im_array_temp[row, :] = im_array[h - row - 1, :]
            im_array = im_array_temp.copy()
        if flip_horizontal:
            im_array_temp = np.zeros([h, w])
            for col in range(w):
                im_array_temp[:, col] = im_array[:, w - col - 1]
            im_array = im_array_temp.copy()
        plt.figure(figsize = (fig_width, fig_width*h/w))
        plt.imshow(im_array, cmap = 'Greys_r')

        # For each cell roi, draw mask boundary
        if show_rois:
            if len(cells) == 0:
                cell_ids = list(rois[session].keys())
            else:
                cell_ids = [cell for cell in list(rois.keys()) if cell - 1 in cells]
            no_cells = len(cell_ids)

            for cell in range(no_cells):
                cell_id = cell_ids[cell]
                roi = rois[session][cell_id]
                if len(cell_labels) == 0:
                    cell_label = cell_id
                else:
                    cell_label = cell_labels[cell]
                vertices = roi['mask']
                if np.sum(vertices == None) > 0:
                    print('Session {0} Cell {1}: mask missing'.format(session, cell_id))
                    continue
                if flip_vertical:
                    vertices[:, 0] = h - vertices[:, 0]
                if flip_horizontal:
                    vertices[:, 1] = w - vertices[:, 1]
                if len(roi_colors) > 0:
                    roi_color = roi_colors[cell]
                plt.plot(vertices[:, 1], vertices[:, 0], color = roi_color, linewidth = roi_width)
                text_x = np.mean(vertices[:, 1]) + w*0.01
                text_y = np.mean(vertices[:, 0])
                plt.text(text_x, text_y, '{0}'.format(cell_label), color = 'w',
                fontsize = 10
                #transform = plt.gca().transAxes
                )

        # Plot scalebar
        scalebar_x = w*0.9
        scalebar_y = h*0.8
        text_x = w*0.9
        text_y = h*0.9
        scalebar_width_px = scalebar_width_um/um_per_px
        x = np.linspace(scalebar_x, scalebar_x + scalebar_width_px, 10)
        y = np.ones(10)*scalebar_y
        plt.plot(x, y, linewidth = 10, color = 'w')
        if scalebar_text:
            plt.text(text_x, text_y, '{0} um'.format(scalebar_width_um), color = 'w')
        plt.title('Session {0}'.format(session))
        plt.axis('off')

        if save_path == None:
            save_path = data_path

        if save_fig:
            plt.savefig('{0}{1}Session{2}_{3}'.format(save_path, sep, session, title))
