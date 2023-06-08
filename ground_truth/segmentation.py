from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import napari
from skimage.measure import profile_line

def make_seg_images(data_path, sub_ids, cells, cell_folders, movies, n_files_seg_img = 2, disp_imgs = False, disp_rois = False):

    seg_images = {}
    for sid in sub_ids:
        try:
            with open('{0}{1}ANM{2}_seg_imgs'.format(data_path, sep, sid), 'rb') as f:
                seg_images[sid] = pkl.load(f)
            print('ANM {0} seg images loaded'.format(sid))
        except:
            print('ANM {0} seg images could not be loaded'.format(sid))
            seg_images[sid] = {cell: {} for cell in cells[sid]}
            for cell in cells[sid]:
                print(' Cell {0}: {1} movies'.format(cell, len(movies[sid][cell])))
                for movie in movies[sid][cell]:
                    print('     Movie {0}'.format(movie))

                    movie_files_dir = '{0}{1}{2}{1}{3}{1}movie'.format(data_path, sep, cell_folders[sid][cell], movie)
                    movie_files = os.listdir(movie_files_dir)
                    movie_files = [file for file in movie_files if file.endswith('.tif')]
                    movie_files = sorted(movie_files)

                    movie_list = list()
                    for movie_file in movie_files[:n_files_seg_img]:
                        movie_list.append(read_tiff(os.path.join(movie_files_dir,movie_file)))
                    movie_raw = np.concatenate(movie_list)
                    seg_images[sid][cell][movie] = np.mean(movie_raw, 0)
            with open('{0}{1}ANM{2}_seg_imgs'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(seg_images[sid], f)

        if disp_imgs:
            n_cells = len(cells[sid])
            max_movies_cell = np.max([len(movies[sid][cell]) for cell in cells[sid]])
            fig, ax = plt.subplots(nrows = n_cells, ncols = max_movies_cell, constrained_layout = True, figsize = [20, 15])
            if disp_rois:
                with open('{0}{1}roi_arrays'.format(data_path, sep), 'rb') as f:
                    roi_arrays = pkl.load(f)

            for row in range(n_cells):
                cell = cells[sid][row]
                for col in range(max_movies_cell):
                    if n_cells < 2:
                        if max_movies_cell < 2:
                            ax_plot = ax
                        else:
                            ax_plot = ax[col]
                    else:
                        if max_movies_cell < 2:
                            ax_plot = ax[row]
                        else:
                            ax_plot = ax[row, col]
                    ax_plot.axis('off')
                    try:
                        movie = movies[sid][cell][col]
                    except:
                        continue
                    ax_plot.imshow(seg_images[sid][cell][movie])
                    ax_plot.set_title('Movie {0}'.format(movie))
                    ax_plot.set_ylabel('Cell {0}'.format(cell))
                    if disp_rois:
                        try:
                            vertices = roi_arrays[sid][cell][movie]
                            ax_plot.plot(vertices[:, 1], vertices[:, 0], color = 'r', linewidth = 1)
                        except KeyError:
                            continue

            fig.suptitle('ANM {0}'.format(sid))
            plt.savefig('{0}{1}ANM{2}_seg_imgs.png'.format(data_path, sep, sid))
    return seg_images

def draw_rois_napari(data_path, sub_ids, cells, movies, seg_images, cell_radius_px = 20, overwrite = False):

    try:
        with open('{0}{1}roi_arrays'.format(data_path, sep), 'rb') as f:
            roi_arrays = pkl.load(f)
            print('ROI arrays loaded')
    except:
        print('Could not load ROI arrays')
        overwrite = True
        roi_arrays = {sid: {cell: {} for cell in cells[sid]} for sid in sub_ids}

    if overwrite:

        viewer = napari.Viewer()
        image_layers = {sid: {cell: {} for cell in cells[sid]} for sid in sub_ids}
        mask_layers = {sid: {cell: {} for cell in cells[sid]} for sid in sub_ids}
        point_layers = {sid: {cell: {} for cell in cells[sid]} for sid in sub_ids}
        for sid in sub_ids:
            print('ANM {0}'.format(sid))
            roi_arrays[sid] = {}
            for cell in cells[sid]:
                movie_idx = 0
                roi_arrays[sid][cell] = {}
                for movie in movies[sid][cell]:
                    image_layers[sid][cell][movie] = viewer.add_image(seg_images[sid][cell][movie], colormap = 'green', name = 'ANM{0}_Cell{1}_{2}'.format(sid, cell, movie_idx))
                    mask_layers[sid][cell][movie] = viewer.add_shapes(data = None, shape_type = 'polygon', opacity = 0.2,
                                                                face_color = 'white', edge_color = 'red', edge_width = 3,
                                                                name = 'ANM{0}_Cell{1}_{2}_masks'.format(sid, cell, movie_idx))
                    point_layers[sid][cell][movie] = viewer.add_points(data = None, name = 'ANM{0}_Cell{1}_{2}_centers'.format(sid, cell, movie_idx))
                    movie_idx += 1

        @viewer.bind_key('n', overwrite = True)
        def new_cell(viewer):
            active_layer = viewer.active_layer
            name = active_layer.name
            sid = int(name[3:9])
            cell = int(name[14])
            movie_idx = int(name[16])
            roi_arrays[sid][cell][movies[sid][cell][movie_idx]] = {}

        @viewer.bind_key('d', overwrite = True)
        def get_mask(viewer):
            active_layer = viewer.active_layer
            name = active_layer.name
            sid = int(name[3:9])
            cell = int(name[14])
            movie_idx = int(name[16])
            center = point_layers[sid][cell][movies[sid][cell][movie_idx]].data[-1] # Last drawn point
            boundary = disc_roi(seg_images[sid][cell][movies[sid][cell][movie_idx]], center, cell_radius_px)
            mask_layers[sid][cell][movies[sid][cell][movie_idx]].add(boundary, shape_type= 'polygon')

        @viewer.bind_key('m', overwrite = True)
        def add_mask(viewer):
            active_layer = viewer.active_layer
            name = active_layer.name
            sid = int(name[3:9])
            cell = int(name[14])
            movie_idx = int(name[16])
            assert(cell in list(roi_arrays[sid].keys()))
            roi_arrays[sid][cell][movies[sid][cell][movie_idx]] = mask_layers[sid][cell][movies[sid][cell][movie_idx]].data[-1]

        @viewer.bind_key('w', overwrite = True)
        def save_all(viewer):
            with open('{0}{1}roi_arrays'.format(data_path, sep), 'wb') as f:
                        pkl.dump(roi_arrays, f)

def read_tiff(path,ROI_coordinates=None, n_images=100000):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)

    images = []
    #dimensions = np.diff(ROI_coordinates.T).T[0]+1
    for i in range(10000000):
        try:
            img.seek(i)
            img.getpixel((1, 1))
            imarray = np.array(img)
            if ROI_coordinates:
                slice_ = imarray[ROI_coordinates[0][1]:ROI_coordinates[1][1]+1,ROI_coordinates[0][0]:ROI_coordinates[1][0]+1]
                images.append(slice_)
            else:
                images.append(imarray)

           # break
        except EOFError:
            # Not enough frames in img
            break
    return np.array(images)

def disc_roi(image, center, cell_radius_px, n_theta = 30, thresh = 0.5):

    full_radius = round(1.3*cell_radius_px)
    thetas = np.linspace(0, 2*np.pi, n_theta)
    [x, y] = center


    line_profiles = np.zeros([full_radius, n_theta])

    r_threshold_cross = np.zeros(n_theta)

    for t in range(n_theta):
        theta = thetas[t]
        f = profile_line(image, (x, y), (x + full_radius*np.cos(theta), y + full_radius*np.sin(theta)))
        r_threshold_cross[t] = find_first_cross(f, thresh)
        line_profiles[:, t] = -1*np.abs(np.array(range(full_radius)) - r_threshold_cross[t]) + full_radius;

    #path = find_path(line_profiles)

    #boundary = np.zeros([2, n_theta])
    #boundary[0, :] = x + np.multiply(path, np.cos(thetas))
    #boundary[1, :] = y + np.multiply(path, np.sin(thetas))
    #boundary = boundary.astype(int)

    angular_dist = np.argmax(line_profiles, axis = 0)
    boundary = np.zeros([n_theta, 2])
    boundary[:, 0] = x + np.multiply(angular_dist, np.cos(thetas))
    boundary[:, 1] = y + np.multiply(angular_dist, np.sin(thetas))

    return boundary

def find_first_cross(line_profile, thresh):

    # Assumes line profile goes from inside cell to outside
    max_val = np.max(line_profile)
    max_idx = np.argmax(line_profile)
    min_val_out = np.min(line_profile[max_idx:]) # Minimum intensity outside cell

    thresh_val = min_val_out + thresh*(max_val - min_val_out)
    r_cross = max_idx + np.where(line_profile[max_idx:] < thresh_val)[0][0]

    return r_cross

def find_path(line_profiles):

    # line_profiles should be of shape radius X n_theta
    radius = line_profiles.shape[0]
    n_theta = line_profiles.shape[1]

    pointer = np.zeros([radius, n_theta])
    value = np.zeros([radius, n_theta])
    value[:, 0] = line_profiles[:, 0]

    for i in range(1, n_theta):
        for j in range(1, radius - 1):

            M = np.max(value[j-1:j+1, i-1])
            ind = np.argmax(value[j-1:j+1, i-1])

            value[j, i] = M + line_profiles[j, i]
            pointer[j,i] = j + ind - 1

    # Second traverse to minimize boundary effect
    pointer = np.zeros([radius, n_theta])
    value[:, 1] = value[:, -1]

    for i in range(1, n_theta):
        for j in range(1, radius - 1):

            M = np.max(value[j-1:j+1, i-1])
            ind = np.argmax(value[j-1:j+1, i-1])

            value[j, i] = M + line_profiles[j, i]
            pointer[j,i] = j + ind - 1

    path = np.zeros(n_theta).astype(int)

    M = np.max(value[:, -1])
    ind = np.argmax(value[:, -1])
    path[-1] = ind

    for j in np.flip(range(1, n_theta)):
        path[j-1] = pointer[path[j], j]

    return path
