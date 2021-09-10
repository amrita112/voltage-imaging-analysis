from os.path import sep
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltpath
import time
import json

from caiman.mmapping import save_memmap
import caiman as cm
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY

def main(data_path, sub_ids, cells, cell_folders, movies, hp_freq_pb = 0.1):

    volpy_results = {}
    mmap_filenames = {}
    with open('{0}{1}roi_arrays'.format(data_path, sep), 'rb') as f:
        roi_arrays = pkl.load(f)

    for sid in sub_ids:
        try:
            with open('{0}{1}ANM{2}_vpy_results.pkl'.format(data_path, sep, sid), 'rb') as f:
                volpy_results[sid] = pkl.load(f)
            #with open('{0}{1}ANM{2}_mmap_filenames.pkl'.format(data_path, sep, sid), 'rb') as f:
            #    mmap_filenames[sid] = pkl.load(f)
            print('ANM {0} volpy results loaded'.format(sid))
        except:
            print('ANM {0} volpy results and/or mmap filenames could not be loaded'.format(sid))
            volpy_results[sid] = {cell: {} for cell in cells[sid]}
            mmap_filenames[sid] = {cell: {} for cell in cells[sid]}
            with open('{0}{1}ANM{2}_seg_imgs'.format(data_path, sep, sid), 'rb') as f:
                seg_images = pkl.load(f)
            for cell in cells[sid]:
                print(' Cell {0}: {1} movies'.format(cell, len(movies[sid][cell])))
                for movie in movies[sid][cell]:
                    print('     Movie {0}'.format(movie))

                    # Get ROI array
                    print('         Getting ROI array')
                    roi_array = get_roi_array(roi_arrays[sid][cell][movie], seg_images[cell][movie])

                    # Make mmap file
                    print('         Making mmap file')
                    t0 = time.time()
                    mmap_filename = get_mmap_filename(data_path, sid, movie, cell_folders[sid][cell])
                    mmap_filenames[sid][cell][movie] = mmap_filename
                    print('         {0} sec'.format(np.round(time.time() - t0, decimals = 2)))

                    # Set parameters
                    opts = set_vpy_params(data_path, cell_folders[sid][cell], movie, roi_array, mmap_filename)

                    # Run volpy
                    print('         Running volpy')
                    volpy_results[sid][cell][movie] = run_volpy(opts, hp_freq_pb)

            # Save results
            with open('{0}{1}ANM{2}_vpy_results.pkl'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(volpy_results[sid], f)
            with open('{0}{1}ANM{2}_mmap_filenames.pkl'.format(data_path, sep, sid), 'wb') as f:
                pkl.dump(mmap_filenames[sid], f)
            print('ANM {0} volpy results and mmap filenames saved'.format(sid))

    return volpy_results

def disp_results(data_path, sub_ids, cells, cell_folders, movies, volpy_results, plot_type = 'Heatmap'):

    # SNR for all cells, all movies
    n_movies = np.concatenate([[len(movies[sid][cell]) for cell in cells[sid]] for sid in sub_ids])
    max_n_movies = max(n_movies)
    n_cells = sum([len(cells[sid]) for sid in sub_ids])
    snr = np.zeros([n_cells, max_n_movies])
    xticklabels = []
    cell_no = 0
    for sid in sub_ids:
        for cell in cells[sid]:
            movie_no = 0
            xticklabels = np.append(xticklabels, 'ANM{0} Cell {1}'.format(sid, cell))
            for movie in movies[sid][cell]:
                snr[cell_no, movie_no] = volpy_results[sid][cell][movie]['snr']
                movie_no += 1
            cell_no += 1

    plt.figure()
    if plot_type == 'Heatmap':
        plt.imshow(snr)
        plt.colorbar(label = 'SNR')
        plt.yticks(ticks = np.linspace(0, n_cells - 1, n_cells), labels = xticklabels)
        plt.xticks(ticks = np.linspace(0, max_n_movies - 1, max_n_movies), labels = np.linspace(1, max_n_movies, max_n_movies).astype(int))
        plt.xlabel('Movie #')
    else:
        if plot_type == 'Histogram':
            snr = np.reshape(snr, [-1])
            plt.hist(snr[snr > 0], color = 'k')
            plt.xlabel('Spike SNR')
            plt.ylabel('# of recordings')

    plt.savefig('{0}{1}SNR_all_cells.png'.format(data_path, sep))

def get_roi_array(vertices, seg_image):

    h = seg_image.shape[0]
    w = seg_image.shape[1]
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

    roi_array = np.zeros([1, h, w])

    path = mpltpath.Path(vertices)
    mask = path.contains_points(points)
    mask = np.reshape(mask, [h, w])
    cell_pixels = np.array(np.where(mask))
    roi_array[0, cell_pixels[0, :], cell_pixels[1, :]] = np.ones(cell_pixels.shape[1])

    return roi_array

def get_mmap_filename(data_path, sid, movie, cell_folder):

    try:
        with open('{0}{1}ANM{2}_mmap_filenames.pkl'.format(data_path, sep, sid), 'rb') as f:
            mmap_filenames = pkl.load(f)
        mmap_file = mmap_filenames[cell][movie]
    except:
        movie_files_dir = '{0}{1}{2}{1}{3}{1}movie'.format(data_path, sep, cell_folder, movie)
        movie_files = os.listdir(movie_files_dir)
        movie_files = [file for file in movie_files if file.endswith('.tif')]
        movie_files = sorted(movie_files)
        movie_files = ['{0}{1}{2}'.format(movie_files_dir, sep, file) for file in movie_files]

        mmap_file = save_memmap(movie_files, base_name = None, order = 'C')

    return mmap_file

def set_vpy_params(data_path, cell_folder, movie, roi_array, mmap_filename):

    with open('{0}{1}{2}{1}{3}{1}movie_metadata.json'.format(data_path, sep, cell_folder, movie)) as json_file:
        movie_metadata = json.load(json_file)

    opts_dict = {
        'fnames': mmap_filename,
        'fr': movie_metadata['movie_frame_rate'],
        'index': [0],
        'ROIs': roi_array,
        'weights': None,
        'pw_rigid': False,
        'max_shifts': (5, 5),
        'gSig_filt': (3, 3),
        'strides': (48, 48),
        'overlaps': (24, 24),
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'method': 'SpikePursuit'
    }
    opts = volparams(params_dict=opts_dict)
    return opts

def run_volpy(opts, hp_freq_pb):

    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts,
                    hp_freq_pb = hp_freq_pb
                    )
    vpy.fit(n_processes=n_processes, dview=dview)
    dview.terminate()
    return vpy.estimates
