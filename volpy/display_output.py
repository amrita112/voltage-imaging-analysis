import matplotlib.pyplot as plt
from os.path import sep
import os
import pickle as pkl
import numpy as np

import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from segmentation import get_roi_arrays

def display_output(data_path, metadata_file, volpy_results,
                dff_scalebar_height = 0.1, scalebar_width = 1,
                disp_dFF = True, disp_bleaching = True, disp_spatial_filters = True,
                disp_snr = True, cells_disp = [], cell_order = [], save_path_dff = None):

    print('Making figures')
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)

    plots_path = metadata['plots_path']
    if not os.path.isdir(plots_path):
        os.mkdir(plots_path)

    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    frame_rate = output['frame_and_trial_times']['frame_rate']

    batch_data = metadata['batch_data']
    roi_arrays = get_roi_arrays.get_roi_arrays(data_path, metadata_file)

    sessions_to_process = metadata['sessions_to_process']
    n_cells = roi_arrays[sessions_to_process[0]].shape[0]

    photobleaching_trace = {cell: [] for cell in range(n_cells)}
    full_trace = {cell: [] for cell in range(n_cells)}
    dFF = {cell: [] for cell in range(n_cells)}
    dFF_raw = {cell: [] for cell in range(n_cells)}
    spike_frames = {cell: [] for cell in range(n_cells)}

    spatial_filters_good = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}
    spatial_filters_bad = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}
    spike_shapes_good = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}
    spike_shapes_bad = {cell: {session: {} for session in sessions_to_process} for cell in range(n_cells)}

    n_batches_total = np.sum([batch_data[session]['n_batches'] for session in sessions_to_process])
    snr = np.zeros([n_batches_total, n_cells])
    n_frames_total = 0
    n_batches_cum = 0

    for session in sessions_to_process:
        print('     Session {0}'.format(session))

        n_batches = batch_data[session]['n_batches']
        n_cells = roi_arrays[session].shape[0]

        for batch in range(n_batches):
            print('         Batch {0} of {1}'.format(batch + 1, n_batches))
            estimates = volpy_results[session][batch]['vpy']
            good_spatial_filters = volpy_results['good_spatial_filters'][session][:, batch]
            for cell in range(n_cells):

                photobleaching_trace[cell] = np.append(photobleaching_trace[cell], estimates['F0'][cell])
                full_trace[cell] = np.append(full_trace[cell], estimates['rawROI'][cell]['full_trace'])
                if good_spatial_filters[cell]:

                    dFF[cell] = np.append(dFF[cell], estimates['dFF'][cell])
                    dFF_raw[cell] = np.append(dFF_raw[cell], estimates['rawROI'][cell]['dFF'])
                    spike_frames[cell] = np.append(spike_frames[cell], estimates['spikes'][cell] + n_frames_total)
                    sf_im = estimates['spatial_filter'][cell]
                    indices = np.where(sf_im != 0)
                    xmin = indices[0].min()
                    xmax = indices[0].max() + 1
                    ymin = indices[1].min()
                    ymax = indices[1].max() + 1
                    spatial_filters_good[cell][session][batch] = sf_im[xmin:xmax, ymin:ymax]
                    spike_shapes_good[cell][session][batch] = estimates['templates'][cell]
                    snr[n_batches_cum, cell] = estimates['snr'][cell]
                else:
                    dFF[cell] = np.append(dFF[cell], np.zeros(len(estimates['dFF'][0])))
                    dFF_raw[cell] = np.append(dFF_raw[cell], np.zeros(len(estimates['dFF'][0])))
                    sf_im = estimates['spatial_filter'][cell]
                    indices = np.where(sf_im != 0)
                    xmin = indices[0].min()
                    xmax = indices[0].max() + 1
                    ymin = indices[1].min()
                    ymax = indices[1].max() + 1
                    spatial_filters_bad[cell][session][batch] = sf_im[xmin:xmax, ymin:ymax]
                    spike_shapes_bad[cell][session][batch] = estimates['templates'][cell]

            n_frames_total += len(estimates['dFF'][0])
            n_batches_cum += 1

    for cell in range(n_cells):
        spike_frames[cell] = np.array(spike_frames[cell]).astype(int)
        dFF[cell] = np.array(dFF[cell])

    tvec = np.linspace(0, n_frames_total/frame_rate[session], n_frames_total)

    if len(cells_disp) == 0:
        cells_disp = list(range(n_cells))

    # Display photobleaching estimates
    if disp_bleaching:
        plt.figure()
        plt.plot(tvec, photobleaching_trace[cell], color = 'k', linewidth = 1, label = 'Estimated photobleaching curve')
        plt.plot(tvec, full_trace[cell], color = 'gray', linewidth = 1.5, alpha = 0.5, label = 'Unfiltered trace')
        for cell in range(n_cells):
            plt.plot(tvec, photobleaching_trace[cell], color = 'k', linewidth = 1)
            plt.plot(tvec, full_trace[cell], color = 'gray', linewidth = 1.5, alpha = 0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Mean pixel intensity in cell ROI')
        plt.legend()
        plt.savefig('{0}{1}Photobleaching.png'.format(plots_path, sep))

    # Display dFF and spikes
    if disp_dFF:
        plt.figure(figsize = [n_frames_total/5000, n_cells*2])
        levels = [0]
        if len(cell_order) == 0:
            cell_order = np.argsort(np.mean(snr, axis = 0))
        colors = cm.Greens(np.mean(snr, axis = 0)/max(np.mean(snr, axis = 0)))
        for cell in range(n_cells):
            if cell in cells_disp:

                #plt.plot(tvec, dFF_raw[cell_order[cell]] + levels[cell], color = 'gray', linewidth = 1.5, alpha = 0.5)

                #plt.plot(tvec, dFF[cell_order[cell]] + levels[cell], color = 'k', linewidth = 0.5)
                plt.plot(tvec, dFF[cell_order[cell]] + levels[-1], color = colors[cell_order[cell]], linewidth = 1.3)

                plt.scatter(tvec[spike_frames[cell_order[cell]]],
                              levels[-1] + dFF[cell_order[cell]][spike_frames[cell_order[cell]]],
                                color = 'k', marker = '.')
                if cell < n_cells - 1:
                    levels = np.append(levels, levels[-1] + np.max(dFF[cell_order[cell]]) - np.min(dFF[cell_order[cell + 1]]))

        norm = mpl_colors.Normalize(vmin = np.min(np.mean(snr, axis = 0)), vmax = np.max(np.mean(snr, axis = 0)))
        plt.colorbar(cm.ScalarMappable(cmap = 'Greens', norm = norm), label = 'SNR')

        [x0, x1] = plt.gca().get_xlim()
        dff_bottom = levels[0]
        dff_right = x1 - 10
        dff_left = dff_right - scalebar_width
        dff_scalebar = patches.Rectangle([dff_left, dff_bottom], scalebar_width, dff_scalebar_height, color = 'k',)
        plt.gca().add_patch(dff_scalebar)
        plt.text(dff_right + 1, dff_bottom + dff_scalebar_height/2, '-{0}% \ndF/F'.format(int(dff_scalebar_height*100)))
        plt.xlabel('Time (s)')
        plt.ylabel('Cell # (in order of decreasing SNR)')
        plt.yticks(ticks = levels, labels = [cell_order[cell] + 1 for cell in cells_disp], fontsize = 20)
        if save_path_dff == None:
            save_path_dff = '{0}{1}dFF.png'.format(plots_path, sep)
        plt.savefig(save_path_dff)

    # Display spatial filters and spike shapes
    if disp_spatial_filters:
        n_frames_spike = len(estimates['templates'][0])

        fig_sf_good, ax_sf_good = plt.subplots(nrows = n_batches_total, ncols = n_cells, constrained_layout = True,
                                                figsize = [n_cells*2, n_batches_total*3])
        fig_sf_bad, ax_sf_bad = plt.subplots(nrows = n_batches_total, ncols = n_cells, constrained_layout = True,
                                                figsize = [n_cells*2, n_batches_total*3])
        fig_sp_good, ax_sp_good = plt.subplots(nrows = n_batches_total, ncols = n_cells, constrained_layout = True,
                                              figsize = [n_cells*2, n_batches_total*3])
        fig_sp_bad, ax_sp_bad = plt.subplots(nrows = n_batches_total, ncols = n_cells, constrained_layout = True,
                                            figsize = [n_cells*2, n_batches_total*3])

        for cell in range(n_cells):
            col = cell
            row = 0
            for session in sessions_to_process:
                n_batches = batch_data[session]['n_batches']
                for batch in range(n_batches):
                    spike_shape = list(spike_shapes_good.values())[0]
                    window_length = int((len(spike_shape[session][batch]) - 1)/2)
                    tvec_spikes = np.int64(np.arange(-window_length, window_length + 1, 1))*1000/frame_rate[session]
                    good_spatial_filters = volpy_results['good_spatial_filters'][session][:, batch]
                    if good_spatial_filters[cell]:
                        ax_sf_good[row, col].imshow(spatial_filters_good[cell][session][batch])
                        ax_sp_good[row, col].plot(tvec_spikes, spike_shapes_good[cell][session][batch])
                        ax_sp_good[row, col].set_xlabel('Time from spike (ms)')
                        ax_sp_good[row, col].set_title('Cell {0} \nSession {1} \nBatch {2}'.format(cell + 1, session, batch + 1))
                        ax_sf_good[row, col].set_title('Cell {0} \nSession {1} \nBatch {2}'.format(cell + 1, session, batch + 1))
                    else:
                        ax_sf_bad[row, col].imshow(spatial_filters_bad[cell][session][batch])
                        ax_sp_bad[row, col].plot(tvec_spikes, spike_shapes_bad[cell][session][batch])
                        ax_sp_bad[row, col].set_xlabel('Time from spike (ms)')
                        ax_sp_bad[row, col].set_title('Cell {0} \nSession {1} \nBatch {2}'.format(cell + 1, session, batch + 1))
                        ax_sf_bad[row, col].set_title('Cell {0} \nSession {1} \nBatch {2}'.format(cell + 1, session, batch + 1))

                    ax_sf_good[row, col].axis('off')
                    row += 1
        fig_sf_good.savefig('{0}{1}Good spatial filters.png'.format(plots_path, sep))
        fig_sp_good.savefig('{0}{1}Good spike shapes.png'.format(plots_path, sep))
        fig_sf_bad.savefig('{0}{1}Bad spatial filters.png'.format(plots_path, sep))
        fig_sp_bad.savefig('{0}{1}Bad spike shapes.png'.format(plots_path, sep))

    # Display SNR
    if disp_snr:
        plt.figure()
        plt.imshow(snr)
        plt.colorbar(label = 'SNR')
        plt.xlabel('Cell #')
        plt.ylabel('Trial #')
        plt.xticks(ticks = list(range(n_cells)), labels = list(range(1, n_cells + 1)))
        plt.yticks(ticks = np.linspace(0, n_batches_total, n_batches_total + 1),
                    labels = np.concatenate([[trial for trial in batch_data[session]['first_trials']] for session in sessions_to_process]))
        plt.savefig('{0}{1}SNR.png'.format(plots_path, sep))
