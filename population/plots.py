import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_spike_psth(spike_psth_array, tvec, ticks, cell_order = [], cluster_boundaries = [], cluster_names = [], save_path = None, save_fig = False, colorbar_label = 'Firing rate (Hz)', ylabel = 'Neuron #', figsize = [12, 12], vmin = None, vmax = None, specify_colorbar_limits = False):

    n_neurons = spike_psth_array.shape[0]
    n_bins = int(spike_psth_array.shape[1]/2)
    if len(cell_order) == 0:
        cell_order = list(range(n_neurons))

    plt.figure(constrained_layout = True, figsize = figsize)
    if specify_colorbar_limits:
        plt.imshow(spike_psth_array[cell_order, :], aspect = 'auto', vmin = vmin, vmax = vmax)
    else:
        plt.imshow(spike_psth_array[cell_order, :], aspect = 'auto')
    cb = plt.colorbar()
    cb.set_label(colorbar_label, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)

    for cb in cluster_boundaries:
        plt.plot(list(range(n_bins*2)), np.ones(n_bins*2)*cb,
                 color = 'white', linewidth = 2, linestyle = '--')
    cluster_boundaries = np.insert(cluster_boundaries, 0, 0)
    for n in range(len(cluster_names)):
        plt.text(-10, (cluster_boundaries[n] + cluster_boundaries[n + 1])/2, cluster_names[n],
                 fontsize = 15, rotation = 'vertical')

    for tick in ticks:
        plt.plot(np.ones(n_neurons)*tick, list(range(n_neurons)),
                 color = 'white', linewidth = 2)
    labels = tvec[ticks]
    plt.xticks(ticks = ticks, labels = np.round(labels, 2), fontsize = 18)
    plt.xlabel('Time from go cue (s)', fontsize = 20)

    plt.text(n_bins/3, -30, 'Lick left trials', fontsize = 20)
    plt.text(n_bins + n_bins/3, -30, 'Lick right trials', fontsize = 20)

    x_ps1 = ticks[0] + 0.1*(ticks[1] - ticks[0])
    x_ps2 = x_ps1 + n_bins
    x_s1 = ticks[1] + 0.1*(ticks[2] - ticks[1])
    x_s2 = x_s1 + n_bins
    x_d1 = ticks[2] + 0.1*(ticks[3] - ticks[2])
    x_d2 = x_d1 + n_bins
    x_r1 = ticks[3] + 0.1*(ticks[4] - ticks[3])
    x_r2 = x_r1 + n_bins

    #plt.text(x_ps1, -2, 'Pre-sample')
    #plt.text(x_ps2, -2, 'Pre-sample')
    plt.text(x_s1, -2, 'S', fontsize = 18)
    plt.text(x_s2, -2, 'S', fontsize = 18)
    plt.text(x_d1, -2, 'D', fontsize = 18)
    plt.text(x_d2, -2, 'D', fontsize = 18)
    plt.text(x_r1, -2, 'R', fontsize = 18)
    plt.text(x_r2, -2, 'R', fontsize = 18)

    font = {'family' : 'normal',
        'size'   : 20}

    matplotlib.rc('font', **font)

    if save_fig:
        plt.savefig(save_path)

def plot_single_cell_spike_psth(psth_cell_left_corr, psth_cell_right_corr, left_corr_trial_nos, right_corr_trial_nos, tvec, ):

    fig, ax = plt.subplots(nrows = 2, ncols = 1, constrained_layout = True, sharex = True, figsize = [8, 10])
    ax[1].set_xlabel('Time from go cue (s)')
    ax[0].set_ylabel('Trial # (correct trials only)')
    ax[1].set_ylabel('Firing rate (Hz)')

    n_trials_left = psth_cell_left_corr.shape[1]
    assert(len(left_corr_trial_nos) == n_trials_left)
    n_trials_right = psth_cell_right_corr.shape[1]
    assert(len(right_corr_trial_nos) == n_trials_right)
    for trial in range(n_trials_left):
        ax[0].scatter(tvec, left_corr_trial_nos[trial]*np.ones(len(psth_cell_left_corr[:, trial])),
                                marker = '.', color = color_spikes)

    # Plot PSTH for each cell
    ax[1].plot(tvec_trial, psth[cell]['left_corr']['mean'], color = 'r', linewidth = 1.2)
    ax[1].fill_between(tvec_trial, psth[cell]['left_corr']['mean'] - psth[cell]['left_corr']['sem'], psth[cell]['left_corr']['mean'] + psth[cell]['left_corr']['sem'],
                        color = 'r', alpha = 0.2, linewidth = 0)
    ax[1].plot(tvec_trial, psth[cell]['left_inc']['mean'], color = 'lightcoral', linewidth = 0.8)
    ax[1].fill_between(tvec_trial, psth[cell]['left_inc']['mean'] - psth[cell]['left_inc']['sem'], psth[cell]['left_inc']['mean'] + psth[cell]['left_inc']['sem'],
                        color = 'lightcoral', alpha = 0.2, linewidth = 0)
    ax[1].plot(tvec_trial, psth[cell]['right_corr']['mean'], color = 'b', linewidth = 1.2)
    ax[1].fill_between(tvec_trial, psth[cell]['right_corr']['mean'] - psth[cell]['right_corr']['sem'], psth[cell]['right_corr']['mean'] + psth[cell]['right_corr']['sem'],
                        color = 'b', alpha = 0.2, linewidth = 0)
    ax[1].plot(tvec_trial, psth[cell]['right_inc']['mean'], color = 'cornflowerblue', linewidth = 0.8)
    ax[1].fill_between(tvec_trial, psth[cell]['right_inc']['mean'] - psth[cell]['right_inc']['sem'], psth[cell]['right_inc']['mean'] + psth[cell]['right_inc']['sem'],
                        color = 'cornflowerblue', alpha = 0.2, linewidth = 0)

    # Plot dashed line to show sample end time and go cue time
    [y0, y1] = ax[0].get_ylim()
    ax[0].plot(np.ones(10)*(sample_end_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 0.5, color = 'gray')
    ax[0].plot(np.ones(10)*(sample_start_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 0.5, color = 'gray')
    ax[0].plot(np.zeros(10), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 0.5, color = 'gray')

    [y0, y1] = ax[1].get_ylim()
    ax[1].plot(np.ones(10)*(sample_end_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 0.5, color = 'gray')
    ax[1].plot(np.ones(10)*(sample_start_time - go_cue_time), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 0.5, color = 'gray')
    ax[1].plot(np.zeros(10), np.linspace(y0, y1, 10), linestyle = '--', linewidth = 0.5, color = 'gray')
    plt.savefig('{0}{1}Cell_{2}_{3}.png'.format(spike_rasters_path, sep, cell + 1, suffix))

def plot_spike_psth_latency_order(spike_psth_array_order, spike_psth_array_display, tvec, ticks, trial_type_text = False, save_path = None, save_fig = False, colorbar_label = 'Normalized activity', ylabel = 'Neuron #', figsize = [12, 12]):

    # To compare spike PSTH to excitatory neurons recorded with electrophysiology. See Fig 7A from this paper:
    # Wei Z, Lin BJ, Chen TW, Daie K, Svoboda K, et al. (2020) A comparison of neuronal population dynamics measured with calcium imaging and electrophysiology. PLOS
    # Heatmap of normalized trial-averaged firing rates for right trials (left) and left trials (right) for ephys data.
    # Firing rates were normalized to maximum of activity across both conditions.
    # Neurons were first divided into two groups by their preferred trial type then sorted by latency of peak activity.

    n_neurons = spike_psth_array_order.shape[0]
    n_bins = int(spike_psth_array_order.shape[1]/2)

    max_per_cell_order = np.max(spike_psth_array_order, axis = 1)
    max_per_cell_display = np.max(spike_psth_array_display, axis = 1)
    assert(len(max_per_cell_order) == n_neurons)
    assert(len(max_per_cell_display) == n_neurons)

    for n in range(n_neurons):
        spike_psth_array_order[n, :] = spike_psth_array_order[n, :]/max_per_cell_order[n]
        spike_psth_array_display[n, :] = spike_psth_array_display[n, :]/max_per_cell_display[n]

    latency = np.argmax(spike_psth_array_order, axis = 1)
    assert(len(latency) == n_neurons)

    cell_order = np.argsort(latency)
    #cell_order = list(range(n_neurons))

    plt.figure(constrained_layout = True, figsize = figsize)
    plt.imshow(spike_psth_array_display[cell_order, :], aspect = 'auto')
    plt.colorbar(label = colorbar_label)
    plt.ylabel(ylabel, fontsize = 15)

    for tick in ticks:
        plt.plot(np.ones(n_neurons)*tick, list(range(n_neurons)),
                 color = 'k', linewidth = 1, linestyle = '--')
    labels = tvec[ticks].astype(int)
    plt.xticks(ticks = ticks, labels = np.round(labels, 2))
    plt.xlabel('Time from go cue (s)', fontsize = 15)

    if trial_type_text:
        plt.text(0, -15, 'Left trials', fontsize = 15)
        plt.text(n_bins, -15, 'Right trials', fontsize = 15)

    x_ps1 = ticks[0] + 0.1*(ticks[1] - ticks[0])
    x_ps2 = x_ps1 + n_bins
    x_s1 = ticks[1] + 0.1*(ticks[2] - ticks[1])
    x_s2 = x_s1 + n_bins
    x_d1 = ticks[2] + 0.1*(ticks[3] - ticks[2])
    x_d2 = x_d1 + n_bins
    x_r1 = ticks[3] + 0.1*(ticks[4] - ticks[3])
    x_r2 = x_r1 + n_bins

    #plt.text(x_ps1, -2, 'Pre-sample')
    #plt.text(x_ps2, -2, 'Pre-sample')
    #plt.text(x_s1, -2, 'Sample')
    #plt.text(x_s2, -2, 'Sample')
    #plt.text(x_d1, -2, 'Delay')
    #plt.text(x_d2, -2, 'Delay')
    #plt.text(x_r1, -2, 'Response')
    #plt.text(x_r2, -2, 'Response')

    if save_fig:
        plt.savefig(save_path)
