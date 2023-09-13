import numpy as np
from matplotlib.colors import TwoSlopeNorm

def smooth(array, window = 5):

    """
    array: NumPy 1-D array containing the data to be smoothed
    window: smoothing window size needs, which must be odd number. Default is 5, as in the MATLAB function 'smooth'
    """
    assert(np.mod(window, 2) == 1)
    out0 = np.convolve(array, np.ones(window, dtype=int),'valid')/window
    r = np.arange(1, window - 1, 2)
    start = np.cumsum(array[:window - 1])[::2]/r
    stop = (np.cumsum(array[:- window :-1])[::2]/r)[::-1]

    return np.concatenate((  start , out0, stop  ))

def get_two_slope_norm(array, percentile_saturation, medium_value = 0):
    """ Get matplotlib.colors.TwoSlopeNorm object for plotting a heatmap with saturated color values
    array: 2-D array containing data to be plotted as a heatmap
    percentile_saturation: percent of values in array at extremes to be shown as extreme color values
    medium_value: value corresponding to center of color map
    """

    n_values = np.product([array.shape[i] for i in range(len(array.shape))])
    n_saturate = int(percentile_saturation*n_values/100)
    sorted = np.sort(np.reshape(array, [-1]))
    min_val = sorted[n_saturate]
    max_val = sorted[-n_saturate]
    assert(max_val >= min_val)

    if min_val >= medium_value:
        medium_value = (min_val + max_val)/2

    return TwoSlopeNorm(vmin = min_val, vmax = max_val, vcenter = medium_value)
