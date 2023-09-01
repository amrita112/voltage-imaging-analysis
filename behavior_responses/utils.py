import numpy as np

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
