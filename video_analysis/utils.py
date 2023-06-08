import sys
from os.path import sep

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

import pims
from matplotlib import cm
import time
from tqdm import tqdm
from scipy.stats import skew
from scipy.sparse.linalg import eigsh
from facemap import utils
import facemap
from pywavesurfer import ws

def get_video_filenames(video_path):

    filenames = os.listdir(video_path)
    filenames = [file for file in filenames if file.endswith('.avi')]
    filenames = natsorted(filenames)
    filenames = ['{0}{1}{2}'.format(video_path, sep, file) for file in filenames]

    return filenames

def plot_mean_frame(filename):

    video = pims.Video(filename)
    Ly = video.frame_shape[0]
    Lx = video.frame_shape[1]
    nframes = len(video)

    # get subsampled mean across frames
    # grab up to 2000 frames to average over for mean

    nf = min(2000, nframes)

    # load in chunks of up to 200 frames (for speed)
    nt0 = min(200, nframes)
    nsegs = int(np.floor(nf / nt0))

    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    avgframe = np.zeros((Ly, Lx), np.float32)
    avgmotion = np.zeros((Ly, Lx), np.float32)

    ns = 0
    for n in range(nsegs):
        t = tf[n]

        im = np.array(video[t : t + nt0])
        # im is TIME x Ly x Lx x 3 (3 is RGB)
        if im.ndim > 3:
            im = im[:, :, :, 0]
        # convert im to Ly x Lx x TIME
        im = np.transpose(im, (1, 2, 0)).astype(np.float32)

        # most movies have integer values
        # convert to float to average
        im = im.astype(np.float32)

        # add to averages
        avgframe += im.mean(axis=-1)
        immotion = np.abs(np.diff(im, axis=-1))
        avgmotion += immotion.mean(axis=-1)
        ns += 1

    avgframe /= float(ns)
    avgmotion /= float(ns)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(avgframe)
    plt.title("average frame")
    plt.axis("off")
