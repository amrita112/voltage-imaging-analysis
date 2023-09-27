import numpy as np
from os.path import sep
import pickle as pkl
import time
from PIL import Image
from skimage.measure import profile_line

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

def load_image_from_tif(image_path):

    im = Image.open(image_path)
    return np.array(im)
