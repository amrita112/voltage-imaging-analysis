import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from tqdm import tqdm

def hier_clust(distances, n_clusters):
    ward = AgglomerativeClustering(n_clusters = n_clusters)
    ward.fit(distances)
    return ward

def k_means_clust(vectors, n_clusters):

    kmeans = KMeans(n_clusters = n_clusters).fit(vectors)
    return kmeans.labels_

def norm_vectors(vectors, method = 'z-score'):
    """ Normalize a set of vectors such as PSTH. Each row is one cell and each column is a time point.
        Normalized vector is calculated by subtracting the mean across cells (rows) and dividing by the standard deviation across cells (rows).
    """
    if method == 'z-score':
        mean_vector = np.mean(vectors, axis = 1)
        assert(len(mean_vector) == vectors.shape[0])
        mean_vector = np.reshape(mean_vector, [len(mean_vector), 1])
        std_vector = np.std(vectors, axis = 1)
        assert(len(std_vector) == vectors.shape[0])
        std_vector = np.reshape(std_vector, [len(std_vector), 1])
        return np.divide(vectors - mean_vector, std_vector)
    else:
        print('Method must be \'z-score\'')
        return

def select_n_clusters(vectors, max_n_clust, min_n_clust = 2, n_iter = 10):

    scores = np.zeros([n_iter, max_n_clust - min_n_clust])
    print('Running k-means clustering: {0} iterations'.format(n_iter))
    for iter in range(n_iter):
        i = 0
        print('Iteration {0} of {1}'.format(iter + 1, n_iter))
        for n_clusters in tqdm(range(min_n_clust, max_n_clust)):
            kmeans = KMeans(n_clusters = n_clusters, random_state = i).fit(vectors)
            scores[iter, i] = kmeans.score(vectors)
            i += 1

    return scores

def order_clusters_by_time_of_peak(vectors, n_clust):

    cluster_labels = k_means_clust(vectors, n_clust)

    n_frames = vectors.shape[1]

    clust_vectors = np.zeros([n_clust, n_frames])
    for clust in range(n_clust):

        clust_vectors[clust, :] = np.mean(vectors[np.where(cluster_labels == clust)[0], :], axis = 0)

    peak_activity = np.argmax(clust_vectors, axis = 1)
    assert(len(peak_activity) == n_clust)
    clust_order = np.argsort(peak_activity)

    cell_order_clust_peak = []
    cb = []
    n_cells_count = 0

    for clust in range(n_clust):
        clust_id = clust_order[clust]
        cells = np.where(cluster_labels == clust_id)[0]
        cell_order_clust_peak = np.append(cell_order_clust_peak, cells)
        n_cells_count += len(cells)
        cb = np.append(cb, n_cells_count)

    return {'cell_order': cell_order_clust_peak.astype(int),
            'cluster_boundaries': cb}
