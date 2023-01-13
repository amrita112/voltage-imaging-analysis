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
# Normalize a set of vectors such as PSTH. Each row is one cell and each column is a time point.
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
    for iter in tqdm(range(n_iter)):
        i = 0
        for n_clusters in range(min_n_clust, max_n_clust):
            kmeans = KMeans(n_clusters = n_clusters, random_state = i).fit(vectors)
            scores[iter, i] = kmeans.score(vectors)
            i += 1

    return scores
