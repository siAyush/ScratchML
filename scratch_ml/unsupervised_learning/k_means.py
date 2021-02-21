import numpy as np
from scratch_ml.utils import euclidean_distance


class KMeans():
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters."""

    def __init__(self, k=2, iterations=500):
        self.k = k
        self.iterations = iterations

    def _init_random_centroids(self, x):
        """Initialize the centroids"""
        n_samples, n_features = np.shape(x)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = x[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        """Return the index of the closest centroid to the sample"""
        closest_idx = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_idx = i
                closest_dist = distance
        return closest_idx

    def _create_clusters(self, centroids, x):
        """Assign the samples to the closest centroids to create clusters"""
        clusters = [[] for _ in range(self.k)]
        for sample_idx, sample in enumerate(x):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(sample_idx)
        return clusters

    def _calculate_centroids(self, clusters, x):
        """Calculate new centroids"""
        n_features = np.shape(x)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(x[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def _get_cluster_labels(self, clusters, x):
        """Classify samples as the index of their clusters"""
        y_pred = np.zeros(np.shape(x)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, x):
        centroids = self._init_random_centroids(x)
        for i in range(self.iterations):
            clusters = self._create_clusters(centroids, x)
            prev_centroids = centroids
            centroids = self._calculate_centroids(clusters, x)
            diff = centroids - prev_centroids
            if not diff.any():
                break
        return self._get_cluster_labels(clusters, x)
