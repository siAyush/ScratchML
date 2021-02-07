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
        closest_dist = float("inf")
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_dist = distance
                closest_idx = i
        return closest_idx
    

    def _create_clusters(self, x, centroids):
        """Assign the samples to the closest centroids to create clusters"""
        clusters = [[] for i in range(self.k)]
        for sample_idx, sample in enumerate(x):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(sample_idx)
        return clusters
    

    def _calculate_centroids(self, x, clusters):
        """Calculate new centroids as the means of the samples in each cluster"""
        n_features = np.shape(x)[1]
        centroids = np.zeros((self.k, n_features))
        for idx, cluster in enumerate(clusters):
            centroid = np.mean(x[cluster], axis=0)
            centroids[idx] = centroid
        return centroids
    

    def _get_cluster_labels(self, x, clusters):
        """Classify samples as the index of their clusters"""
        y_pred = np.zeros(np.shape(x)[0])
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
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