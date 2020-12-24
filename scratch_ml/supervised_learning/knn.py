import numpy as np
from scratch_ml.utils import euclidean_distance
class KNN():
    """ K Nearest Neighbors classifier
    k: The number of closest neighbors"""

    def __init__(self, k=5):
        self.k = k

    def _vote(self, labels):
        count = np.bincount(labels.astype("int"))
        return count.argmax()
    
    def predict(self, x_train, y_train, x_test):
        y_pred = np.empty(x_test.shape[0])
        for i, sample in enumerate(x_test):
            index = np.argsort([euclidean_distance(sample, x) for x in x_train])[:self.k]
            # labels of the K nearest neighboring training samples
            knn = np.array([y_train[i] for i in index])
            y_pred[i] = self._vote(knn)
        return y_pred