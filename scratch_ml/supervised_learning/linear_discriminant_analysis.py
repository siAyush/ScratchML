import numpy as np
from scratch_ml.utils import covariance_matrix


class LDA():
    """
    Linear Discriminant Analysis can besides from 
    classification also be used to reduce the dimensionaly
    """

    def __init__(self):
        self.weight = None

    def fit(self, x, y):
        x1 = x[y == 0]
        x2 = x[y == 1]

        cov1 = covariance_matrix(x1)
        cov2 = covariance_matrix(x2)
        cov_total = cov1 + cov2

        mean1 = x1.mean(0)
        mean2 = x2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)
        self.weight = np.linalg.pinv(cov_total).dot(mean_diff)

    def predict(self, x):
        y_pred = []
        for sample in x:
            h = sample.dot(self.weight)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
