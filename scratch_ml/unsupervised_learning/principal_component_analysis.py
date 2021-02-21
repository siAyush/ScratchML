import numpy as np
from scratch_ml.utils import covariance_matrix


class PCA():
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis."""

    def transform(self, x, n_components):
        covariance = covariance_matrix(x)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        # sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]
        x_transformed = x.dot(eigenvectors)
        return x_transformed
