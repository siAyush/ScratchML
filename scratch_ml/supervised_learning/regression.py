import numpy as np
import math
from scratch_ml.utils import normalize


class l1_regularization():
    """Regularization for Lasso Regression"""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization():
    """Regularization for Ridge Regression"""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class l1_l2_regularization():
    """ Regularization for Elastic Net Regression """

    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)


class Regression():
    """Base regression class"""

    def __init__(self, n_iterations,   learning_rate):
        self.n_iterations = n_iterations
        self.  learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """Initialize weights"""
        limit = 1 / math.sqrt(n_features)
        self.weight = np.random.uniform(-limit, limit, n_features)

    def fit(self, x, y):
        # adding bias
        x = np.insert(x, 0, 1, axis=1)
        self.initialize_weights(n_features=x.shape[1])
        # gradient descent
        for i in range(self.n_iterations):
            y_pred = x.dot(self.weight)
            mse = np.mean(0.5*(y-y_pred)**2 + self.regularization(self.weight))
            grad_weight = -(y-y_pred).dot(x) + \
                self.regularization.grad(self.weight)
            # update the weights
            self.weight -= self.  learning_rate * grad_weight

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        y_pred = x.dot(self.weight)
        return y_pred


class LinearRegression(Regression):
    """Linear Regression"""

    def __init__(self, n_iterations=1000,   learning_rate=0.01):
        # no regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(
            n_iterations=n_iterations,   learning_rate=learning_rate)


class LassoRegression(Regression):
    """Linear regression model with a  l1 regularization"""

    def __init__(self, reg_factor, n_iterations=1000,   learning_rate=0.01):
        self.regularization = l1_regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations,   learning_rate)


class RidgeRegression(Regression):
    """Linear regression model with a  l2 regularization"""

    def __init__(self, reg_factor, n_iterations=1000,   learning_rate=0.01):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations,   learning_rate)


class ElasticNet(Regression):
    """Regression where a combination of l1 and l2 regularization are used"""

    def __init__(self, reg_factor=0.05, l1_ratio=0.5, n_iterations=1000,   learning_rate=0.01):
        self.regularization = l1_l2_regularization(
            alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations,   learning_rate)
