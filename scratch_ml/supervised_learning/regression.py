import numpy as np
import math


class Regression():
    """Base regression class"""
    def __init__(self, n_iterations, learing_rate):
        self.n_iterations = n_iterations
        self.learing_rate = learing_rate


    def initialize_weights(self, n_features):
        """Initialize weights"""
        limit = 1 / math.sqrt(n_features)
        self.weight = np.random.uniform(-limit, limit, n_features)
    

    def fit(self,x, y):
        # adding bias 
        x = np.insert(x, 0, 1, axis=1)
        self.initialize_weights(n_features=x.shape[1])
        # gradient descent
        for i in range(self.n_iterations):
            y_pred = x.dot(self.weight)
            mse = np.mean(0.5*(y-y_pred)**2 + self.regularization(self.weight))
            grad_weight = -(y-y_pred).dot(x) + self.regularization.grad(self.weight)
            # update the weights
            self.weight -= self.learing_rate * grad_weight


    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        y_pred = x.dot(self.weight)
        return y_pred


class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learing_rate=0.001):
        # no regularization
        self.regularization = lambda x : 0
        self.regularization.grad = lambda x : 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learing_rate=learing_rate)
    

    def fit(self, x, y):
        super(LinearRegression, self).fit(x, y)