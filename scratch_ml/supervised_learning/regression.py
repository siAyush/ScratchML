import numpy as np
import math


class Regression():
    """Base regression class"""
    def __init__(self, n_iterations, learing_rate):
        self.n_iterations = n_iterations
        self.learing_rate = learing_rate
        self.weight = None
    

    def initialize_weights(self, n_features):
        """Initialize weights"""
        limit = 1 / math.sqrt(n_features)
        self.weight = np.random.uniform(-limit, limit, n_features)
    

    def fit(self,x, y):
        pass



    def predict(self, x):
        pass
