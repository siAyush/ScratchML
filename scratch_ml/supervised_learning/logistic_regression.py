import numpy as np 
import math

from scratch_ml.utils.activation_functions import Sigmoid


class LogisticRegression():
    """Logistic Regression"""
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()
        self.weight = None
    

    def _initialize_weight(self,x):
        n_features = np.shape(x)[1]
        range_ = 1/math.sqrt(n_features)
        self.weight = np.random.uniform(-range_, range_, n_features)
    

    def fit(self, x, y, n_iterations=2000):
        self._initialize_weight(x)
        y_pred = self.sigmoid(x.dot(self.weight))
        for i in range(n_iterations):
            self.weight -= self.learning_rate * -(y - y_pred).dot(X)
    

    def predict(self, x):
        y_pred = np.round(self.sigmoid(x.dot(self.weight))).astype(int)
        return y_pred 