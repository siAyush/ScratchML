import numpy as np 
import progressbar
import math
from scratch_ml.utils import Sigmoid, bar_widget, SquareLoss


class Perceptron():
    """The Perceptron is one of the simplest neural network architectures.
       One layer neural network classifier."""
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.activation_function = activation_function
        self.loss = loss
        self.learning_rate = learning_rate
        self.progressbar = progressbar.ProgressBar(bar_widget)


    def fit(self, x, y):
        n_samples, n_features = np.shape(x)
        n, n_outputs = np.shape(y)
        # Initialize weights
        limit = 1 / math.sqrt(n_features)
        self.weight = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.bias =  np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):
            linear_output = x.dot(self.weight) + self.bias
            y_pred = self.activation_function(linear_output)
            # Calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.gradient(y, y_pred) * self.activation_function.derivative(linear_output)
            grad_wrt_weight = x.T.dot(error_gradient)
            grad_wrt_bias = np.sum(error_gradient, axis=0, keepdims=True)
            self.weight -= self.learning_rate * grad_wrt_weight
            self.bias -= self.learning_rate * grad_wrt_bias


    def predict(self, x):
        y_pred = self.activation_function(x.dot(self.weight) + self.bias)
        return y_pred