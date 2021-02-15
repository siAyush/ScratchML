import numpy as np 
import math
from scratch_ml.utils import Sigmoid, CrossEntropy, Softmax


class  MultilayerPerceptron():
    """Multilayer Perceptron classifier. A neural network with one hidden layer."""
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
    

    def initialize_weights(self, x, y):
        n_samples, n_features = np.shape(x)
        n, n_outputs = np.shape(y)
        # hidden layer 
        limit   = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 =  np.zeros((1, self.n_hidden))
        # output layer 
        limit   = 1 / math.sqrt(self.n_hidden)
        self.v = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 =  np.zeros((1, n_outputs))


    def fit(self, x, y):
        self.initialize_weights(x, y)
        for i in range(self.n_iterations):
            # forward pass
            hidden_input = x.dot(self.w) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            output_layer_input = hidden_output.dot(self.v) + self.v0
            y_pred = self.output_activation(output_layer_input)

            # backward pass






    def predict(self, x):
        hidden_input = x.dot(self.w) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.v) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred