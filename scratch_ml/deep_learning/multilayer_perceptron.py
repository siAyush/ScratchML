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
    

    def  initialize_weights(self, x, y):
        pass 


    def fit(self, x, y):
        pass


    def predict(self, x):
        pass