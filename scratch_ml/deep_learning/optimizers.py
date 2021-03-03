import numpy as np


class StochasticGradientDescent():

    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_updt = None

    def update(self, weight, grad_wrt_w):
        # If not initialized
        if self.weight_updt is None:
            self.weight_updt = np.zeros(np.shape(weight))
        self.weight_updt = self.momentum * \
            self.weight_updt + (1 - self.momentum) * grad_wrt_w
        return weight - self.learning_rate * self.weight_updt
