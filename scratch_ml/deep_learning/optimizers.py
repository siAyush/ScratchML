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
        # Use momentum if set
        self.weight_updt = self.momentum * \
            self.weight_updt + (1 - self.momentum) * grad_wrt_w
        return weight - self.learning_rate * self.weight_updt


class Adagrad():
    pass


class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None  # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))
        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.Eg + self.eps)


class Adam():
    pass
