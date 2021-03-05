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
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None  # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)


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

    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))
        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)
        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)
        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return w - self.w_updt
