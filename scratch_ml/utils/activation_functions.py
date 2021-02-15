import numpy as np


class Sigmoid():
    def __call__(self, x):   
        x = x.astype(float)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        x = x.astype(float)
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)