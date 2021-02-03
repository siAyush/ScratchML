import numpy as np


class Sigmoid():
    def __call__(self, x):   
        x = x.astype(float)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        x = x.astype(float)
        return self.__call__(x) * (1 - self.__call__(x))