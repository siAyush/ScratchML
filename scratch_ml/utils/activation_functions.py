# activation functions
import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1/(1+np.exp(-x))
    
    def derivative(self, x):
        f = self.__call__(x)
        return f * (1-f)