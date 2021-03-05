import numpy as np
from scratch_ml.utils import activation_functions
from scratch_ml.utils import Sigmoid, ReLU, LeakyReLU, Softmax, TanH


class Layer():
    """Base Layer class."""

    def set_input_shape(self, shape):
        """Sets the shape that the layer expects of the input."""
        self.input_shape = shape

    def output_shape(self):
        """The shape of the output produced by forward_pass."""
        return NotImplementedError()

    def layer_name(self):
        """The name of the layer"""
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer."""
        return 0

    def forward_pass(self, x, training):
        """Propogates the data forward in the network."""
        return NotImplementedError()

    def backward_pass(self, gradient):
        """ Propogates the gradient backwards in the network."""
        return NotImplementedError()


activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
}


class Activation(Layer):
    """A layer that applies an activation operation to the input."""

    def __init__(self, name):
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation %s" % (self.activation_func.__class__.__name__)

    def forward_pass(self, x, training):
        self.layer_input = x
        return self.activation_func(x)

    def backward_pass(self, gradient):
        return gradient * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Dense(Layer):

    def __init__(self, n_units, input_shape=None):
        """A fully-connected NN layer."""
        pass
