import numpy as np
import math
import copy
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
        """The number of trainable parameters  parameters(used by the layer."""
        return 0

    def forward_pass(self, x, training):
        """Propogates the data forward in the network."""
        return NotImplementedError()

    def backward_pass(self, gradient):
        """ Propogates the gradient backwards in the network."""
        return NotImplementedError()


class Dense(Layer):

    def __init__(self, n_units, input_shape=None):
        """A fully-connected NN layer."""
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.w = None
        self.w0 = None

    def initialize(self, optimizer):
        # Initialize weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.w = np.random.uniform(-limit, limit,
                                   (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        # Weight optimizers
        self.w_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.w.shape) + np.prod(self.w0.shape)

    def forward_pass(self, x, training=True):
        self.layer_input = x
        return x.dot(self.w) + self.w0

    def backward_pass(self, gradient):
        w = self.w
        if self.trainable:
            grad_w = self.layer_input.T.dot(gradient)
            grad_w0 = np.sum(gradient, axis=0, keepdims=True)
            self.w = self.w0_opt.update(self.w, grad_w)
            self.w0 = self.w_opt.update(self.w0, grad_w0)
        # Return accumulated gradient for next layer
        return gradient.dot(w.T)

    def output_shape(self):
        return (self.n_units, )


class RNN(Layer):
    """Fully Connected RNN layer."""
    pass


class Conv2D(Layer):
    """2D Convolution Layer"""
    pass


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
        return self.activation_func.__class__.__name__

    def forward_pass(self, x, training=True):
        self.layer_input = x
        return self.activation_func(x)

    def backward_pass(self, gradient):
        return gradient * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Dropout(Layer):
    """A layer that randomly sets a fraction p(float) of the output units of the previous layer
    to zero."""

    def __init__(self, p=0.2):
        self.p = p
        self.trainable = True
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True

    def forward_pass(self, x, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=x.shape) > self.p
            c = self._mask
        return x * c

    def backward_pass(self, gradient):
        return gradient * self._mask

    def output_shape(self):
        return self.input_shape


class Flatten(Layer):
    """Turns a multidimensional matrix into two-dimensional."""

    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.trainable = True
        self.prev_shape = None

    def forward_pass(self, x, training=True):
        self.prev_shape = x.shape
        return x.reshape((x.shape[0], -1))

    def backward_pass(self, gradient):
        return gradient.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)


class Reshape(Layer):
    """Reshapes the input tensor into specified shape.
    shape : tuple
        The shape which the input shall be reshaped to.
    """

    def __init__(self, shape, input_shape=None):
        self.input_shape = input_shape
        self.shape = shape
        self.trainable = True
        self.prev_shape = None

    def forward_pass(self, x, training=True):
        self.prev_shape = x.shape
        return x.reshape((x.shape[0], ) + self.shape)

    def backward_pass(self, gradient):
        return gradient.reshape(self.prev_shape)

    def output_shape(self):
        return self.shape


class PoolingLayer(Layer):
    """A parent class of MaxPooling and AveragePooling."""
    pass
