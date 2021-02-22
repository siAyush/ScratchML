import numpy as np


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
