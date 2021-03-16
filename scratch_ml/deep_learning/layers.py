import numpy as np
import math
import copy

from numpy.lib.function_base import gradient
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

    def __init__(self, n_units, activation='tanh', bptt=5, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.trainable = True
        self.bptt = bptt  # Backpropagation Through Time = bppt
        self.W = None  # Weight of the previous state
        self.V = None  # Weight of the output
        self.U = None  # Weight of the input

    def initialize(self, optimizer):
        timesteps, input_dim = self.input_shape
        limit = 1 / math.sqrt(input_dim)
        self.U = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W = np.random.uniform(-limit, limit, (self.n_units, self.n_units))
        # Weight optimizers
        self.U_opt = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.U.shape) + np.prod(self.V.shape) + np.prod(self.W.shape)

    def forward_pass(self, x, training=True):
        self.layer_input = x
        batch_size, timesteps, input_dim = x.shape

        # Save these values for use in backprop.
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))
        # Set last time step to zero for calculation of the state_input at time step zero
        self.states[:, -1] = np.zeros((batch_size, self.n_units))

        for t in range(timesteps):
            self.state_input[:, t] = x[:, t].dot(
                self.U.T) + self.states[:, t-1].dot(self.W.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_pass(self, gradient):
        _, timesteps, _ = gradient.shape

        # Variables where we save the accumulated gradient
        grad_U = np.zeros(self.U)
        grad_V = np.zeros(self.V)
        grad_W = np.zeros(self.W)
        accum_grad_next = np.zeros(gradient)

        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += gradient[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = gradient[:, t].dot(
                self.V) * self.activation.gradient(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(
                    self.W) * self.activation.gradient(self.state_input[:, t_-1])

        self.U = self.U_opt.update(self.U, grad_U)
        self.V = self.V_opt.update(self.V, grad_V)
        self.W = self.W_opt.update(self.W, grad_W)

        return accum_grad_next

    def output_shape(self):
        return self.input_shape


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

    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True

    def forward_pass(self, x, training=True):
        self.layer_input = x
        batch_size, channels, height, width = x.shape
        _, out_height, out_width = self.output_shape()
        x = x.reshape(batch_size*channels, 1, height, width)
        x_col = image_to_column(x, self.pool_shape, self.stride, self.padding)
        # MaxPool or AveragePool method
        output = self._pool_forward(x_col)
        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)
        return output

    def backward_pass(self, gradient):
        batch_size, _, _, _ = gradient.shape
        channels, height, width = self.input_shape
        accum_grad = gradient.transpose(2, 3, 0, 1).ravel()
        # MaxPool or AveragePool method
        accum_grad_col = self._pool_backward(accum_grad)
        accum_grad = column_to_image(
            accum_grad_col, (batch_size * channels, 1, height, width), self.pool_shape, self.stride, 0)
        accum_grad = accum_grad.reshape((batch_size,) + self.input_shape)
        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, x_col):
        arg_max = np.argmax(x_col, axis=0).flatten()
        output = x_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, gradient):
        gradient_col = np.zeros((np.prod(self.pool_shape), gradient.size))
        arg_max = self.cache
        gradient_col[arg_max, range(gradient.size)] = gradient
        return gradient_col


class AveragePooling2D(PoolingLayer):
    def _pool_forward(self, x_col):
        output = np.mean(x_col, axis=0)
        return output

    def _pool_backward(self, gradient):
        gradient_col = np.zeros((np.prod(self.pool_shape), gradient.size))
        gradient_col[:, range(gradient.size)] = 1. / \
            gradient_col.shape[0] * gradient
        return gradient_col


def determine_padding(filter_shape, output_shape="same"):
    """Method which calculates the padding based on the specified output shape
       and the shape of the filters."""
    if output_shape == "valid":
        return (0, 0), (0, 0)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape
        # output_height = (height + pad_h - filter_height) / stride + 1
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))
        return (pad_h1, pad_h2), (pad_w1, pad_w2)


def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height *
                  filter_width).reshape(-1, 1)
    return (k, i, j)


def image_to_column(images, filter_shape, stride, output_shape="same"):
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    # Add padding to the image
    images_padded = np.pad(
        images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(
        images.shape, filter_shape, (pad_h, pad_w), stride)
    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(
        filter_height * filter_width * channels, -1)
    return cols


def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.zeros(
        (batch_size, channels, height_padded, width_padded))
    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(
        images_shape, filter_shape, (pad_h, pad_w), stride)
    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)
    # Return image without padding
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]
