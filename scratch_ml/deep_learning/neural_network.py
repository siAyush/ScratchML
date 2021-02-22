import numpy as np
import progressbar
from scratch_ml.utils import bar_widget


class NeuralNetwork():
    """Deep Learning base model."""

    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.loss_function = loss
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.progressbar = progressbar.ProgressBar(bar_widget)
        self.validation_set = None
        if validation_data:
            x, y = validation_data
            self.validation_set = x, y

    def set_trainable(self, trainable):
        """Method which enables freezing of the weights of the network layers."""
        if layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """Method which adds a layer to the neural network."""
        
