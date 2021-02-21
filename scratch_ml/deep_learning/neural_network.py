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

    def fit():
        pass
