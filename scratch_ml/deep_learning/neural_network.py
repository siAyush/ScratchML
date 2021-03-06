import numpy as np
import progressbar
from terminaltables import AsciiTable
from scratch_ml.utils import bar_widget, batch_iterator


class NeuralNetwork():
    """Neural Networ base model."""

    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widget)
        self.val_set = None
        if validation_data:
            x, y = validation_data
            self.val_set = {"x": x, "y": y}

    def set_trainable(self, trainable):
        """Method which enables freezing of the weights of the network's layers."""
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """Method which adds a layer to the neural network."""
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        # weights that needs to be initialized
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def test_on_batch(self, x, y):
        """Evaluates the model over a single batch of samples."""
        y_pred = self._forward_pass(x, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.accuracy(y, y_pred)
        return loss, acc

    def train_on_batch(self, x, y):
        """Single gradient update over one batch of samples."""
        y_pred = self._forward_pass(x)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.accuracy(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward_pass(loss_grad=loss_grad)
        return loss, acc

    def fit(self, x, y, n_epochs, batch_size):
        """Trains the model for a fixed number of epochs."""
        for _ in self.progressbar(range(n_epochs)):
            batch_error = []
            for X_batch, y_batch in batch_iterator(x, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
            self.errors["training"].append(np.mean(batch_error))
            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(
                    self.val_set["x"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, x, training=True):
        layer_output = x
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        return layer_output

    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def summary(self, name="Model Summary"):
        print(AsciiTable([[name]]).table)
        print("Input Shape: %s" % str(self.layers[0].input_shape))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        print(AsciiTable(table_data).table)
        print("Total Parameters: %d\n" % tot_params)

    def predict(self, x):
        return self._forward_pass(x, training=False)
