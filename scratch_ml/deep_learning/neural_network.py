import numpy as np
import progressbar
from terminaltables import AsciiTable
from scratch_ml.utils import bar_widget, batch_iterator


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
            self.validation_set = {"x": x, "y": y}

    def set_trainable(self, trainable):
        """Method which enables freezing of the weights of the network layers."""
        if layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """Method which adds a layer to the neural network."""
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
         # If the layer has weights that needs to be initialized
        if hasattr(layer, "initialize"):
            layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def _forward_pass(self, x, training=True):
        """Calculate the output."""
        layer_output = x
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

    def _backward_pass(self, loss_grad):
        """Propagate the gradient 'backwards' and update the weights in each layer."""
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def train_on_batch(self, x, y):
        """Single gradient update over one batch of samples."""
        y_pred = self._forward_pass(x)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.accuracy(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward_pass(loss_grad=loss_grad)
        return loss, acc

    def test_on_batch(self, x, y):
        """Evaluates the model over a single batch of samples."""
        y_pred = self._forward_pass(x, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.accuracy(y, y_pred)
        return loss, acc

    def fit(self, x, y, n_epochs, batch_size):
        """Trains the model."""
        for i in self.progressbar(range(n_epochs)):
            batch_error = []
            for x_batch, y_batch in batch_iterator(x, y, batch_size=batch_size):
                loss, acc = self.train_on_batch(x_batch, y_batch)
                batch_error.append(loss)
            self.errors["training"].append(np.mean(batch_error))

            if self.validation_set != None:
                val_loss, val_acc = self.test_on_batch(
                    self.validation_set["x"], self.validation_set["y"])
                self.errors["validation"].append(val_loss)

        return self.errors["training"], self.errors["validation"]

    def predict(self, x):
        return self._forward_pass(x, training=False)

    def summary(self, name="Model Summary"):
        """Print model name."""
        print(AsciiTable([[name]]).table)
        print("Input Shape %s" % str(self.layers[0].input_shape))
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        total_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            total_params += params
        print(AsciiTable(table_data).table)
        print("Total Parameters: %d\n" % total_params)
