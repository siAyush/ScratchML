import math
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scratch_ml.deep_learning.optimizers import Adam
from scratch_ml.deep_learning import NeuralNetwork
from scratch_ml.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization
from scratch_ml.utils import to_categorical, train_test_split, Plot, CrossEntropy


def main():

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, seed=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    optimizer = Adam()
    Model = NeuralNetwork(optimizer=optimizer,
                          loss=CrossEntropy,
                          validation_data=(X_test, y_test))

    Model.add(Conv2D(n_filters=16, filter_shape=(3, 3),
                     stride=1, input_shape=(1, 8, 8), padding='same'))
    Model.add(Activation("relu"))
    Model.add(Dropout(0.25))
    Model.add(BatchNormalization())
    Model.add(Conv2D(n_filters=32, filter_shape=(
        3, 3), stride=1, padding="same"))
    Model.add(Activation("relu"))
    Model.add(Dropout(0.25))
    Model.add(BatchNormalization())
    Model.add(Flatten())
    Model.add(Dense(256))
    Model.add(Activation("relu"))
    Model.add(Dropout(0.4))
    Model.add(BatchNormalization())
    Model.add(Dense(10))
    Model.add(Activation("softmax"))
    Model.summary(name="CNN")

    train_err, val_err = Model.fit(
        X_train, y_train, n_epochs=50, batch_size=256)
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

    _, accuracy = Model.test_on_batch(X_test, y_test)
    print("Accuracy:", accuracy)
    y_pred = np.argmax(Model.predict(X_test), axis=1)
    X_test = X_test.reshape(-1, 8*8)

    Plot().plot_2d(X_test, y_pred, title="Convolutional Neural Network",
                   accuracy=accuracy, legend_label=range(10))


if __name__ == "__main__":
    main()
