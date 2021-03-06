import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scratch_ml.deep_learning import NeuralNetwork
from scratch_ml.deep_learning import optimizers
from scratch_ml.utils import Plot, train_test_split, to_categorical, CrossEntropy
from scratch_ml.deep_learning.layers import Dense, Activation, Dropout
from scratch_ml.deep_learning.optimizers import Adam


def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target
    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))
    n_samples, n_features = X.shape
    n_hidden = 512
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)

    optimizer = Adam()
    model = NeuralNetwork(optimizer=optimizer,
                          loss=CrossEntropy, validation_data=(X_test, y_test))

    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(Activation("leaky_relu"))
    model.add(Dense(n_hidden))
    model.add(Activation("leaky_relu"))
    model.add(Dropout(0.25))
    model.add(Dense(n_hidden))
    model.add(Activation("leaky_relu"))
    model.add(Dropout(0.25))
    model.add(Dense(n_hidden))
    model.add(Activation("leaky_relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.summary(name="Neural Network")

    train_err, val_err = model.fit(
        X_train, y_train, n_epochs=50, batch_size=256)

    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

    _, accuracy = model.test_on_batch(X_test, y_test)
    print("Accuracy:", accuracy)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    Plot().plot_2d(X_test, y_pred, title="Neural Network",
                   accuracy=accuracy, legend_label=range(10))


if __name__ == "__main__":
    main()
