import numpy as np
import matplotlib.pyplot as plt
from scratch_ml.utils import to_categorical, train_test_split, CrossEntropy, accuracy_score
from scratch_ml.deep_learning import NeuralNetwork
from scratch_ml.deep_learning.layers import RNN, Activation
from scratch_ml.deep_learning.optimizers import Adam


def main():

    def gen_mult_ser(nums):
        """Method which generates multiplication series."""
        X = np.zeros([nums, 10, 61], dtype=float)
        y = np.zeros([nums, 10, 61], dtype=float)
        for i in range(nums):
            start = np.random.randint(2, 7)
            mult_ser = np.linspace(start, start*10, num=10, dtype=int)
            X[i] = to_categorical(mult_ser, n_col=61)
            y[i] = np.roll(X[i], -1, axis=0)
        y[:, -1, 1] = 1
        return X, y

    X, y = gen_mult_ser(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    optimizer = Adam()
    Model = NeuralNetwork(optimizer=optimizer, loss=CrossEntropy)
    Model.add(RNN(10, activation="tanh", bptt=5, input_shape=(10, 61)))
    Model.add(Activation('softmax'))
    Model.summary("RNN")

    tmp_X = np.argmax(X_train[0], axis=1)
    tmp_y = np.argmax(y_train[0], axis=1)

    print("Number Series Problem:")
    print("X = [" + " ".join(tmp_X.astype("str")) + "]")
    print("y = [" + " ".join(tmp_y.astype("str")) + "]")
    print()

    train_err, _ = Model.fit(X_train, y_train, n_epochs=500, batch_size=512)
    y_pred = np.argmax(Model.predict(X_test), axis=2)
    y_test = np.argmax(y_test, axis=2)

    print("Results:")
    for i in range(5):
        tmp_X = np.argmax(X_test[i], axis=1)
        tmp_y1 = y_test[i]
        tmp_y2 = y_pred[i]
        print("X      = [" + " ".join(tmp_X.astype("str")) + "]")
        print("y_true = [" + " ".join(tmp_y1.astype("str")) + "]")
        print("y_pred = [" + " ".join(tmp_y2.astype("str")) + "]")
        print()

    accuracy = np.mean(accuracy_score(y_test, y_pred))
    print("Accuracy:", accuracy)

    plt.plot(range(500), train_err, label="Training Error")
    plt.title("Error Plot")
    plt.ylabel("Training Error")
    plt.xlabel("Iterations")
    plt.show()


if __name__ == "__main__":
    main()
