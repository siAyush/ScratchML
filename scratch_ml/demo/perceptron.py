import numpy as np 
from sklearn import datasets
from scratch_ml.deep_learning import Perceptron
from scratch_ml.utils import Plot, CrossEntropy, Sigmoid
from scratch_ml.utils import train_test_split, normalize, to_categorical, accuracy_score


def main():
    print("Perceptron")
    data = datasets.load_digits()
    x = normalize(data.data)
    y = data.target
    #one-hot encoding
    y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=1)

    clf = Perceptron(n_iterations=5000,
                    learning_rate=0.001, 
                    loss=CrossEntropy,
                    activation_function=Sigmoid)
    clf.fit(x_train, y_train)

    y_pred = np.argmax(clf.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    Plot().plot_2d(x_test, y_pred, title="Perceptron", accuracy=accuracy, legend_label=np.unique(y))


if __name__ == "__main__":
    main()