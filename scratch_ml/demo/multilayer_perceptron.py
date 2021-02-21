import numpy as np
from sklearn import datasets
from scratch_ml.deep_learning import MultilayerPerceptron
from scratch_ml.utils import to_categorical, normalize, train_test_split, Plot, accuracy_score


def main():
    data = datasets.load_digits()
    x = normalize(data.data)
    y = data.target
    y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=1)

    clf = MultilayerPerceptron(
        n_hidden=16, n_iterations=1000, learning_rate=0.01)
    clf.fit(x_train, y_train)
    y_pred = np.argmax(clf.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    Plot().plot_2d(x_test, y_pred, title="Multilayer Perceptron",
                   accuracy=accuracy, legend_label=np.unique(y))


if __name__ == "__main__":
    main()
