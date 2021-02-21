import numpy as np
from sklearn import datasets
from scratch_ml.supervised_learning import Adaboost
from scratch_ml.utils import train_test_split, accuracy_score
from scratch_ml.utils import Plot


def main():
    print("Adaboost")
    data = datasets.load_digits()
    x = data.data
    y = data.target
    digit1 = 1
    digit2 = 5
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # Change labels to {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    x = data.data[idx]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    clf = Adaboost(n_clf=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    Plot().plot_2d(x_test, y_pred, title="Adaboost", accuracy=accuracy)


if __name__ == "__main__":
    main()
