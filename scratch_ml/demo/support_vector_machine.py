import numpy as np
from sklearn import datasets
from scratch_ml.supervised_learning import SupportVectorMachine
from scratch_ml.utils import normalize, train_test_split, accuracy_score, Plot


def main():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = SupportVectorMachine(power=4, coef=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    Plot().plot_2d(X_test, y_pred, title="Support Vector Machine", accuracy=accuracy)


if __name__ == "__main__":
    main()
