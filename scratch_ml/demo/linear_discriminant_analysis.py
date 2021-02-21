import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from scratch_ml.supervised_learning import LDA
from scratch_ml.utils import normalize, train_test_split, Plot, accuracy_score


def main():
    print("Linear Discriminant Analysis")
    data = datasets.load_iris()
    x = data.data
    y = data.target
    # Three -> two classes
    x = x[y != 2]
    y = y[y != 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    lda = LDA()
    lda.fit(x_train, y_train)
    y_pred = lda.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    Plot().plot_2d(x_test, y_pred, title="LDA", accuracy=accuracy)


if __name__ == "__main__":
    main()
