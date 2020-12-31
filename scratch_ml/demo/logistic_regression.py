import numpy as np
from sklearn import datasets

from scratch_ml.utils import Plot
from scratch_ml.supervised_learning import LogisticRegression
from scratch_ml.utils import normalize, train_test_split, accuracy_score

def main():
    data = datasets.load_iris()
    # using two class only
    x = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=1)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # plot the results
    Plot().plot_2d(x_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()