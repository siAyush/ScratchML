import numpy as np
from sklearn import datasets

from scratch_ml.utils import Plot , train_test_split, accuracy_score
from scratch_ml.supervised_learning import ClassificationTree


def main():
    print ("Classification Tree")
    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    Plot().plot_2d(X_test, y_pred, title="Decision Tree", accuracy=accuracy, legend_label=data.target_names)


if __name__ == "__main__":
    main()