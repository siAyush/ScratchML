import numpy as np
from sklearn import datasets
from scratch_ml.utils import Plot , train_test_split, accuracy_score
from scratch_ml.supervised_learning import ClassificationTree


def main():
    print ("Classification Tree")
    data = datasets.load_iris()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    clf = ClassificationTree()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    Plot().plot_2d(x_test, y_pred, title="Decision Tree", accuracy=accuracy, legend_label=data.target_names)


if __name__ == "__main__":
    main()