import numpy as np
from sklearn import datasets
from scratch_ml.utils import train_test_split, normalize, accuracy_score, Plot
from scratch_ml.supervised_learning import NaiveBayes


def main():
    print("Naive Bayes")
    data = datasets.load_digits()
    x = normalize(data.data)
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    clf = NaiveBayes()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)
    Plot().plot_2d(x_test, y_pred, title="Naive Bayes", accuracy=accuracy, legend_label=data.target_names)


if __name__ == "__main__":
    main()