import numpy as np
from sklearn import datasets

from scratch_ml.supervised_learning import KNN
from scratch_ml.utils import accuracy_score, train_test_split, euclidean_distance, normalize
from scratch_ml.utils import Plot



def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    knn = KNN(k=5)
    y_pred = knn.predict(x_test, x_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    Plot().plot_2d(x_test, y_pred, title="K Nearest Neighbors", accuracy=accuracy, legend_label=data.target_names)


if __name__ == "__main__":
    main()