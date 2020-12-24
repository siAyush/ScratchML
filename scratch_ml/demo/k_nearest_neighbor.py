import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from scratch_ml.supervised_learning import KNN
from scratch_ml.utils import accuracy_score, train_test_split, euclidean_distance, normalize



def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    clf = KNN(k=5)
    y_pred = clf.predict(x_test, x_train, y_train)
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()


