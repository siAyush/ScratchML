import numpy as np
from sklearn import datasets
from scratch_ml.utils import Plot
from scratch_ml.unsupervised_learning import KMeans


def main():
    print("KMeans")
    x, y = datasets.make_blobs()

    clf = KMeans(k=3)
    y_pred = clf.predict(x)

    p = Plot()
    p.plot_2d(x, y_pred, title="K-Means Clustering")


if __name__ == "__main__":
    main()
