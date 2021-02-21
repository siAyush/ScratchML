import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scratch_ml.unsupervised_learning import PCA


def main():
    data = datasets.load_digits()
    x = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    x_transformed = PCA().transform(x, 2)
    x1 = x_transformed[:, 0]
    x2 = x_transformed[:, 1]

    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
    class_distr = []
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    plt.legend(class_distr, y, loc=1)
    plt.suptitle("PCA Dimensionality Reduction")
    plt.title("Digit Dataset")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
