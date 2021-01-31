import numpy as np
from sklearn.datasets import make_regression
from scratch_ml.utils import train_test_split, mean_squared_error
from scratch_ml.supervised_learning import GradientBoostingRegressor
import matplotlib.pyplot as plt


def main():
    print ("Gradient Boosting Regression")
    x, y = make_regression(n_samples=200, n_features=1, noise=20)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cmap = plt.get_cmap('plasma')
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean Squared Error:", mse)

    m1 = plt.scatter(366 * x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * x_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * x_test, y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()