import numpy as np
from sklearn.datasets import make_regression
from scratch_ml.utils import train_test_split, mean_squared_error
from scratch_ml.supervised_learning import LinearRegression
import matplotlib.pyplot as plt


def main():
    x, y = make_regression(n_samples=100, n_features=1, noise=30)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    model = LinearRegression(n_iterations=100)
    model.fit(x, y)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s" %(mse))


if __name__ == "__main__":
    main()