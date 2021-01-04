import numpy as np
from sklearn.datasets import make_regression
from scratch_ml.utils import train_test_split, mean_squared_error
from scratch_ml.supervised_learning import LassoRegression
import matplotlib.pyplot as plt
import pandas as pd


def main():
    #x, y = make_regression(n_samples=100, n_features=1, noise=25)
    #x_train, x_test, y_train, y_test = train_test_split(x, y)

    data = pd.read_csv('scratch_ml/data/data.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = data["temp"].values

    x = time # fraction of the year [0, 1]
    y = temp
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    model = LassoRegression(degree=15, reg_factor=0.05, learing_rate=0.001, n_iterations=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #mse = mean_squared_error(y_test, y_pred)
    mse = 1
    print ("Mean squared error: %s" %(mse))

    # plot the results
    cmap = plt.get_cmap('plasma')
    y_pred_line = model.predict(x)
    train = plt.scatter(400*x_train, y_train, color=cmap(0.9), s=10)
    test = plt.scatter(400*x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(400*x, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Lasso Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend((train, test), ("Training data", "Test data"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()