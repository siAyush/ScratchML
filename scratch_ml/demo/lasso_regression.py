import numpy as np
from sklearn.datasets import make_regression
from scratch_ml.utils import train_test_split, mean_squared_error
from scratch_ml.supervised_learning import LassoRegression
import matplotlib.pyplot as plt


def main():
    print("Lasso Regression")
    x, y = make_regression(n_samples=200, n_features=1, noise=25)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    model = LassoRegression(reg_factor=0.05,   learning_rate=0.001, n_iterations=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
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