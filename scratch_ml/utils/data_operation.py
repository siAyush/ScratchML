import numpy as np
import math

def euclidean_distance(x1, x2):
    """Calculates the euclidean distance between two vectors"""
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def accuracy_score(y_true, y_pred):
    """Return the accuracy"""
    accuracy = np.sum(y_true == y_pred, axis=0)/len(y_true)
    return accuracy


def normalize(v):
    """Normalize the dataset"""
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def shuffel_data(x, y, seed=None):
    """Random shuffle of the data"""
    if seed:
        np.random.seed(seed)
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index], y[index]
    


def train_test_split(x, y, test_size=0.25, shuffel=True, seed=None):
    """Split the data into train and test sets"""
    if shuffel:
        x, y = shuffel_data(x,y, seed)
    split_i = int(len(y)*test_size)
    x_train, x_test = x[:split_i], x[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
       
    return x_train, x_test, y_train, y_test


def mean_squared_error(y_true, y_pred):
    """Calculate the mean squared error"""
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_variance():
    """Calculate the variance"""

    
def calculate_std_dev(x):
    """ Calculate the standard deviations"""
    std_dev = np.sqrt(calculate_variance(x))
    return std_dev


def covariance_matrix(x,y=None):
    """Calculate the covariance matrix"""
    if y is None:
        y = x
    n_samples = np.sahpe(x)[0]
    matrix = ((1/(n_samples-1))*(x-x.mean(axis=0)).T.dot(y-y.mean(axix=0)))
    return np.array(matrix, dtype=float)


def correlation_matrix(x, y=None):
    """Calculate the correlation matrix"""
    if y is None:
        y = x
    n_samples = np.shape(x)[0]
    covariance = ((1/(n_samples-1))*(x-x.mean(axis=0)).T.dot(y-y.mean(axix=0)))
    std_dev_x = np.expand_dims(std_dev(x), 1)
    std_dev_y = np.expand_dims(std_dev(y), 1)
    matrix = np.divide(covariance, std_dev_x.T.dot(std_dev_y))
    return np.array(matrix, dtype=float)