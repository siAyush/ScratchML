import numpy as np
import math
from itertools import combinations_with_replacement


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


def calculate_variance(x):
    """Calculate the variance"""
    mean = np.ones(np.shape(x)) * x.mean(0)
    n_samples = np.shape(x)[0]
    variance = (1 / n_samples) * np.diag((x - mean).T.dot(x - mean))
    return variance


def calculate_std_dev(x):
    """ Calculate the standard deviations"""
    std_dev = np.sqrt(calculate_variance(x))
    return std_dev


def covariance_matrix(x,y=None):
    """Calculate the covariance matrix"""
    if y is None:
        y = x
    n_samples = np.shape(x)[0]
    matrix = ((1/(n_samples-1))*(x-x.mean(axis=0)).T.dot(y-y.mean(axis=0)))
    return np.array(matrix, dtype=float)


def correlation_matrix(x, y=None):
    """Calculate the correlation matrix"""
    if y is None:
        y = x
    n_samples = np.shape(x)[0]
    covariance = ((1/(n_samples-1))*(x-x.mean(axis=0)).T.dot(y-y.mean(axix=0)))
    std_dev_x = np.expand_dims(calculate_std_dev(x), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(y), 1)
    matrix = np.divide(covariance, std_dev_x.T.dot(std_dev_y))
    return np.array(matrix, dtype=float)


def calculate_entropy(y):
    """Calculate the entropy"""
    log2 = lambda x : math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def divide_on_feature(x, feature_i, threshold):
    """Divide dataset based on if sample value on feature index is larger than
        the given threshold."""
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample : sample[feature_i] >= threshold
    else:
        split_func = lambda sample : sample[feature_i] == threshold     
        
    x_1 = np.array([sample for sample in x if split_func(sample)])
    x_2 = np.array([sample for sample in x if not split_func(sample)])
    return np.array([x_1, x_2], dtype=object)


def to_categorical(x, n_col=None):
    """Onehot encoding of nominal values"""
    if not n_col:
        n_col = np.amax(x)+1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def get_random_subsets(x, y, n_subsets, replacements=True):
     """Return random subsets with replacements of the data"""
