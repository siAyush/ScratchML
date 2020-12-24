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