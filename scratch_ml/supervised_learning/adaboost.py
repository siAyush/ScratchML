import numpy as np
import math


class DecisionStump():
    """A decision stump is a machine learning model consisting of a one-level decision tree.
       Decision stump used as weak classifier."""
    def __init__(self):
        self.polarity = 1   # Determines if sample shall be classified as -1 or 1 given threshold
        self.feature_index = None
        self.threshold = None
        self.alpha = None   # Value indicative of the classifier's accuracy


class Adaboost():
    """Boosting method that uses a number of weak classifiers in 
    ensemble to make a strong classifier."""
    def __init__(self, n_clf=5):
        self.n_clf = n_clf      # The number of weak classifiers
    
    
    def fit(self, x, y):
        n_samples, n_features = np.shape(x)
        w = np.full(n_samples, (1/n_samples))     # Initialize weights to 1/N

        self.clfs = []
        for i in range(self.n_clf):
            

