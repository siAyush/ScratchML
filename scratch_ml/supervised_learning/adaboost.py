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
            clf = DecisionStump()
            min_error = float('inf')    # Minimum error
            for feature_i in range(n_features):
                feature_values = np.expand_dims(x[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    p = 1                # Set all predictions to '1' initially
                    prediction = np.ones(np.shape(y))
                    prediction[x[:,feature_i] < threshold] = -1     # Label the samples whose values are below threshold as '-1'
                    error = sum(w[y != prediction])     # Sum of weights of misclassified samples
                    # If the error is over 50% we flip the polarity so that samples that
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    # If this threshold resulted in the smallest error we save the configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
            
            # Calculate the alpha which is used to update the sample weights
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = np.ones(np.shape(y))   # Set all predictions to '1' initially
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * x[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1       # Label those as '-1'
            # Missclassified samples gets larger weights and correctly classified samples smaller
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)
    

    def  predict(self, x):
        n_samples = np.shape(x)[0]
        y_pred = np.zeros((n_samples, 1))
        for clf in self.clfs:
            predictions = np.ones(np.shape(y_pred))
            negative_idx = (clf.polarity * x[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions
        y_pred = np.sign(y_pred).flatten()
        return y_pred