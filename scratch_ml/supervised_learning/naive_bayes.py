import numpy as np
import math


class NaiveBayes():
    """The Gaussian Naive Bayes classifier"""


    def _calculate_likelihood(self, mean, var, x):
        """Gaussian likelihood of the data x given mean and var"""
        eps = 1e-4 
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent
    

    def _calculate_prior(self, c):
        """Calculate the prior of the probability the class c."""
        frequency = np.mean(self.y == c)
        return frequency
    

    def _classify(self, sample):
        """Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X)"""
        posteriors = []
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # return the class with the largest probability
        return self.classes[np.argmax(posteriors)]
    

    def fit(self, x, y):
        self.x, self.y = x, y
        self.classes = np.unique(y)
        self.parameters = []
        # calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            x_where_c = x[np.where(y == c)]
            self.parameters.append([])
            for col in x_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)
    
    def predict(self, x):
        y_pred = [self._classify(sample) for sample in x]
        return y_pred