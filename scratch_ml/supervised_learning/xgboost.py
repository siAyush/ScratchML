import numpy as np 
from scratch_ml.supervised_learning import DecisionTree

class XGBoostRegressionTree(DecisionTree):
    def _gain(self, y, y_pred):
        nominator = np.power(y * self.loss.gradient(y, y_pred).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator/denominator)
    
    def _split(self, y):
        """y contains y_true in left half of the middle column and
        y_pred in the right half"""
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred
    
    def _gain_by_taylor(self, y, y1, y2):
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain
    
    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        pass



class XGBoost():
    def __init__(self):
        pass


    def fit(self, x, y):
        pass


    def predict(self, x):
        pass