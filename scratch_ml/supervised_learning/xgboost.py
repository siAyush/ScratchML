import numpy as np 
import progressbar
from scratch_ml.supervised_learning import DecisionTree
from scratch_ml.utils import Sigmoid, bar_widget, to_categorical


class LogisticLoss():
    def __init__(self):
        self.log_func = Sigmoid()
    

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)
    

    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    
    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)


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
        gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation =  gradient / hessian
        return update_approximation
    

    def fit(self, x, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(x, y)

    
class XGBoost():
    """The XGBoost classifier"""
    def __init__(self, n_estimators=200, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators            # Number of trees
        self.learning_rate = learning_rate          # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity            # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree
        self.bar = progressbar.ProgressBar(widgets=bar_widget)
        self.loss = LogisticLoss()                  # Log loss for classification

        self.tree = []
        for i in range(n_estimators):
            tree = XGBoostRegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth,
                    loss=self.loss)
            self.tree.append(tree)


    def fit(self, x, y):
        y = to_categorical(y)
        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            tree = self.tree[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(x, y_and_pred)
            update_pred = tree.predict(x)
            y_pred = y_pred - np.multiply(self.learning_rate, update_pred)


    def predict(self, x):
        y_pred = None
        for tree in self.tree:
            update_pred = tree.predict(x)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred = np.multiply(self.learning_rate, update_pred)
        # Turn into probability distribution
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred