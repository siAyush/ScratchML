import numpy as np
import progressbar
from scratch_ml.supervised_learning import RegressionTree
from scratch_ml.utils import SquareLoss, CrossEntropy, to_categorical, bar_widget


class  GradientBoosting():
    """
    Super class of GradientBoostingClassifier and GradientBoostinRegressor. 

    n_estimators: The number of classification trees that are used.
    learning_rate: The step length that will be taken when following the negative gradient during training.
    min_samples_split: The minimum number of samples needed to make a split when building a tree.
    min_impurity: The minimum impurity required to split the tree further. 
    max_depth: The maximum depth of a tree.
    regression: True or false depending on if we're doing regression or classification.
    """
    def __init__(self, n_estimators, learning_rate, min_samples_split, 
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.bar = progressbar.ProgressBar(widgets=bar_widget)

        # SquareLoss for regression, CrossEntropy for classification
        self.loss = SquareLoss()
        if self.regression is False:
            self.loss = CrossEntropy()
        
        self.tree = []
        for i in range(n_estimators):
            tree = RegressionTree(min_samples_split=self.min_samples_split,
                                  max_depth=self.max_depth,
                                  min_impurity=self.min_impurity)
            self.tree.append(tree)
            

    def fit(self, x, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in self.bar(range(self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)
            self.tree[i].fit(x, gradient)
            update = self.tree[i].predict(x)
            y_pred = y_pred - np.multiply(self.learning_rate, update)


    def predict(self, x):
        y_pred = np.array([])
        for tree in self.tree:
            update = tree.predict(x)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update
        
        if not self.regression:
            # Turn into probability distribution
             y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
             y_pred = np.argmax(y_pred, axis=1)
             return y_pred


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True)


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False)
            
    def fit(self, x, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(x, y)