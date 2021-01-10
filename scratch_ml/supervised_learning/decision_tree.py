import numpy as np
from scratch_ml.utils import train_test_split, mean_squared_error
from scratch_ml.utils import accuracy_score, calculate_variance, calculate_entropy


class DecisionNode():
    """
    Decision node class for decision tree.
    --------------------------------------

    Parameters:
    feature_i: int 
        Index for the feature that is tested.
    threshold: float
        Threshold value for feature.
    value: float
        Value if the node is a leaf in the tree.
    true_branch: DecisionNode
        Left subtree 
    false_branch: DecisionNode
        Right subtree
    """
    def __init__(self, feature_i=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = None
        self.true_branch = None
        self.false_branch = None


# Class of RegressionTree and ClassificationTree
class DecisionTree():
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), loss=None):
        self.root = None    # Root node
        self.min_samples_split = min_samples_split  # Minimum number of samples to split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss    # If Gradient Boost
        self._impurity_calculation = None   # Function to calculate impurity (classif=>information gain, regr=>variance reduct)
        self._leaf_value_calculation = None     # Function to determine prediction of y at leaf
        self.one_dim = None     # If y is one-hot encoded  
    

    def _build_tree(self, x, y):
        pass


    def fit(self, x, y, loss=None):
        self.one_dim = len(np.shape(y)) == 1
        self.loss = loss
        self.root = _build_tree(x,y)