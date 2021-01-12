import numpy as np
from scratch_ml.utils import calculate_entropy, divide_on_feature


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
        self.root = None                            # Root node
        self.min_samples_split = min_samples_split  # Minimum number of samples to split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss                        # If Gradient Boost
        self._impurity_calculation = None       # Function to calculate impurity (classif=>information gain, regr=>variance reduct)
        self._leaf_value_calculation = None     # Function to determine prediction of y at leaf
        self.one_dim = None                     # If y is one-hot encoded  
    

    def _build_tree(self, x, y, current_depth=0):
        """Recursive method which builds out the decision tree and splits x and respective y."""
        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        #add y as last column of X
        xy = np.concatenate(x, y, axis=1)
        n_samples, n_features = np.shape(x)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # calculate the impurity for each feature
            for feature_i in range(n_features):
                # all values of feature_i
                feature_values = np.expand_dims(x[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # calculate the impurity for all unique values of feature
                for threshold in unique_values:
                    # divide x and y
                    xy1, xy2 = divide_on_feature(xy, feature_i, threshold)
                    if len(xy1)>0 and len(xy2)>0:
                        # Select the y-values
                        y1 = xy1[:, n_features]
                        y2 = xy2[:, n_features]
                        # calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i":feature_i, "threshold":threshold}
                            best_sets = {
                                "leftx": xy1[:, :n_features],   # x of left subtree
                                "lefty": xy1[:, n_features:],   # y of left subtree
                                "rightx": xy2[:, :n_features],  # x of right subtree
                                "righty": xy2[:, n_features:]   # y of right subtree
                                }
        
        if largest_impurity > self.min_impurity:
            # build subtrees
            true_branch = self._build_tree(best_sets["leftx"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightx"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)
        
        # for leaf node
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)
    

    def fit(self, x, y, loss=None):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(x, y)
        self.loss = loss


    def predict_value(self, x, y, tree=None):
        """Do a recursive search down the tree and make a prediction."""
        if tree is None:
            return self.root
        if tree.value is not None:
            return tree.value
        
        feature_value = x[tree.feature]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        
        return self.predict_value(x, branch)
    

    def predict(self, x):
        y_pred = [self.predict_value(sample) for sample in x]
        return y_pred
    

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if tree is None:
            tree = self.root
        if tree.value is not None:
            print (tree.value)
        else:
            print("%s:%s " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)