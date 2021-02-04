import numpy as np 
import progressbar
import math
from scratch_ml.utils import bar_widget, get_random_subsets 
from scratch_ml.supervised_learning import ClassificationTree


class  RandomForest():
    """Random Forest classifier"""
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators    # Number of trees
        self.max_features = max_features    # Maxmimum number of features per tree   
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain            # Minimum information gain
        self.max_depth = max_depth          # Maximum depth for tree
        self.progressbar = progressbar.ProgressBar(widgets=bar_widget)

        self.tree = []
        for i in range(n_estimators):
            tree = ClassificationTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_gain,
                max_depth=self.max_depth)
            self.tree.append(tree)
          

    def fit(self, x, y):
        n_features = np.shape(x)[1]   
        # If max_features have not been defined select sqrt(n_features) 
        if self.max_features is None:
            self.max_features = int(math.sqrt(n_features))
        # Choose one random subset of the data for each tree
        subsets =  get_random_subsets(x, y, self.n_estimators)
        for i in self.progressbar(range(self.n_estimators)):
            x_subset, y_subset = subsets[i]
            # select random subsets of the features
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            self.tree[i].feature_i = idx
            x_subset = x_subset[:, idx]
            self.tree[i].fit(x_subset, y_subset)
        

    def predict(self, x):
        y_preds = np.empty((x.shape[0], len(self.tree)))
        for i, tree in enumerate(self.tree):
            idx = tree.feature_i
            # Make a prediction based on those features
            prediction = tree.predict(x[:, idx])
            y_preds[:, i] = prediction
        y_pred = []
        # Select the most common class prediction
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype("int")).argmax())
        return y_pred