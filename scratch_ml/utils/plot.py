import numpy as np
import progressbar
import matplotlib.pyplot as plt 
from scratch_ml.utils import covariance_matrix


bar_widget = ['Training: ', progressbar.Percentage(), ' ', 
              progressbar.Bar(marker="-", left="[", right="]"), ' ', progressbar.ETA()]


class Plot():
    def __init__(self):
        self.cmap = plt.get_cmap("plasma")
    

    def _transfrom(self, x, dim):
        covariance = covariance_matrix(x)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:dim]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
        x_new = x.dot(eigenvectors)
        return x_new


    def plot_2d(self, x, y=None, title=None, accuracy=None, legend_label=None):
        """Plot the dataset x in 2D using PCA"""
        x_transformed = self._transfrom(x, dim=2)
        pc1 = x_transformed[:, 0]
        pc2 = x_transformed[:, 1]
        class_dist = []
        y = np.array(y).astype(int)
        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # plotting
        for i, j in enumerate(np.unique(y)):
            _x1 = pc1[y == j]
            _x2 = pc2[y == j]
            class_dist.append(plt.scatter(_x1, _x2, color=colors[i]))

        # plot leagend
        if legend_label is not None:
            plt.legend(class_dist, legend_label, loc=1)
        
        # plot title
        if title:
            if accuracy:
                per = 100*accuracy
                plt.suptitle(title)
                plt.title("Accuracy : {:.2f}".format(per), fontsize=10)
            else:
                plt.title(title)
        
        # Axis
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        plt.show()
    

    def plot_3d(self, x ,y=None):
        """Plot the dataset x in 3D using PCA"""
        x_transformed = self._transfrom(x, dim=3)
        x1 = x_transformed[:, 0]
        x2 = x_transformed[:, 1]
        x3 = x_transformed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        plt.show()