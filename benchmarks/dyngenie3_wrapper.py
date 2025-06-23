import numpy as np
import warnings

from dynGENIE3 import *
# from dyngenie3_alternate import *

def run_dynGENIE3(X):
    """
    Run dynGENIE3 on a dataset. If the input is a single dataset, it will be 
    promoted to a batch of datasets.

    Args:
        X (np.ndarray): A data matrix with shape (n_genes, n_samples) or a list 
            of such matrices

    Returns:
        np.ndarray: A matrix of scores for each gene-gene pair
    """
    ## Promote single dataset to batch of datasets
    if len(X[0].shape) == 1:
        Xall = [X.copy()]
    else:
        Xall = [item.copy() for item in X]

    time_points = [np.arange(item.shape[0]) for item in Xall]
    out = dynGENIE3(Xall, time_points)
    (VIM, alphas, prediction_score, stability_score, treeEstimators) = out
    cmat = VIM.copy()
    return cmat

