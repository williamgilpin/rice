import numpy as np
import warnings

from dynGENIE3 import *
# from dyngenie3_alternate import *

def run_dynGENIE3(X, ntrees=1000, nthreads=1, max_cells=None):
    """
    Run dynGENIE3 on a dataset. If the input is a single dataset, it will be
    promoted to a batch of datasets.

    Args:
        X (np.ndarray): A data matrix with shape (n_genes, n_samples) or a list
            of such matrices
        ntrees (int): Number of trees in the random forest for each gene
        nthreads (int): Number of parallel workers (one gene per worker)
        max_cells (int): If set, subsample the rows of each dataset to at most
            this many timepoints, evenly spaced along the (pseudotime-ordered)
            first axis. Preserves temporal ordering.

    Returns:
        np.ndarray: A matrix of scores for each gene-gene pair
    """
    ## Promote single dataset to batch of datasets
    if len(X[0].shape) == 1:
        Xall = [X.copy()]
    else:
        Xall = [item.copy() for item in X]

    if max_cells is not None:
        Xall = [
            item[np.linspace(0, item.shape[0] - 1, max_cells).astype(int)]
            if item.shape[0] > max_cells else item
            for item in Xall
        ]

    time_points = [np.arange(item.shape[0]) for item in Xall]
    out = dynGENIE3(Xall, time_points, ntrees=ntrees, nthreads=nthreads)
    (VIM, alphas, prediction_score, stability_score, treeEstimators) = out
    cmat = VIM.copy()
    return cmat

