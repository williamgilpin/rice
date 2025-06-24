import os
import sys
import warnings
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

## Add swing directory to path
sys.path.append(os.path.join(current_dir, "SWING"))

from Swing import Swing

def run_swing(X, tempdir=""):
    N, D = X.shape
    tpts = np.arange(N)
    X2 = np.vstack((tpts, X.T)).T
    gene_names = [f"G{i}" for i in range(D)]
    gene_names = ["\"Time\""] + gene_names
    X2 = np.vstack((gene_names, X2))
    np.savetxt("temp.txt", X2, fmt="%s", delimiter="\t")

    gene_start_column = 1
    time_label = "Time"
    separator = "\t"
    gene_end = None
    file_path = "temp.txt"

    k_min = 1
    k_max = 3
    w = 10
    method = 'RandomForest'

    trees = 100
    sg = Swing(
        file_path, gene_start_column, gene_end, time_label, separator, 
        min_lag=k_min, max_lag=k_max, window_width=w, window_type=method
    )
    sg.zscore_all_data()
    sg.create_windows()
    sg.optimize_params()
    sg.fit_windows(n_trees=trees, show_progress=True, n_jobs=-1)

    ## Double mean aggregation
    all_amats = [item.edge_importance.to_numpy() for item in sg.window_list]
    all_amats = [np.mean(np.array(np.array_split(item, 3, axis=1)), axis=0) for item in all_amats]
    cmat = np.mean(np.array(all_amats), axis=0)

    cmat = np.abs(cmat)
    np.fill_diagonal(cmat, 0)

    return cmat