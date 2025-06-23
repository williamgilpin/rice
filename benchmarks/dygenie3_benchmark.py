import glob
import pandas as pd
import numpy as np
import warnings

import subprocess

import os
import sys

from dynGENIE3 import *
from GENIE3 import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from dygene.surrogate import CausalGraph
from dygene.utils import indices_to_adjacency
from dygene.metrics import compute_metrics_indices, compute_metrics

nval = 100
ORGANISM_NAME = "yeast"
HIGHER_ORDER = True

if HIGHER_ORDER:
    higher_order_str = "_higher_order"
else:
    higher_order_str = ""

DATADIR = f"/Users/william/program_repos/dygene/dygene/benchmarks/benchmark_datasets/gene_expression/{ORGANISM_NAME}_100genes{higher_order_str}"
files = glob.glob(DATADIR + "/*.npz")
data_name_pattern = f"{ORGANISM_NAME}{higher_order_str}_[0-9][0-9][0-9].npz"
output_genie3_fname = f"{ORGANISM_NAME}{higher_order_str}_scores_genie3.txt"
output_dygenie3_fname = f"{ORGANISM_NAME}{higher_order_str}_scores_dygenie3.txt"

data_fnames = sorted(glob.glob(os.path.join(DATADIR, data_name_pattern)))
results_genie3 = pd.DataFrame()
results_dygenie3 = pd.DataFrame()

print(f"{len(data_fnames)} files found.")
for data_fname in data_fnames[:25]:
    print(data_fname)
    ## get index file
    if HIGHER_ORDER:
        index_fname = data_fname.replace(
            f"order/{ORGANISM_NAME}{higher_order_str}", 
            f"order/{ORGANISM_NAME}{higher_order_str}_indices"
        )
    else:
        index_fname = data_fname.replace(
            f"genes/{ORGANISM_NAME}_",
            f"genes/{ORGANISM_NAME}_indices_"
        )     
    if not os.path.exists(index_fname):
        print(f"no index file found for {data_fname} and {index_fname}")
        continue

    seed_val = int(data_fname.split("_")[-1].replace(".npz", ""))
    
    print(data_fname, index_fname)
    X = np.load(data_fname, allow_pickle=True)["X"]
    all_true_pairs = np.load(index_fname, allow_pickle=True)["interactions"]
    amat = indices_to_adjacency(all_true_pairs, nval)

    out = GENIE3(X)
    cmat = out.copy()
    scores = compute_metrics(amat, cmat, verbose=False, check_transpose=False)
    results_genie3[seed_val] = scores
    results_genie3.transpose().to_csv(output_genie3_fname, sep="\t")

    time_points = np.arange(X.shape[0])
    out = dynGENIE3([X], [time_points])
    (VIM, alphas, prediction_score, stability_score, treeEstimators) = out
    cmat = VIM.copy()
    scores = compute_metrics(amat, cmat, verbose=False, check_transpose=False)
    results_dygenie3[seed_val] = scores
    results_dygenie3.transpose().to_csv(output_dygenie3_fname, sep="\t")
