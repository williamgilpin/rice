import glob
import pandas as pd
import numpy as np
import warnings

import subprocess

import os
import sys

if not os.path.exists("dump"):
    os.makedirs("dump")

def run_ni(X, name="mi"):
    N, D = X.shape
    ## add a column of time points
    tpts = np.arange(N).astype(int)
    X2 = np.vstack((tpts, X.T)).T
    gene_names = [f"G{str(i).zfill(3)}" for i in range(D)]
    X2 = np.vstack(([" "] + gene_names, X2))
    np.savetxt("dump/temp.txt", X2.T, fmt="%s", delimiter="\t", newline='\n\n')
    # print("Saved successfully", flush=True)

    # command_str = "import Pkg; Pkg.add(\"NetworkInference\")"
    # subprocess.run(command_str.split(" "), capture_output=True, text=True)
    command_str = f"julia run_ni_benchmarks.jl {name}"
    subprocess.run(command_str.split(" "), capture_output=False, text=True)

    file_path = "dump/" + name + '_output.txt'
    df = pd.read_csv(file_path, sep='\t', header=None, names=['src', 'dest', 'weight'])
    # nodes = sorted(set(df['src']).union(set(df['dest'])))
    # node_to_index = {node: i for i, node in enumerate(nodes)}
    # D2 = len(nodes)
    # if D2 > D:
    #     warnings.warn(f"Number of nodes in the output ({D2}) is greater than the number of nodes in the input ({N}).")
    
    adj_matrix = np.zeros((D, D))
    for _, row in df.iterrows():
        # i, j = node_to_index[row['src']], node_to_index[row['dest']]
        i, j = int(row['src'][1:]), int(row['dest'][1:])
        adj_matrix[i, j] = row['weight']

    return adj_matrix
