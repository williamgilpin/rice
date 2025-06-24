import glob
import pandas as pd
import numpy as np
import warnings

import subprocess

import os
import sys

def run_scode(X, tempdir="", num_trials=5, num_iterations=100, dval=4):
    """
    A wrapper for running the SCODE algorithm on a dataset using the Julia 
    command line interface. 
    """
    ## Save to disk
    np.savetxt("dump/scode_temp.txt", X.T, fmt="%.4f", delimiter="\t")
    tpts = np.arange(0, X.shape[0])
    tvals = np.linspace(0, 1, X.shape[0])
    time_vals = np.vstack((tpts, tvals)).T
    # shape = (n, 2)
    np.savetxt("dump/scode_time.txt", time_vals, fmt="%.4f", delimiter="\t")

    C, G = X.shape
    dval = G

    if not os.path.exists("dump"):
        os.makedirs("dump")
    
    all_cmat = []
    for i in range(num_trials):
        print(f"trial {i}", flush=True)
        command_str = "julia -e 'import Pkg; Pkg.add(\"CSV\")'"
        subprocess.run(command_str.split(" "), capture_output=True, text=True)
        command_str = "julia -e 'import Pkg; Pkg.add(\"DataFrame\")'"
        subprocess.run(command_str.split(" "), capture_output=True, text=True)
        command_str = f"julia SCODE.jl dump/scode_temp.txt dump/scode_time.txt dump {G} {dval} {C} {num_iterations}"
        result = subprocess.run(command_str.split(" "), capture_output=True, text=True)
        print(result.stdout, flush=True)
        print("done", flush=True)

        # load the matrix in dump/A.txt
        cmat = np.loadtxt("dump/A.txt", delimiter="\t")
        all_cmat.append(np.abs(cmat))
    all_cmat = np.array(all_cmat)
    cmat = np.mean(all_cmat, axis=0)

    return cmat