"""
Rescore saved McCalla benchmark matrices against additional gold standards.

The McCalla expression data does not depend on the gold standard, so an
inferred network computed under one goldtype (e.g. chipunion) can be scored
against the others without re-running inference. This script loads the
matrices saved by run_benchmarks.py (in benchmark_output/matrices/) and
writes score files for the requested goldtypes, in the same format and with
the same naming convention as run_benchmarks.py.

Usage:
    python rescore_mccalla.py --model ensemble_noprune deepsem \
        --source_goldtype chipunion --goldtype KDUnion chipunion_KDUnion_intersect
"""
import argparse
import os

import numpy as np
import pandas as pd

from dataloaders import McCalla
from benchmark_suite import compute_metrics

parser = argparse.ArgumentParser(description="Rescore saved McCalla matrices")
parser.add_argument("--model", nargs="+", required=True, help="Model names to rescore")
parser.add_argument(
    "--source_goldtype",
    default="chipunion",
    help="Goldtype under which the matrices were originally saved",
)
parser.add_argument(
    "--goldtype",
    nargs="+",
    default=["KDUnion", "chipunion_KDUnion_intersect"],
    help="Goldtype(s) to score the saved matrices against",
)
parser.add_argument("--directory", default="benchmark_output", help="Benchmark output directory")
args = parser.parse_args()

loader = McCalla(goldtypes=args.goldtype)

for goldtype in args.goldtype:
    for ngenes in loader.ngenes:
        for celltype in loader.celltype:
            ## Gold standard matrix for the target goldtype. The expression data
            ## (and hence the gene ordering of the saved matrix) is identical
            ## across goldtypes.
            _, amat = loader.fetch_data((ngenes, goldtype, celltype))
            amat = amat[0]

            for model in args.model:
                src_fname = f"{model}_mccalla_{ngenes}_{args.source_goldtype}_{celltype}_scores.txt"
                matrix_path = os.path.join(args.directory, "matrices", "matrix_0_" + src_fname)
                if not os.path.exists(matrix_path):
                    print(f"skipping (no saved matrix): {matrix_path}")
                    continue
                cmat = np.loadtxt(matrix_path)
                if cmat.shape != amat.shape:
                    print(f"skipping (shape mismatch {cmat.shape} vs {amat.shape}): {matrix_path}")
                    continue

                ## Mirror the scoring in benchmark_suite.run_benchmark
                amat_h, cmat_h = amat.copy(), cmat.copy()
                np.fill_diagonal(cmat_h, 0)
                np.fill_diagonal(amat_h, 0)
                scores = compute_metrics(amat_h, cmat_h, verbose=False, check_transpose=True, hollow=True)

                ## Carry over the inference time from the source run
                src_path = os.path.join(args.directory, src_fname)
                scores["time"] = None
                if os.path.exists(src_path):
                    src = pd.read_csv(src_path, sep="\t", index_col=0)
                    if "time" in src.columns:
                        scores["time"] = src["time"].iloc[0]

                results = pd.DataFrame()
                results[0] = scores
                out_fname = f"{model}_mccalla_{ngenes}_{goldtype}_{celltype}_scores.txt"
                results.transpose().to_csv(os.path.join(args.directory, out_fname), sep="\t")
                print(f"wrote {out_fname}")
