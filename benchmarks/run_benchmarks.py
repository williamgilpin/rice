import os
import sys
import warnings
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from benchmark_suite import run_benchmark_model

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite on a specified dataset")
    parser.add_argument(
        "--dataset",
        choices=["mccalla", "kuramoto", "beeline", "smoketest", "dream4", "nonlinear"],
        default="dream4",
        help="Name of dataset to run benchmark on"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="List of methods to use for benchmarking",
    )
    parser.add_argument(
        "--nsplit",
        type=int,
        default=None,
        help="Number of chunks to split the data into. If None, the data will not be chunked."
    )
    parser.add_argument(
        "--goldtype",
        nargs="+",
        default=None,
        help="Gold standard network(s) to score against. For beeline: STRING (default), "
             "ChIP-seq, Non-ChIP. For mccalla: chipunion, KDUnion, "
             "chipunion_KDUnion_intersect (default)."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip any model-condition whose score file already exists in benchmark_output."
    )

    args = parser.parse_args()
    dataset = args.dataset.lower()
    models = args.model

    if dataset == "mccalla":
        from dataloaders import McCalla
        for item, condition in McCalla(goldtypes=args.goldtype):
            name_str = "_".join([str(item) for item in condition[0]])
            output_fname = f"mccalla_{name_str}_scores.txt"
            run_benchmark_model(item, output_fname, nval=100, DREAM4_flag=False, save_matrix=True, models=models, n_datasets=args.nsplit, skip_existing=args.skip_existing)

    elif dataset == "kuramoto":
        from dataloaders import Kuramoto
        for item, condition in Kuramoto():
            nval = 100
            name_str = "_".join([str(item) for item in condition])
            output_fname = f"Kuramoto_{name_str}_scores.txt"
            run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, save_matrix=True, models=models, n_datasets=args.nsplit, skip_existing=args.skip_existing)

    elif dataset == "beeline":
        from dataloaders import BEELINE
        for item, condition in BEELINE(goldtypes=args.goldtype):
            name_str = "_".join([str(item) for item in condition[0]])
            output_fname = f"BEELINE_{name_str}_scores.txt"
            run_benchmark_model(item, output_fname, nval=100, DREAM4_flag=False, save_matrix=True, models=models, n_datasets=args.nsplit, skip_existing=args.skip_existing)

    elif dataset == "smoketest":
        from dataloaders import Smoketest
        for item, (engine,) in Smoketest():
            nval = 100
            output_fname = f"smoketest_{engine}_scores.txt"
            run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, models=models, n_datasets=args.nsplit, skip_existing=args.skip_existing)

    elif dataset == "dream4":
        from dataloaders import DREAM4
        for item, (nval,) in DREAM4():
            output_fname = f"DREAM4_InSilico{nval}_scores.txt"
            run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=True, models=models, n_datasets=args.nsplit, skip_existing=args.skip_existing)

    elif dataset in ("twist", "nonlinear"):
        from dataloaders import NonlinearDataset
        for item, (organism_name, higher_order) in NonlinearDataset():
            nval = 100
            higher_order_str = "_higher_order" if higher_order else ""
            output_fname = f"nonlinear_scores_{organism_name}{higher_order_str}.txt"
            run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, models=models, n_datasets=args.nsplit, skip_existing=args.skip_existing)

    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)


## The __main__ guard is required: benchmark models that use multiprocessing
## (e.g. dynGENIE3 with nthreads > 1) spawn workers that re-import this module
## on macOS, which would otherwise re-execute the whole benchmark in each worker.
if __name__ == "__main__":
    main()


# sd = SingleDataset("../../../data/pancreas_top500.csv.gz" , species=10090)
# sd = SingleDataset("../../../data/pbmc68k_top500.csv.gz" , species=9606)
# sd = SingleDataset("../../../data/dentategyrus_top500.csv.gz" , species=10090)
# sd = SingleDataset("../../../data/bonemarrow_top500.csv.gz" , species=9606)
# for item, condition in sd:
#     name_str = sd.name
#     print(name_str)
#     output_fname = f"{name_str}_scores.txt"
