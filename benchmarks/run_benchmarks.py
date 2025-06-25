import os
import sys
import warnings
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from benchmark_suite import run_benchmark_model

import argparse

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

args = parser.parse_args()
dataset = args.dataset.lower()
models = args.model

if dataset == "mccalla":
    from dataloaders import McCalla
    for item, condition in McCalla():
        n_datasets = item[0].shape[1] // 1000
        name_str = "_".join([str(item) for item in condition[0]])
        output_fname = f"mccalla_{name_str}_scores.txt"
        run_benchmark_model(item, output_fname, nval=100, DREAM4_flag=False, save_matrix=False, models=models, n_datasets=n_datasets)

elif dataset == "kuramoto":
    from dataloaders import Kuramoto
    for item, condition in Kuramoto():
        nval = 100
        name_str = "_".join([str(item) for item in condition])
        output_fname = f"Kuramoto_{name_str}_scores.txt"
        run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, save_matrix=True, models=models)

elif dataset == "beeline":
    from dataloaders import BEELINE
    for item, condition in BEELINE():
        nval = 100
        name_str = "_".join([str(item) for item in condition[0]])
        output_fname = f"BEELINE_{name_str}_scores.txt"
        run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, models=models)

elif dataset == "smoketest":
    from dataloaders import Smoketest
    for item, (engine,) in Smoketest():
        nval = 100
        output_fname = f"smoketest_{engine}_scores.txt"
        run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, models=models)

elif dataset == "dream4":
    from dataloaders import DREAM4
    for item, (nval,) in DREAM4():
        output_fname = f"DREAM4_InSilico{nval}_scores.txt"
        run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=True, models=models)

elif dataset == "twist":
    from dataloaders import NonlinearDataset
    for item, (organism_name, higher_order) in NonlinearDataset():
        nval = 100
        higher_order_str = "_higher_order" if higher_order else ""
        output_fname = f"nonlinear_scores_{organism_name}{higher_order_str}.txt"
        run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, models=models)

else:
    print(f"Unknown dataset: {dataset}")
    sys.exit(1)

# sd = SingleDataset("../../../data/pancreas_top500.csv.gz" , species=10090)
# sd = SingleDataset("../../../data/pbmc68k_top500.csv.gz" , species=9606)
# sd = SingleDataset("../../../data/dentategyrus_top500.csv.gz" , species=10090)
# sd = SingleDataset("../../../data/bonemarrow_top500.csv.gz" , species=9606)
# for item, condition in sd:
#     name_str = sd.name
#     print(name_str)
#     output_fname = f"{name_str}_scores.txt"

# import os
# import sys
# import warnings
# import numpy as np

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# from run_benchmark import run_benchmark_model

# sd = SingleDataset("../../../data/pancreas_top500.csv.gz" , species=10090)
# sd = SingleDataset("../../../data/pbmc68k_top500.csv.gz" , species=9606)
# sd = SingleDataset("../../../data/dentategyrus_top500.csv.gz" , species=10090)
# sd = SingleDataset("../../../data/bonemarrow_top500.csv.gz" , species=9606)
# for item, condition in sd:
#     name_str = sd.name
#     print(name_str)
#     output_fname = f"{name_str}_scores.txt"


# from dataloaders import McCalla
# for item, condition in McCalla():
#     name_str = "_".join([str(item) for item in condition[0]])
#     print(name_str)
#     output_fname = f"mccalla_{name_str}_scores.txt"
#     run_benchmark_model(item, output_fname, nval=100, DREAM4_flag=False, save_matrix=False)

# from dataloaders import Kuramoto
# for item, condition in Kuramoto():
#     nval = 100
#     print(condition)
#     name_str = "_".join([str(item) for item in condition])
#     print(name_str)
#     output_fname = f"Kuramoto_{name_str}_scores.txt"
#     run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False, save_matrix=True)

# from dataloaders import BEELINE
# for item, condition in BEELINE():
#     nval = 100
#     BEELINE_flag = True
#     name_str = "_".join([str(item) for item in condition[0]])
#     print(name_str)
#     output_fname = f"BEELINE_{name_str}_scores.txt"
#     run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False)

# from dataloaders import Smoketest
# for item, condition in Smoketest():
#     nval = 100
#     engine, = condition
#     output_fname = f"smoketest_{engine}_scores.txt"
#     run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False)
    
# from dataloaders import DREAM4
# for item, condition in DREAM4():
#     nval, = condition
#     output_fname = f"DREAM4_InSilico{nval}_scores.txt"
#     run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=True)

# from dataloaders import NonlinearDataset
# for item, condition in NonlinearDataset():
#     print(condition)
#     nval = 100
#     organism_name, higher_order = condition
#     if higher_order:
#         higher_order_str = "_higher_order"
#     else:
#         higher_order_str = ""
#     output_fname = f"nonlinear_scores_{organism_name}{higher_order_str}.txt"

#     run_benchmark_model(item, output_fname, nval=nval, DREAM4_flag=False)