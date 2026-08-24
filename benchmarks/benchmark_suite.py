import os
import sys
import warnings

import pandas as pd
import numpy as np

SAVE_MATRIX = False
SKIP_EXISTING = False

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rice.models import CausalDetection
from rice.metrics import compute_metrics

import time
def run_benchmark(all_X, all_amat, method, output_fname, hollow=True,
                  n_datasets=None,
                  batch=False,
                  save_matrix=None,
                  directory="benchmark_output"):
    """
    Run a benchmark on the given dataset.

    Args:
        X (np.ndarray): The dataset to run the benchmark on. Should have 
            shape (n_times, n_genes).
        amat (np.ndarray): The ground truth adjacency matrix.
        method (callable): The method to fit the model. This should return
            an (n_genes, n_genes) matrix.
        output_fname (str): The name of the output file.
        hollow (bool): Whether to compute the metrics while excluding self-loops.
        n_datasets (int): The number of chunks to split the data into. If
            None, the data will not be chunked.
        batch (bool): For benchmark models that accept batch processing. If false,
            the method will be called for each dataset and the final result will be
            the average matrix.
        save_matrix (bool): Whether to save the output matrix in a "matrices"
            subdirectory of `directory`. If None, falls back to the module-level
            SAVE_MATRIX flag.
        directory (str): The directory to save the output file.

    Returns:
        None
    """
    if save_matrix is None:
        save_matrix = SAVE_MATRIX
    # n_datasets = None # For non-DREAM datasets
    os.makedirs(directory, exist_ok=True)
    if SKIP_EXISTING and os.path.exists(os.path.join(directory, output_fname)):
        print(f"Skipping existing output: {output_fname}", flush=True)
        return None
    results = pd.DataFrame()
    for data_index, (X, amat) in enumerate(zip(all_X, all_amat)):

        try:
            start = time.time()
            if not n_datasets:
                cmat = method(X)
            else:
                Xall = np.array_split(X, n_datasets)
                if batch:
                    cmat = method(Xall)
                else:
                    all_cmat = []
                    for Xv in Xall:
                        cmat = method(Xv)
                        all_cmat.append(cmat)
                    cmat = np.mean(np.array(all_cmat), axis=0)
            finish = time.time()
            if hollow:
                np.fill_diagonal(cmat, 0)
                np.fill_diagonal(amat, 0)
            scores = compute_metrics(amat, cmat, verbose=False, check_transpose=True, hollow=True)
            scores["time"] = finish - start
        except Exception as e:
            warnings.warn(f"Error: {e}")
            amat_dummy = np.ones_like(amat)
            cmat_dummy = np.ones_like(amat)
            scores = compute_metrics(amat_dummy, cmat_dummy, verbose=False, check_transpose=True, hollow=True)
            scores = {key: None for key in scores}
            scores["time"] = None
            cmat = np.ones_like(amat) * np.nan

        results[data_index] = scores
        results.transpose().to_csv(os.path.join(directory, output_fname), sep="\t")

        if save_matrix:
            matrix_dir = os.path.join(directory, "matrices")
            os.makedirs(matrix_dir, exist_ok=True)
            ## np.savetxt writes ~25 bytes per entry in its default format
            est_bytes = cmat.size * 25
            if est_bytes > 1e9:
                warnings.warn(
                    f"Saved matrix for {output_fname} will be approximately "
                    f"{est_bytes / 1e9:.1f} GB on disk"
                )
            np.savetxt(
                os.path.join(matrix_dir, f"matrix_{data_index}_" + output_fname),
                cmat
            )

    return results


## Import benchmarking datasets
# from dataloaders import Smoketest, BEELINE, DREAM4, NonlinearDataset, SingleDataset


def chunk_benchmark(X, process, n_chunks=2):
    """
    Split the data into chunks and process each chunk separately.

    Args:
        X (ndarray): The input data, of shape (T, D).
        process (callable): The function that processes an array, producing an output
            of shape (D, D).
        n_chunks (int): The number of chunks to split the data into.

    Returns:
        ndarray: The processed data of shape (D, D).
    """
    T, D = X.shape
    indices = np.array_split(np.arange(D), n_chunks)
    blocks = []
    for inds_i in indices:
        row_blocks = []
        for inds_j in indices:
            X_i = X[:, inds_i]
            X_j = X[:, inds_j]
            # process should accept two arrays and return the corresponding block.
            row_blocks.append(process(X_i, X_j))
        blocks.append(np.hstack(row_blocks))
    return np.vstack(blocks)


model_list = [
    # "regdiffusion", #
    # "deepsem", # Need to re-run
    "ensemble_prune",
    "ensemble_noprune",
    "isolated_noprune",
    "ccm",
    "smap",
    # "swing", # stalled out
    # "wcorr",
    # "scode", ##
    # "mi",
    # "clr",
    # "puc",
    # "pidc",
    # "dyngenie3", # slow
    # "grenadine_clr", 
    # "grenadine_genie3", # stalled out
    # "bayesian_ridge",
    # "svr", # stalled out
    # "tigress", # Very slow but runs
    # "elastica", # Very slow but runs
    # "adaboost",
    # "grnboost2", # Slow for large networks
    # "xgenie3",
    # "wasserstein_gren",
    # "energy",
    # "wilcoxon",
    # "glasso",
    # "genie3",
    # "sincerities",
    # "aracne" # Expensive for large networks
]


def run_benchmark_model(
        item, 
        output_fname, 
        DREAM4_flag=False, 
        nval=None, 
        save_matrix=False, 
        models=None,
        n_datasets=None,
        skip_existing=False,
    ):
    """
    Run a benchmark suite on the given dataset.

    Args:
        item (tuple): The dataset to run the benchmark on.
        output_fname (str): The name of the output file.
        DREAM4_flag (bool): Whether to run the benchmark on the DREAM4 dataset.
        nval (int): The number of samples to use for each dataset.
        save_matrix (bool): Whether to save the output matrix.
        models (list): A list of models to run the benchmark on.
        n_datasets (int): Split the dataset into this many chunks and run the benchmark 
            on each chunk. If None, the dataset will not be chunked.

    Returns:
        None
    """
    global SAVE_MATRIX
    global SKIP_EXISTING
    global model_list
    SAVE_MATRIX = save_matrix
    SKIP_EXISTING = skip_existing

    ## If a list of models is provided, use it instead of the default model list
    if models is not None:
        model_list = models
    
    ## Loop over benchmark models
    for name in model_list:
        if name not in model_list:
            continue
        print("\n", name, "\n", flush=True)

        if name == "regdiffusion":
            import regdiffusion as rd
            def fit_model(X):
                rd_trainer = rd.RegDiffusionTrainer(X.copy())
                rd_trainer.train()
                cmat = rd_trainer.get_adj()
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname)

        if name == "deepsem":
            from deepsem import run_deepsem
            ## Subsample cells so that the full deepsem benchmark finishes in hours
            ## rather than days; deepsem treats cells as i.i.d. samples
            def fit_model(X):
                return run_deepsem(X, max_cells=128)
            if DREAM4_flag:
                if nval == 10:
                    num_datasets = 5
                elif nval == 100:
                    num_datasets = 10
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=num_datasets)
            else:
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "ensemble_prune":
            def fit_model(X):
                model = CausalDetection(
                    d_embed=3, 
                    neighbors="simplex", 
                    forecast="smap", 
                    ensemble=True, 
                    prune_indirect=True,
                )
                cmat = model.fit(X)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "ensemble_noprune":
            def fit_model(X):
                model = CausalDetection(
                    d_embed=3, 
                    neighbors="simplex", 
                    forecast="smap", 
                    ensemble=True, 
                )
                cmat = model.fit(X)
                # cmat = cmat / cmat.sum(axis=1, keepdims=True)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "ensemble_noprune_dense":
            ## ensemble_noprune with a denser library-size schedule. The default
            ## dilation_factor of 1.5 trades some accuracy for speed on small
            ## datasets; 1.05 approximately recovers the original dense schedule.
            def fit_model(X):
                model = CausalDetection(
                    d_embed=3,
                    neighbors="simplex",
                    forecast="smap",
                    ensemble=True,
                    dilation_factor=1.05,
                )
                cmat = model.fit(X)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "isolated_noprune":
            def fit_model(X):
                model = CausalDetection(
                    d_embed=3, 
                    neighbors="simplex", 
                    forecast="smap", 
                    ensemble=False, 
                )
                cmat = model.fit(X)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "ccm":
            def fit_model(X):
                model = CausalDetection(
                    d_embed=3, 
                    neighbors="knn", 
                    forecast="sum", 
                    ensemble=False, 
                )
                cmat = model.fit(X)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname)

        if name == "smap":
            def fit_model(X):
                model = CausalDetection(
                    d_embed=3, 
                    neighbors="knn", 
                    forecast="smap", 
                    ensemble=False, 
                )
                cmat = model.fit(X)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)


        # Baseline models

        if name == "swing":
            from run_swing import run_swing
            if DREAM4_flag:
                def fit_model(X):
                    return run_swing(X)
                if nval == 10:
                    num_datasets = 5
                elif nval == 100:
                    num_datasets = 10
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=num_datasets)
            else:
                def fit_model(X):
                    return run_swing(X)
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "wcorr":
            from models import LaggedCorrelations
            ## Cap cells: the number of lags scales with the number of timepoints
            ## (max_lag=0.3), so the 36k-cell McCalla datasets would require ~20k
            ## lagged correlation matrices (~170 GB). BEELINE datasets are unaffected.
            def fit_model(X):
                max_cells = 2000
                if X.shape[0] > max_cells:
                    idx = np.linspace(0, X.shape[0] - 1, max_cells).astype(int)
                    X = X[idx]
                model = LaggedCorrelations(method="spearman", max_lag=0.3)
                cmat = model.fit(X)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "scode":
            from scode import run_scode
            def fit_model(X):
                return run_scode(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "mi":
            from ni import run_ni
            def fit_model(X):
                return run_ni(X, name="mi")
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "clr":
            from ni import run_ni
            def fit_model(X):
                return run_ni(X, name="clr")
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "puc":
            from ni import run_ni
            def fit_model(X):
                return run_ni(X, name="puc")
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "pidc":
            from ni import run_ni
            def fit_model(X):
                return run_ni(X, name="pidc")
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "dyngenie3":
            from dyngenie3_wrapper import run_dynGENIE3
            if DREAM4_flag:
                def fit_model(X):
                    return run_dynGENIE3(X)
                if nval == 10:
                    num_datasets = 5
                elif nval == 100:
                    num_datasets = 10
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=num_datasets, batch=True)
            else:
                ## max_cells caps the huge McCalla datasets (36k cells for mESC);
                ## BEELINE datasets are all under 2000 cells and are unaffected
                def fit_model(X):
                    return run_dynGENIE3(X, nthreads=10, max_cells=2000)
                run_benchmark(*item, fit_model, name + "_" + output_fname, batch=True, n_datasets=n_datasets)

        if name == "grenadine_clr":
            from grenadine_models import clr as clr2
            def fit_model(X):
                return clr2(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "grenadine_genie3":
            from grenadine_models import genie3 as genie3_2
            def fit_model(X):
                return genie3_2(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "bayesian_ridge":
            from grenadine_models import bayesian_ridge
            def fit_model(X):
                return bayesian_ridge(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "svr":
            from grenadine_models import svr
            def fit_model(X):
                return svr(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "tigress":
            from grenadine_models import tigress
            def fit_model(X):
                return tigress(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "elastica":
            from grenadine_models import elastica
            def fit_model(X):
                return elastica(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "adaboost":
            from grenadine_models import adaboost
            def fit_model(X):
                return adaboost(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "grnboost2":
            from grenadine_models import grnboost2
            def fit_model(X):
                return grnboost2(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "xgenie3":
            from grenadine_models import xgenie3
            def fit_model(X):
                return xgenie3(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "wasserstein_gren":
            from grenadine_models import wasserstein_gren
            def fit_model(X):
                return wasserstein_gren(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "energy":
            from grenadine_models import energy
            def fit_model(X):
                return energy(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "wilcoxon":
            from grenadine_models import wilcoxon
            def fit_model(X):
                return wilcoxon(X)
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "glasso":
            from sklearn.covariance import GraphicalLassoCV
            def fit_model(X):
                model = GraphicalLassoCV(alphas=np.logspace(-6, -2, 5))
                model.fit(X)
                cmat = model.covariance_.copy()
                np.fill_diagonal(cmat, 0)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)
        
        if name == "genie3":
            from GENIE3 import GENIE3
            def fit_model(X):
                out = GENIE3(X)
                cmat = out.copy()
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)

        if name == "sincerities":
            from sincerities import SINCERITIES
            if DREAM4_flag:
                def fit_model(X):
                    model = SINCERITIES(X, t=None, dd_metric='ks')
                    cmat = model.fit()
                    cmat = np.abs(cmat)
                    np.fill_diagonal(cmat, 0)
                    return cmat
                if nval == 10:
                    num_datasets = 5
                elif nval == 100:
                    num_datasets = 10
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=num_datasets)
            else:
                def fit_model(X):
                    model = SINCERITIES(X, t=None, dd_metric='ks')
                    cmat = model.fit()
                    cmat = np.abs(cmat)
                    np.fill_diagonal(cmat, 0)
                    return cmat
                run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)
        
        if name == "aracne":
            from aracne import ARACNE
            def fit_model(X):
                model = ARACNE(n_permutations=5, random_state=0)
                cmat = model.fit_transform(np.copy(X))
                cmat = np.abs(cmat)
                np.fill_diagonal(cmat, 0)
                return cmat
            run_benchmark(*item, fit_model, name + "_" + output_fname, n_datasets=n_datasets)



