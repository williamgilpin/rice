
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

## get path of this file
file_path = os.path.dirname(os.path.abspath(__file__))
## add current directory to path
sys.path.append(file_path)
from dataloader_utils import fetch_interaction_matrix

import itertools
from itertools import product

## add to path
sys.path.append(os.path.join(file_path, ".."))
from rice.utils import indices_to_adjacency

def quantile_binning(data, n=100, agg=np.nanmean):
    """
    Bin data into evenly spaced quantiles and apply an aggregation function to each bin

    Args:
        data (np.ndarray): data to bin
        n (int): number of bins
        agg (function): aggregation function to apply to each bin

    Returns:
        quantiles (np.ndarray): quantile values
        bin_vals (np.ndarray): aggregated values for each bin
    """
    quantiles = np.linspace(0, 1, n+1)
    bin_edges = np.quantile(data, quantiles)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    indices = np.digitize(data, bin_edges, right=True) - 1
    indices = np.clip(indices, 0, n-1)
    bin_vals = np.array([
        agg(data[indices == i]) if np.any(indices == i) else np.nan
        for i in range(n)
    ])
    return bin_centers, bin_vals

def make_goldstandard_matrix(gene_names, gold_links, interactions=None,
                             mask_tf=True, symmetric=False):
    """Given a list of gene names and a list of known interactions, create a gold 
    standard matrix.

    Args:
        gene_names (list): A list of gene names. The order of the genes in the list will
            match the order of the rows and columns in the returned gold standard matrix.
        gold_links (list): A list of pairs of known interactions, of shape (n_links, 2).
        interactions (np.ndarray): A matrix of interaction strengths or signs, of 
            shape (n_links,). If None, all interactions are assumed to be positive.
        mask_tf (bool): If True, mask out the rows and columns corresponding to the 
            transcription factors in the gold standard matrix.
        symmetric (bool): If True, make the gold standard matrix symmetric.

    Returns:
        np.ndarray: A gold standard matrix.
    """
    if interactions is None:
        interactions = np.ones(len(gold_links))

    upstream_names, downstream_names = gold_links.T
    n_genes = len(gene_names)
    amat = np.zeros((n_genes, n_genes))
    for k, (gene1, gene2) in enumerate(gold_links):
        ## Ignore links not included in the gene_names list
        if gene1 not in gene_names or gene2 not in gene_names:
            continue
        i, j = list(gene_names).index(gene1), list(gene_names).index(gene2)
        amat[i, j] = interactions[k]
        if symmetric:
            amat[j, i] = interactions[k]

    # Mask out the rows and columns wbere no gene was tested
    if mask_tf:
        upstream_mask = np.array([item in upstream_names for item in gene_names])
        downstream_mask = np.array([item in downstream_names for item in gene_names])
        amat[~upstream_mask] += np.nan
        amat[:, ~downstream_mask] += np.nan

    return amat

class DataLoader:
    """
    A base dataloader class for benchmarking
    
    Attributes:
        missing (str): The missing value to use. Should be "nan" or "zero".
        standardize (bool): Whether to standardize the data.
        bootstrap (bool): Whether to bootstrap the data.
    """

    def __init__(self, standardize=False, missing="nan", bootstrap=False):
        self.standardize = standardize
        self.missing = missing
        self.bootstrap = bootstrap
        self.conditions = [[None], [None]]
        
    def __iter__(self):
        for condition in self.conditions:
            yield self.fetch_data(*condition), condition

    def bootstrap_data(self, X):
        """
        Bootstrap a dataset along the first dimension.

        Args:
            X (np.ndarray): The dataset to bootstrap.

        Returns:
            np.ndarray: The bootstrapped dataset.
        """
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[bootstrap_indices]


class Smoketest(DataLoader):
    """A dataloader for the Smoketest dataset"""

    def __init__(self):
        super().__init__()
        self.conditions = [["gaussian"], ["logistic"]]

    def fetch_data(self, engine):
        
        DATADIR = file_path + "/benchmark_datasets/smoketest/"
        # smoketest_logistic_amat_012.csv
        data_name_pattern = f"smoketest_{engine}_data_[0-9][0-9][0-9].csv"
        amat_name_pattern = f"smoketest_{engine}_amat_[0-9][0-9][0-9].csv"
        data_fnames = sorted(glob.glob(os.path.join(DATADIR, data_name_pattern)))
        amat_fnames = sorted(glob.glob(os.path.join(DATADIR, amat_name_pattern)))
        assert len(data_fnames) == len(amat_fnames), "Number of data and ground truth files do not match."
        print(f"{len(data_fnames)} files found.")
        
        all_X, all_amat = [], []
        for i in range(15):
            
            data_fname = data_fnames[i]
            X = np.loadtxt(data_fname, delimiter=",")
            amat = np.loadtxt(amat_fnames[i], delimiter=",")
            np.fill_diagonal(amat, 0)

            all_X.append(X.copy())
            all_amat.append(amat.copy())

        return all_X, all_amat


class NonlinearDataset(DataLoader):
    """A dataloader for the Nonlinear Dynamics dataset"""

    def __init__(self):
        super().__init__()
        self.conditions = itertools.product(["yeast", "ecoli"], [False, True])

    def fetch_data(self, organism_name, higher_order):

        nval = 100
        if higher_order:
            higher_order_str = "_higher_order"
        else:
            higher_order_str = ""

        # DATADIR = f"/Users/william/program_repos/dygene/dygene/benchmarks/benchmark_datasets/"
        DATADIR = file_path + "/benchmark_datasets/"
        DATADIR += f"gene_expression/{organism_name}_100genes{higher_order_str}"
        # print(DATADIR)
        data_name_pattern = f"{organism_name}{higher_order_str}_[0-9][0-9][0-9].npz"
        data_fnames = sorted(glob.glob(os.path.join(DATADIR, data_name_pattern)))
        # files = glob.glob(DATADIR + "/*.npz")
        print(f"{len(data_fnames)} files found.")
        

        all_X, all_amat = [], []
        for i in range(25):
        # for i in range(5):
            data_fname = data_fnames[i]
            X = np.load(data_fname, allow_pickle=True)["X"]
            # X = X[:50] # Realistic gene expression time series are short
            # X = X[:75] # Too hard
            X = X[:100] # About right
            # X = X[:150] # Too easy
            # print("X", X.shape)

            if higher_order:
                index_fname = data_fname.replace(
                    f"order/{organism_name}{higher_order_str}", 
                    f"order/{organism_name}{higher_order_str}_indices"
                )
            else:
                index_fname = data_fname.replace(
                    f"genes/{organism_name}_",
                    f"genes/{organism_name}_indices_"
                )     
            if not os.path.exists(index_fname):
                print(f"no index file found for {data_fname} and {index_fname}")
                continue

            all_true_pairs = np.load(index_fname, allow_pickle=True)["interactions"]
            amat = indices_to_adjacency(all_true_pairs, nval)

            all_X.append(X.copy())
            all_amat.append(amat.copy())

        return all_X, all_amat




class DREAM4(DataLoader):
    """A dataloader for the DREAM4 dataset"""

    def __init__(self):
        super().__init__()
        self.conditions = [[10], [100]]

    # def fetch_data(self, nval, data_index):

    def fetch_data(self, n):
        """
        Retrieve the DREAM4 datasets for the given number of genes and data index.

        Args:
            n (int): The number of genes in the dataset.

        Returns:
            np.ndarray: The dataset.
            np.ndarray: The ground truth adjacency matrix.
        
        """
        DATADIR = file_path + "/benchmark_datasets/"
        # DATADIR += f"gene_expression/{organism_name}_100genes{higher_order_str}"
        DATA_DIR = DATADIR + f"DREAM4_InSilico_Size{n}/"
        GROUND_DIR = DATADIR + f"DREAM4_Challenge2_GoldStandards/Size {n}/"

        all_X, all_amat = [], []
        for i in range(1, 6):
            datapath = f"insilico_size{n}_{i}/insilico_size{n}_{i}_timeseries.tsv"
            df = pd.read_csv(os.path.join(DATA_DIR, datapath), sep="\t")
            df.set_index(df.columns[0], inplace=True)
            gene_names = np.array(df.columns)
            X = df.values
            if self.standardize:
                X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

            groundpath = f"DREAM4_GoldStandard_InSilico_Size{n}_{i}.tsv"

            ground_arr = np.loadtxt(os.path.join(GROUND_DIR, groundpath), delimiter="\t", dtype=str)
            interactions = np.array(ground_arr[:, -1]).astype(float)
            gold_links = ground_arr[:, :2]
            amat = make_goldstandard_matrix(gene_names, 
                                            gold_links, 
                                            interactions=interactions,
                                            mask_tf=True, 
                                            symmetric=False
                                            )

            all_X.append(X.copy())
            all_amat.append(amat.copy())
        return all_X, all_amat

# from umap.umap_ import fuzzy_simplicial_set
relu = lambda x: np.maximum(0, x)
from scipy.optimize import fsolve
from sklearn.neighbors import NearestNeighbors
def simplex_neighbors(X, metric='euclidean', k=20, tol=1e-6):
    """
    Compute the distance between points in a dataset using the simplex distance metric.

    Args:
        X (np.ndarray): dataset of shape (n, d)
        Y (np.ndarray): dataset of shape (m, d)
        metric (str): distance metric to use
        k (int): number of nearest neighbors to use in the distance calculation
        tol (float): tolerance for the distance calculation

    Returns:
        np.ndarray: distance matrix of shape (n, m)

    """
    tree = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric)
    tree.fit(X)
    dists, idx  = tree.kneighbors(X)
    dists, idx = dists[:, 1:].T, idx[:, 1:].T
    rhos = dists[0]
    sigmas = np.array([find_sigma(drow, tol=tol)[0] for drow in dists.T])
    sigmas += tol # Add a small tolerance to avoid division by zero
    wgts = np.exp(-relu(dists - rhos[None, :]) / sigmas[None, :])
    return wgts, idx, sigmas

def find_sigma(dists, tol=1e-6):
    """
    Given a list of distances to k nearest neighbors, find the sigma for each point

    Args:
        dists (np.ndarray): A matrix of shape (k,)
        tol (float): The tolerance for the sigma

    Returns:
        float: The sigma corresponding to the neighborhood scale
        np.ndarray: The transformed distances
    """
    k = dists.shape[0]
    rho = np.min(dists)
    func = lambda sig: sum(np.exp(-relu(dists - rho) / (sig + tol))) - np.log2(k)
    jac = lambda sig: sum(np.exp(-relu(dists - rho) / (sig + tol)) * relu(dists - rho)) / (sig + tol)**2
    sigma = fsolve(func, rho, fprime=jac, xtol=tol)[0]
    dists_transformed = np.exp(-relu(dists - rho) / (sigma + tol))
    return sigma, dists_transformed

def calculate_sigma(X0, d_embed=4, tol=1e-6, channelwise=True):
    """Given a matrix of time series, calculate the sigma for each time series.

    Args:
        X0: (ntx, d) matrix of time series
        d_embed: embedding dimension
        tol: tolerance for simplex neighbors
        channelwise: whether to embed each time series separately or not 

    Returns:
        all_sig: (ntx, d) matrix of sigmas if channelwise is True, otherwise 
            (ntx, 1) matrix of sigmas
    """
    X = X0.squeeze().copy()
    if channelwise:
        Xe = embed_ts(X, m=d_embed)
    else:
        Xe = X[None, ...]
    m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
    all_sig = list()
    for Xe_i in Xe:
        wgts, idx, sig = simplex_neighbors(Xe_i, k=min(ntx - 1, d_embed + 1), tol=tol)
        all_sig.append(sig)
    all_sig = np.array(all_sig)
    if channelwise:
        all_sig = np.pad(all_sig, [[0, 0], [0, d_embed - 1]], mode="edge")
    return all_sig

def compute_pseudotime(X0, root_index=None, n_components=40, return_ordering=False):
    """
    Compute pseudotime using PCA and diffusion pseudotime

    Args:
        X0 (np.ndarray): The data matrix of shape (n_genes, n_timepoints)
        root_index (int): The index of the root cell. If None, the root cell is assumed to
            be the cell with the smallest DC1 value.
        n_components (int): The number of components to use in the PCA.
        return_ordering (bool): If True, return the indices used to sort the data matrix.

    Returns:
        np.ndarray: The data matrix sorted by pseudotime.
        np.ndarray: The pseudotime values.
    """

    # Use PCA pseudotime
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=1)
    # ptvals = pca.fit_transform(X).flatten()
    # sort_inds = np.argsort(ptvals)
    # X, ptvals = X[sort_inds], ptvals[sort_inds]

    ## Alternative pseudotime doens't help much
    # import scanpy.external as sce
    # import scanpy as sc
    # adata = sc.AnnData(X)
    # sc.pp.pca(adata, n_comps=10)
    # sce.tl.palantir(adata)
    # start_cell = str(root_index)
    # pr_res = sce.tl.palantir_results(
    #     adata,
    #     early_cell=start_cell,
    #     ms_data='X_palantir_multiscale',
    #     num_waypoints=500,
    # )
    # ptvals = np.array(pr_res.pseudotime).copy()
    # sort_inds = np.argsort(ptvals)
    # X, ptvals = X[sort_inds], ptvals[sort_inds]


    import scanpy as sc
    X = X0.copy()
    adata = sc.AnnData(X)
    # wgts, idx, sig = simplex_neighbors(X, k=(X.shape[1] + 1), tol=1e-6)
    # sig = (sig - np.mean(sig)) / np.std(sig)
    # all_sig = sig[:, None] * np.ones(X.shape[0])[None, :]
    # adata = sc.AnnData(np.hstack([all_sig, X]))

    if root_index is None:
        sc.pp.neighbors(adata, method='umap')
        sc.tl.diffmap(adata, n_comps=10)  # Ensure diffusion map is computed
        root_index = np.argmin(adata.obsm['X_diffmap'][:, 0])  # Smallest DC1 value

    sc.pp.pca(adata, n_comps=1)
    sc.pp.neighbors(adata, method='umap')
    adata.uns['iroot'] = np.argmin(root_index) # Root cell from original
    sc.tl.diffmap(adata, n_comps=n_components)
    sc.tl.dpt(adata, n_dcs=n_components)
    ptvals = np.array(adata.obs["dpt_pseudotime"])
    sort_inds = np.argsort(ptvals)
    X, ptvals = X[sort_inds], ptvals[sort_inds]
    if return_ordering:
        return X, ptvals, sort_inds
    return X, ptvals


## get current directory
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))


class SingleDataset(DataLoader):
    """
    A generic dataloader for a single dataset"""

    def __init__(self, fpath, species=9606):
        super().__init__()   
        self.fpath = fpath
        self.species = species
        self.name = os.path.basename(self.fpath).split(".")[0]

    def fetch_data(self, condition, metadata=False):
        Xdf = pd.read_csv(self.fpath, index_col=0)
        gene_names = list(Xdf.columns)
        X = Xdf.values.copy().astype(float)
        X += np.random.normal(0, 1e-8, X.shape) # Jitter the data
        X, ptvals = compute_pseudotime(X)
        amat = fetch_interaction_matrix(gene_names, species=self.species, min_confidence=0)
        amat = amat.values
        amat[amat > 1] = 1
        if metadata:
            gene_names = list(Xdf.columns)
            return X[None, :], amat[None, :], gene_names
        return X[None, :], amat[None, :]



class Kuramoto(DataLoader):
    """
    A dataloader for the Kuramoto dataset. This dataset was generated by simulating the
    Kuramoto model with 100 oscillators and nearest-neighbor coupling. The model is simulated
    for 10000 timepoints and 400 total time units. The phase variables are transformed to
    bounded dynamics using a sinusoidal function.
    """
    def __init__(self):
        super().__init__()
        self.conditions = [[100]]

    def fetch_data(self, condition):
        # DATADIR = "/Users/william/program_repos/dygene/data/"
        DATADIR = os.path.join(file_path, "benchmark_datasets")
        amat = np.loadtxt(os.path.join(DATADIR, "kuramoto100_adjmat.csv.gz"), delimiter=',')
        X = np.loadtxt(os.path.join(DATADIR, "kuramoto100_X.csv.gz"), delimiter=',')
        return X[None, :], amat[None, :]


class McCalla(DataLoader):
    """
    A dataloader for the McCalla et al. scRNA-seq dataset

    """

    def __init__(self):
        super().__init__()

        self.ngenes = [1000, 500]
        # self.goldtypes = ["KDUnion"]
        # self.goldtypes = ["chipunion"]
        self.goldtypes = ["chipunion_KDUnion_intersect"]
        self.celltype = ["hESC", "yeastA2S", "yeastFBS", "mDC", "mESC"]
        # self.celltype = ["hESC", "mDC"]
        self.conditions = [[item] for item in list(product(self.ngenes, self.goldtypes, self.celltype))]
        print(self.conditions)

    def fetch_data(self, condition, metadata=False):

        ngenes, goldtype, celltype = condition

        if goldtype not in ["chipunion_KDUnion_intersect"]:
            raise ValueError("gold standard type must be chipunion_KDUnion_intersect")
        
        if celltype not in ["hESC", "yeastA2S", "yeastFBS", "mDC", "mESC"]:
            raise ValueError("Celltype must be one of hESC, yeastA2S, yeastFBS, mDC, mESC")
        
        # DATADIR = "/Users/william/program_repos/dygene/data/mccalla/"
        DATADIR = os.path.join(file_path, "benchmark_datasets", "mccalla")

        if celltype == "hESC":
            df = pd.read_csv(os.path.join(DATADIR, "imputed/han_GSE107552.csv.gz"), header=0).transpose()
        elif celltype == "yeastA2S":
            df = pd.read_csv(os.path.join(DATADIR, "imputed/tran_A2S.csv.gz"), header=0).transpose()
        elif celltype == "yeastFBS":
            df = pd.read_csv(os.path.join(DATADIR, "imputed/tran_FBS.csv.gz"), header=0).transpose()
        elif celltype == "mDC":
            df = pd.read_csv(os.path.join(DATADIR, "imputed/shalek_GSE48968.csv.gz"), header=0).transpose()
        elif celltype == "mESC":
            df = pd.read_csv(os.path.join(DATADIR, "normalized/mESC_zhao_GSE114952.csv.gz"), header=0).transpose()
        else:
            raise ValueError("Celltype must be one of hESC, yeastA2S, yeastFBS, mDC, mESC")
        ## set first row as column names
        df.columns = df.iloc[0]
        df = df.iloc[1:]

        ## Find the highest-variance genes
        var_genes = np.var(df.values, axis=0)
        var_genes = np.argsort(var_genes)[::-1]
        df = df.iloc[:, var_genes[:ngenes]]

        # ## select first 1000 genes
        # df = df.iloc[:, :1000]
        # ## select first 1000 timepoints
        # df = df.iloc[:1000, :]

        if "yeast" in celltype:
            celltype = "yeast"

        gold_links = np.array(
            pd.read_csv(
                os.path.join(DATADIR, f"gold_standards/{celltype}/{celltype}_{goldtype}.txt"), 
                sep="\t", 
                header=None
            )
        )
        gene_names = list(df.columns)
        amat = make_goldstandard_matrix(gene_names, gold_links, mask_tf=True, symmetric=False)

        X = df.values.copy().astype(float)

        ## Recompute pseudotime
        X, ptvals = compute_pseudotime(X)

        ## Jitter to avoid numerical issues
        X += np.random.normal(0, 1e-8, X.shape)

        X2 = np.log1p(np.abs(X)).copy()
        X2 = (X2 - np.mean(X2, axis=0)) / np.std(X2, axis=0)
        sigma_values = calculate_sigma(X2, channelwise=False)
        print(
            np.mean(np.log(1/sigma_values.squeeze())), 
            np.mean(1/sigma_values.squeeze()),
            np.median(np.log(1/sigma_values.squeeze())), 
            np.median(1/sigma_values.squeeze())
        )

        if metadata:
            return X[None, :], amat[None, :], gene_names
        

        return X[None, :], amat[None, :] 

class BEELINE(DataLoader):
    """
    A dataloader for the BEELINE dataset
    
    "mHSC-E", "mHSC-GM", "mHSC-L" denote distinct subpopulations of mouse hematopoietic 
    stem cells. "mHSC-E" is the early subpopulation, "mHSC-GM" is the granulocyte-monocyte,
    and "mHSC-L" is the late subpopulation. The other cell types are mouse embryonic stem
    cells (mESC), human embryonic stem cells (hESC), human hepatocytes (hHep), and mouse
    dendritic cells (mDC).
    """

    def __init__(self):
        super().__init__()
        # self.dataset_names = ["mESC", "hESC", "hHEP", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
        # self.ground_truths = ["mouse/mESC-ChIP-seq-network.csv", 
        #                 "human/hESC-ChIP-seq-network.csv", 
        #                 "human/HepG2-ChIP-seq-network.csv",
        #                 "mouse/mDC-ChIP-seq-network.csv",
        #                 "mouse/mHSC-ChIP-seq-network.csv",
        #                 "mouse/mHSC-ChIP-seq-network.csv",
        #                 "mouse/mHSC-ChIP-seq-network.csv"]

        # self.dataset_names = ["mESC"]
        # self.ground_truths = ["mouse/STRING-network.csv"]
        # self.conditions = [[item] for item in zip(self.dataset_names, self.ground_truths)]

        self.ngenes = [1000, 500]
        self.goldtypes = ["STRING"]
        # self.goldtypes = ["ChIP-seq"]
        self.celltype = ["mHSC-E", "mHSC-GM", "mHSC-L", "mESC", "hESC", "hHep", "mDC"]
        self.conditions = [[item] for item in list(product(self.ngenes, self.goldtypes, self.celltype))]
        print(self.conditions)

    def fetch_data(self, condition, metadata=False):

        n_genes, goldtype, celltype = condition

        if n_genes not in [500, 1000]:
            raise ValueError("n_genes must be 500 or 1000")
        
        if goldtype not in ["ChIP-seq", "STRING"]:
            raise ValueError("gold standard type must be ChIP-seq or STRING")
        
        if celltype not in ["hESC", "mESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]:
            raise ValueError("Cellt ype must be one of hESC, mESC, hHep, mDC, mHSC-E, mHSC-GM, mHSC-L")
        
        # DATADIR = "/Users/william/program_repos/dygene/data/"
        DATADIR = os.path.join(file_path, "benchmark_datasets/")

        data_dir = DATADIR + f"BEELINE/{n_genes}_{goldtype}_{celltype}/"
        fpath = os.path.join(data_dir, "data.csv")
        df = pd.read_csv(fpath, header=0)
        df.set_index(df.columns[0], inplace=True)
        gene_names = list(df.index)
        n_genes = len(gene_names)

        ## Sort the data by pseudotime
        fpath_pt = DATADIR + f"BEELINE/{celltype}_PseudoTime.csv"
        df_pt = pd.read_csv(fpath_pt, header=0)
        df_pt.set_index(df_pt.columns[0], inplace=True)
        fpath_label = os.path.join(data_dir, "label.csv")
        gold_links = np.array(pd.read_csv(fpath_label, header=0))

        amat = make_goldstandard_matrix(gene_names, gold_links, mask_tf=True, symmetric=False)

        X = df.values.copy().astype(float).T

        ## Recompute pseudotime
        X, ptvals = compute_pseudotime(X, root_index=df_pt["PseudoTime"])

        ## Use precomputed pseudotime (appears to be worse than recomputing)
        # ptvals = np.array(df_pt["PseudoTime"])
        # sort_inds = np.argsort(ptvals)
        # X, ptvals = X[sort_inds], ptvals[sort_inds]

        

        ## Jitter to avoid numerical issues
        X += np.random.normal(0, 1e-8, X.shape)
        ## Log scaling seems to make worse results
        # X = np.log1p(np.abs(X))
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        ## Remove duplicated values
        # _, idx = np.unique(df_pt["PseudoTime"], return_index=True)
        # X = X[idx]
        # ptvals = df_pt["PseudoTime"].values[idx]
        # time_linear = np.linspace(np.min(ptvals), np.max(ptvals), len(ptvals))
        # f = interp1d(np.sort(ptvals), X, axis=0, kind="cubic")
        # X = f(time_linear)

        # X2 = np.log1p(np.abs(X)).copy()
        # X2 = (X2 - np.mean(X2, axis=0)) / np.std(X2, axis=0)
        # sigma_values = calculate_sigma(X2, channelwise=False)
        # print(
        #     np.mean(np.log(1/sigma_values.squeeze())), 
        #     np.mean(1/sigma_values.squeeze()),
        #     np.median(np.log(1/sigma_values.squeeze())), 
        #     np.median(1/sigma_values.squeeze())
        # )

        if metadata:
            return X[None, :], amat[None, :], gene_names
        

        return X[None, :], amat[None, :]  
