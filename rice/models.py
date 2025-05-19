import numpy as np
import warnings
import os

# from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr

from .utils import embed_ts, multivariate_embed_ts
from .utils import batch_pearson, batch_spearman, flatten_along_axis
from .utils import progress_bar, debug_print
from .utils import hollow_matrix, banded_matrix, max_linear_correlation_ridge

## Disable optimization warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", message="The iteration is not making good progress")
warnings.filterwarnings("ignore", message="overflow encountered")
warnings.filterwarnings('ignore', message='Forecast type not recognized')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

relu = lambda x: np.maximum(0, x)
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
    # dists = -np.log(wgts + tol)
    # return dists, idx
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
    # func = lambda isig: sum(np.exp(-relu(dists - rho) * isig)) - np.log2(k)
    # jac = lambda isig: -sum(np.exp(-relu(dists - rho) * isig) * relu(dists - rho))
    # isigma = fsolve(func, 1/rho, fprime=jac, xtol=tol)[0]
    # sigma = 1 / isigma
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


def data_processing_inequality(M, i, j, k):
    """
    Filter out edges resulting from indirect relationships. If the matrix M represents
    mutual information, then this function filters out edges where X -> Y -> Z induces
    a non-zero mutual information between X and Z.

    The criterion is that if I[i, k] < min(I[i, j], I[j, k]), then the edge from i to k 
    is filtered.

    Args:
        M (np.ndarray): Mutual information matrix
        i (int): Source node
        j (int): Intermediate node
        k (int): Target node

    Returns:
        tuple: Source and target nodes
    """
    m_ij, m_ik, m_jk = M[i, j], M[i, k], M[j, k]
    lowest = m_ij
    edge = (i, j)
    if m_ik < lowest:
        lowest = m_ik
        edge = (i, k)
    if m_jk < lowest:
        edge = (j, k)
    
    return edge

def filter_loops(M0, max_neighbors=100):
    """
    Filter out loops from the mutual information matrix based on the Data Processing 
    Inequality (DPI).

    Args:
        M0 (np.ndarray): Mutual information matrix
        max_neighbors (int): Maximum number of neighbors to consider for each node

    Returns:
        np.ndarray: Mutual information matrix with loops removed
    """
    M = M0.copy()
    n = M.shape[0]

    # Sparsify large matrices
    if n > max_neighbors:
        n_links_per_node = max_neighbors / n
        threshold = np.percentile(M.ravel(), 100 * (1 - n_links_per_node))
        M[np.abs(M) < threshold] = 0
    
    # Build adjacency list (neighbors[i] = indices j where M[i,j] != 0)
    neighbors = []
    for i in range(n):
        nbrs = set(np.where(M[i] != 0)[0])
        neighbors.append(nbrs)
    
    set_to_zero = []
    for i in range(n):
        for j in neighbors[i]:
            # Intersect neighbors of i and j
            common_k = neighbors[i].intersection(neighbors[j])
            # Optionally discard i and j from the intersection:
            common_k.discard(i); common_k.discard(j);
            
            for k in common_k:
                # Decide whether to zero out the edge i -> k
                res = data_processing_inequality(M, i, j, k)
                if res is not None:
                    set_to_zero.append(res)
    
    if set_to_zero:
        # Convert list of tuples into array so we can do M[set_to_zero] = 0
        set_to_zero = np.array(set_to_zero).T
        M[tuple(set_to_zero)] = 0

    return M


from scipy.optimize import fsolve, minimize_scalar
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, MultiTaskElasticNetCV

class CausalDetection:
    """
    Find the causal relationships among sets of univariate time series.
    The i,j th element of the causal matrix denotes the degree to which i is caused by j
    Equivalently, it measures how much the dynamics x_i is coupled to x_j via the
    matrix sum_j A_{ij} x_j.

    Attributes:
        d_embed (int): Number of dimensions to embed the time series into
        k (int): Number of neighbors to consider for cross-mapping
        verbose (bool): Whether to show progress bar
        library_sizes (np.ndarray): Array of library sizes to use for cross-mapping. If
            None, use all library sizes
        max_library_size (int): Maximum library size to use for cross-mapping. Defaults 
            to None, in which case the number of library sizes equals the number of 
            timepoints
        store_intermediates (bool): Whether to store intermediate results
        neighbors (str): Type of neighbors to use for cross-mapping. Defaults to classic
            'knn' for K nearest neighbors, while 'simplex' uses fuzzy simplicial set
            neighbors, which take longer but are more accurate  
        forecast (str): Type of forecast to use for cross-mapping, either "sum" or "smap".
            Defaults to "sum"
        prune_indirect (bool): Whether to prune indirect relationships due to causal
            transitivity. This helps reduce false positives. Defaults to False
        ensemble (bool): Whether to use ensemble-level cross-mapping. Defaults to False
        significance_threshold (float): Threshold for significance in cross-mapping. Defaults
            to None, in which case the causal matrix is not thresholded
        sweep_d_embed (bool): Whether to sweep the embedding dimension. Defaults to False
    """
    def __init__(
            self, 
            d_embed=10, 
            k=None,
            verbose=True, 
            library_sizes=None, 
            max_library_size=None,
            store_intermediates=False, 
            neighbors="knn", 
            forecast="sum",
            return_features=False,
            prune_indirect=False,
            ensemble=False,
            significance_threshold=None,
            sweep_d_embed=False
        ):
        self.n_genes = None
        self.d_embed = d_embed
        self.causal_matrix = None
        self.all_causmat = None
        self.verbose = verbose
        self.library_sizes = library_sizes
        self.max_library_size = max_library_size
        self.store_intermediates = store_intermediates
        self.k = k
        self.neighbors = neighbors
        self.forecast = forecast
        self.return_features = return_features
        self.prune_indirect = prune_indirect
        self.ensemble = ensemble
        self.significance_threshold = significance_threshold
        self.sweep_d_embed = sweep_d_embed
        if self.k is None:
            self.k = self.d_embed + 1

        if self.store_intermediates:
            self.y_pred = list()

        if self.return_features:
            self.features = dict()

    def compute_crossmap(self, Xe, Y, X=None, stride=-1, tpred=0, tol=1e-10):
        """
        Use cross-mapping to to predict Y from Xe

        Args:
            Xe (np.ndarray): A matrix of shape (n_genes, nt, d_embed)
            Y (np.ndarray): A matrix of shape (n_genes, nt, 1)
            stride (int): Stride to use for cross-mapping. Defaults to -1, in which case
                the entire time series is used
            tpred (int): Timepoint to predict. Defaults to 0, in which case the last timepoint
                is predicted

        Can modify to hold out test
        """
        m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
        nt = Y.shape[0]
        if len(Y.shape) < 3:
            Y = Y.T[..., None] # (n_genes, nt, 1)
        else:
            Y = Y.T

        causal_matrix = np.zeros((m, m))

        ## Outer index runs over causes, which we use for lookups into the downstream
        ## causees. 
        for i in range(m):
            if self.neighbors == "simplex":
                wgts, idx, sig = simplex_neighbors(Xe[i], k=min(ntx - 1, self.k), tol=tol)
            else:
                if self.neighbors != "knn":
                    warnings.warn("Neighbor type not recognized, falling back to K nearest neighbors")
                tree = NearestNeighbors(n_neighbors=min(ntx, self.k+1), algorithm='auto', metric='euclidean')
                tree.fit(Xe[i])
                dists, idx  = tree.kneighbors(Xe[i])
                dists, idx = dists[:, 1:].T, idx[:, 1:].T # Remove self distance
                dmin = np.min(dists, axis=0) + 1e-8
                wgts = np.exp(-dists / dmin) # (k, nt) weights of k neighbors for each point

            if self.forecast == "smap":
                # print(wgts.shape, idx.shape)
                Ax = (Xe[:, idx.T, :1] * wgts.T[None, ..., None]).squeeze().copy()  # Input batch matrix (B x T x M)
                Ay = (Y[:, idx.T] * wgts.T[None, ..., None]).squeeze().copy()  # Input batch matrix (B x T x M)
                Cx = Xe[:, :idx.shape[1], 0].squeeze().copy()  # Target batch matrix (B x T)
                Cy = Y[:, :idx.shape[1], 0].squeeze().copy() # Not used. Target batch matrix (B x T)
                M = Ax.shape[2] # Number of features
                lambda_reg = 0.5 * M # Scale regularization parameter to the number of features
                I = np.eye(M)[None, :, :]  # Identity matrix broadcasted across batches (B x M x M)
                AtA = np.einsum('btm,btn->bmn', Ax, Ax)  ## Compute A^T @ A over batch (B x M x M) --> (B x M x M)
                AtC = np.einsum('btm,bt->bm', Ax, Cx)    ## Compute A^T @ C over batch (B x M) --> (B x M)
                B_sol = np.linalg.solve(AtA + lambda_reg * I, AtC[:, :, None])  # Solve (B x M x 1)
                B_sol = B_sol.squeeze(-1)  # Final shape (B x M)
                # B_sol = np.ones_like(B_sol)
                y_pred = np.einsum('btm,bm->bt', Ay, B_sol) ## Predict Y with the B coeffs fit from X
                y_target = Y[:, :y_pred.shape[1], 0].copy()
            else:
                if self.forecast != "sum":
                    warnings.warn("Forecast type not recognized, falling back to sum over neighbors")
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
                y_pred, y_target = y_pred[:, :, 0].copy(), Y[:, :y_pred.shape[1], 0].copy()

                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
                y_target = Y[:, :y_pred.shape[1], :].copy()
                y_pred, y_target = np.squeeze(y_pred), np.squeeze(y_target)

            if self.store_intermediates:
                self.y_pred.append(y_pred.copy())

            ## Score the prediction, weighted by the p-value
            rho, pval = batch_pearson(y_pred, y_target, pvalue=True)

            ## Set any non-significant CCM scores to zero
            if self.significance_threshold is not None:
                causal_matrix[pval > self.significance_threshold] = 0

            causal_matrix[i] = rho.copy() * (1 - pval)

        ## Fully vectorized triggers memory error
        # idx = np.argsort(dX, axis=1)[:, 1:self.k+1]
        # dists = np.sort(dX, axis=1)[:, 1:self.k+1]
        # wgts = np.exp(-dists / np.min(dists, axis=1, keepdims=True))
        # Xe_sel = np.moveaxis(Xe[:, idx.T], (0,1,2,3,4), (1,2,3,0,4))
        # y_pred = np.sum(Xe_sel * np.swapaxes(np.swapaxes(wgts, 1, 2)[None, ..., None], 0, 1), axis=3)
        # y_true = Xe.copy()
        # ytc = y_true - np.mean(y_true, axis=-1, keepdims=True)
        # ypc = y_pred - np.mean(y_pred, axis=-1, keepdims=True)
        # rho = np.sum(ytc * ypc, axis=-1) / np.sqrt(np.sum(ytc ** 2, axis=-1) * np.sum(ypc ** 2, axis=-1))
        # causal_matrix = np.mean(rho.copy(), axis=-1)

        np.fill_diagonal(causal_matrix, 0)
        return causal_matrix

    def compute_crossmap_ensemble(self, Xe, Y, batch_indices=None, stride=-1, tpred=0, tol=1e-10):
        """
        Use cross-mapping to to predict Y from Xe

        Args:
            Xe (np.ndarray): A matrix of shape (n_genes, nt, d_embed)
            Y (np.ndarray): A matrix of shape (n_genes, nt, 1)
            batch_indices (np.ndarray): A list of indices to use for batched cross-mapping.
                Defaults to None, in which case all indices are used
            stride (int): Stride to use for cross-mapping. Defaults to -1, in which case
                the entire time series is used
            tpred (int): Timepoint to predict. Defaults to 0, in which case the last timepoint
                is predicted

        Can modify to hold out test
        """
        m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
        nt = Y.shape[0]
        if len(Y.shape) < 3:
            Y = Y.T[..., None] # (n_genes, nt, 1)
        else:
            Y = Y.T
        
        all_y_pred = list()
        all_y_true = list()

        debug_print(0)
        causal_matrix = np.zeros((m, m))

        ## Outer index runs over causes, which we use for lookups into the downstream
        ## causees. 
        for i in range(m):

            if self.neighbors == "simplex":
                wgts, idx, sig = simplex_neighbors(Xe[i], k=min(ntx - 1, self.k), tol=tol)
            else:
                if self.neighbors != "knn":
                    warnings.warn("Neighbor type not recognized, falling back to K nearest neighbors")
                tree = NearestNeighbors(n_neighbors=min(ntx, self.k+1), algorithm='auto', metric='euclidean')
                tree.fit(Xe[i])
                dists, idx  = tree.kneighbors(Xe[i])
                dists, idx = dists[:, 1:].T, idx[:, 1:].T # Remove self distance
                dmin = np.min(dists, axis=0)
                wgts = np.exp(-dists / dmin) # (k, nt) weights of k neighbors for each point

            ## Regularized smap often seems to overfit
            if self.forecast == "smap":
                # print(wgts.shape, idx.shape)
                Ax = (Xe[:, idx.T, :1] * wgts.T[None, ..., None]).squeeze().copy()  # Input batch matrix (B x T x M)
                Ay = (Y[:, idx.T] * wgts.T[None, ..., None]).squeeze().copy()  # Input batch matrix (B x T x M)
                Cx = Xe[:, :idx.shape[1], 0].squeeze().copy()  # Target batch matrix (B x T)
                Cy = Y[:, :idx.shape[1], 0].squeeze().copy() # Not used. Target batch matrix (B x T)

                M = Ax.shape[2] # Number of features
                lambda_reg = 0.5 * M * 100000 # Scale regularization parameter to the number of features
                I = np.eye(M)[None, :, :]  # Identity matrix broadcast across batches (B x M x M)
                AtA = np.einsum('btm,btn->bmn', Ax, Ax)  ## Compute A^T @ A over batch (B x M x M) --> (B x M x M)
                AtC = np.einsum('btm,bt->bm', Ax, Cx)    ## Compute A^T @ C over batch (B x M) --> (B x M)
                B_sol = np.linalg.solve(AtA + lambda_reg * I, AtC[:, :, None]).squeeze(-1)
                # inverse = np.linalg.inv(AtA + lambda_reg * I)
                # B_sol = np.einsum('bmn,bm->bn', inverse, AtC)[:, :, None]
                # B_sol = B_sol.squeeze(-1)  # Final shape (B x M)

                y_pred = np.einsum('btm,bm->bt', Ay, B_sol) ## Predict Y with the B coeffs fit from X
                y_target = Y[:, :y_pred.shape[1], 0].copy()
                ## Include all context points
                # y_pred = flatten_along_axis((Ay * B_sol[:, None]).squeeze(), (0, 2)).T
            else:
                if self.forecast != "sum":
                    warnings.warn("Forecast type not recognized, falling back to sum over neighbors")
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
                y_pred, y_target = y_pred[:, :, 0].copy(), Y[:, :y_pred.shape[1], 0].copy()
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
                y_target = Y[:, :y_pred.shape[1], :].copy()
                y_pred, y_target = np.squeeze(y_pred), np.squeeze(y_target)
                ## Include all context points
                # y_pred = flatten_along_axis((Y[:, idx.T] * wgts.T[None, ..., None]).squeeze(), (0, 2)).T

            ## Classical CCM
            # rho, pval = batch_pearson(y_pred, y_target, pvalue=True)
            # if self.significance_threshold is not None:
            #     causal_matrix[pval > self.significance_threshold] = 0
            # causal_matrix[i] = rho.copy() * (1 - pval)

            all_y_pred.append(y_pred.copy())
            all_y_true.append(y_target.copy())
            # continue
            # print("t", y_pred.shape, y_target.shape)

            ## Should match default behavior
            # rho, pval = batch_pearson(y_pred, y_target, pvalue=True)
            # causal_matrix[i] = rho.copy() * (1 - pval)
            # continue

            # ## Xd (y_pred) is the cause, Yd (y_target) is the effect.
            # ## Index is over timepoints x N_responses
            # ## Really, we want to have N_upstream x T and N_responses x T
            # Xd, Yd = y_pred.T, y_target.T 
            # if len(Xd.shape) == 1:
            #     Xd, Yd = Xd[:, None], Yd[:, None]
            # Xd = (Xd - np.mean(Xd, axis=0, keepdims=True))
            # Yd = (Yd - np.mean(Yd, axis=0, keepdims=True))
            # rho, pval = max_linear_correlation_ridge(Xd, Yd, return_pvalue=True)
            # causal_matrix[i] = rho.copy() * (1 - pval)
            # continue

            # Xd, Yd = y_pred.T, y_target.T 
            # Xd, Yd = Yd, Xd
            # if len(Xd.shape) == 1:
            #     Xd, Yd = Xd[:, None], Yd[:, None]
            # # print(Xd.shape, Yd.shape)
            # Xd = (Xd - np.mean(Xd, axis=0, keepdims=True))
            # Yd = (Yd - np.mean(Yd, axis=0, keepdims=True))
            # # print(np.mean(np.std(Xd, axis=0, keepdims=True)), np.mean(np.std(Yd, axis=0, keepdims=True)))
            # # print(np.mean(Xd, axis=0, keepdims=True).shape)
            # # lasso = MultiTaskLassoCV(
            # #     fit_intercept=False, 
            # #     cv=min(5, ntx), 
            # #     alphas=np.logspace(-5, 3, 5)
            # # )
            # # lasso.fit(Xd, Yd)
            # # # A = lasso.coef_.T # check this transpose
            # # A = lasso.coef_

            # # max_r2 = np.zeros(Xd.shape[1])
            # # for lambda_val in np.logspace(-5, 3, 5):
            # #     A = np.linalg.inv(Xd.T @ Xd + lambda_val * np.eye(Xd.shape[1])) @ Xd.T @ Yd
            # #     Yd_pred = Xd @ A
            # #     residuals = Yd - Yd_pred
            # #     ## compute r2 along the first dimension, contracting from (5, 100) to (100,)
            # #     r2 = 1 - np.sum(residuals ** 2, axis=0) / np.sum(Yd ** 2, axis=0)
            # #     max_r2 = r2
            # # causal_matrix[i] = max_r2


            # lambda_ = 1 * m * 1000
            # # A = np.linalg.inv(Xd.T @ Xd + lambda_ * np.eye(Xd.shape[1])) @ Xd.T @ Yd
            # A = Xd.T @ Yd # overregularized limit
            # Yd_pred = Xd @ A
            # residuals = Yd - Yd_pred
            # r2 = 1 - np.sum(residuals ** 2, axis=0) / np.sum(Yd ** 2, axis=0)
            # causal_matrix[i] = r2
            # # causal_matrix[i] = np.mean(np.abs(A.T), axis=0)

        all_y_pred = np.array(all_y_pred)
        all_y_true = np.array(all_y_true)[0]

        # from .hccm import split_indices
        # batch_indices = split_indices(m, 20)
        # # print("batch_indices", batch_indices)
        # if batch_indices is not None:
        #     nb = len(batch_indices)
        #     for index_set in batch_indices:
        #         # print("index_set", index_set)
        #         for i in range(m):
        #             # Loop over responses
        #             Xd, Yd = all_y_pred[index_set, i].T, all_y_true[i][:, None] # sweep downstreams
        #             Xd = (Xd - np.mean(Xd, axis=0, keepdims=True))
        #             Yd = (Yd - np.mean(Yd, axis=0, keepdims=True))
        #             A = (Xd.T @ Yd).T
        #             Yd_pred = Xd @ A.T
        #             from scipy.stats import pearsonr
        #             corr = pearsonr(Yd_pred.squeeze(), Yd.squeeze())
        #             corr = corr[0] * (1 - corr[1])
        #             r2 = corr
        #             A = np.abs(A)
        #             A *= r2
        #             A = A.squeeze()

        #             causal_matrix[index_set, i] = A.squeeze()
        #     np.fill_diagonal(causal_matrix, 0)
        #     return causal_matrix     
        # print("hit", flush=True)     

        # For each response, fit a ridge regression model over timepoints
        # To assign a causal score to each upstream gene
        for i in range(m):
            
            # Loop over responses
            Xd, Yd = all_y_pred[:, i].T, all_y_true[i][:, None] # sweep downstreams
            # Xd, Yd = all_y_pred[i].T, all_y_true[i][:, None] # sweep upstreams
           
            Xd = (Xd - np.mean(Xd, axis=0, keepdims=True))
            Yd = (Yd - np.mean(Yd, axis=0, keepdims=True))

            # lambda_val = 1e0 / Xd.shape[1] * 1e10
            # ridge = Ridge(alpha=lambda_val, fit_intercept=False)
            # ridge.fit(Xd, Yd)
            # Yd_pred = ridge.predict(Xd) # error doesn't distinguish upstream
            # from scipy.stats import pearsonr
            # corr = pearsonr(Yd_pred.squeeze(), Yd.squeeze())
            # corr = corr[0] * (1 - corr[1])
            # r2 = corr
            # A = ridge.coef_.T.squeeze()
            # A = np.abs(A)
            # A *= r2
            
            ## Strong regularization limit
            A = (Xd.T @ Yd).T
            Yd_pred = Xd @ A.T
            from scipy.stats import pearsonr
            corr = pearsonr(Yd_pred.squeeze(), Yd.squeeze())
            corr = corr[0] * (1 - corr[1])
            # corr = np.nan_to_num(corr, nan=1.0) # constant time series no variance
            r2 = corr
            A = np.abs(A)
            A *= r2

            causal_matrix[:, i] = A.squeeze()

        np.fill_diagonal(causal_matrix, 0)
        return causal_matrix


    def fit(self, X, y=None):
        """
        Fit the model to the data

        Args:
            X (np.ndarray): Upstream data matrix of shape (n_timepoints, n_features)
            y (np.ndarray): Downstream data matrix of shape (n_timepoints, n_features)
                Defaults to None, in which case X is used as the target

        Returns:
            np.ndarray: Stack of causal matrices

        """
        if self.sweep_d_embed:
            self.d_embed = 2
            self.sweep_d_embed = False
            cmat = self.fit(X, y)
            self.sweep_d_embed = True 
            for d_embed in np.arange(3, 12):
                if self.verbose:
                    print(f"Fitting model with d_embed: {d_embed}")
                self.d_embed = d_embed
                self.sweep_d_embed = False
                cmat = np.nanmax([cmat, self.fit(X, y)], axis=0)
                self.sweep_d_embed = True
            return cmat

        if y is None:
            y = X

        self.n = X.shape[0]

        if self.library_sizes is None:
            if self.max_library_size is None:
                self.library_sizes = np.arange(1, int(np.floor(self.n  / (self.d_embed + 1))))[::-1]
                # self.library_sizes = (2 ** np.arange(0, int(np.floor(np.log2(self.n  / (self.d_embed + 1)))))).astype(int)[::-1]
            else:
                self.library_sizes = np.unique(np.linspace(1, int(np.floor(self.n  / (self.d_embed + 1))), self.max_library_size).astype(int))[::-1]

        all_causmat = np.zeros((len(self.library_sizes), X.shape[1], X.shape[1]))

        ## Iterate over library sizes to test robustness of causal matrix
        for i, stride in enumerate(self.library_sizes):
            
            if self.verbose:
                progress_bar(i, len(self.library_sizes))

            if self.ensemble:
                Xe = embed_ts(X, m=self.d_embed)
                # all_causmat.append(self.compute_crossmap_ensemble(Xe[:, ::stride], y[:-(self.d_embed - 1)][::stride]))
                all_causmat[i] = self.compute_crossmap_ensemble(Xe[:, ::stride], y[:-(self.d_embed - 1)][::stride])

            else:
                # Xe = embed_ts_pca(X, m=self.d_embed, scaled=True)
                Xe = embed_ts(X, m=self.d_embed)
                # all_causmat.append(self.compute_crossmap(Xe[:, ::stride], y[:-(self.d_embed - 1)][::stride]))
                all_causmat[i] = self.compute_crossmap(Xe[:, ::stride], y[:-(self.d_embed - 1)][::stride])

        # all_causmat = np.array(all_causmat)

        if self.store_intermediates:
            self.ac = all_causmat.copy()
        
        ## Kernel still dies
        ## Downweight non-monotonic scaling with data size
        traces = np.reshape(all_causmat, (all_causmat.shape[0], -1)).T
        rho_mono, pval_mono = batch_spearman(traces, pvalue=True)
        rho_mono = np.reshape(rho_mono, all_causmat.shape[1:])
        pval_mono = np.reshape(pval_mono, all_causmat.shape[1:])
        cause_matrix = all_causmat[-1] * rho_mono

        ## Kernel still dies
        # rho_mono = np.zeros((X.shape[1], X.shape[1]))
        # for inds in np.ndindex(X.shape[1], X.shape[1]):
        #     trace = all_causmat[:, *inds]
        #     baseline = np.argsort(np.argsort(trace, axis=-1), axis=-1).astype(np.float32)
        #     corr = spearmanr(trace, baseline)
        #     rho_mono[*inds] = corr[0]
        # cause_matrix = all_causmat[-1] * rho_mono

        ## Prune indirect connections
        if self.prune_indirect:
            cause_matrix = filter_loops(cause_matrix)

        return cause_matrix




