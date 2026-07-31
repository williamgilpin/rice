import numpy as np
import warnings
import os
import tempfile

from sklearn.neighbors import NearestNeighbors
from scipy.special import betainc
from sklearn.exceptions import ConvergenceWarning

from .utils import embed_ts
from .utils import batch_pearson, batch_spearman
from .utils import progress_bar

import hnswlib

warnings.filterwarnings("ignore", message="The iteration is not making good progress")
warnings.filterwarnings("ignore", message="overflow encountered")
warnings.filterwarnings('ignore', message='Forecast type not recognized')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

relu = lambda x: np.maximum(0, x)


def neighbors_hnswlib(X, metric='euclidean', k=20):
    """
    Args:
        X (np.ndarray): dataset of shape (n, d), will be cast to float32
        metric (str): 'euclidean' or 'cosine'
        k (int): number of nearest neighbors (excludes self after query)

    Returns:
        (idx, dists): both np.ndarray with shapes (n, k+1) before postproc
    """
    n, d = X.shape
    space = 'l2' if metric == 'euclidean' else 'cosine'
    if space not in ('l2', 'cosine'):
        raise ValueError(f"Metric {metric} not supported")

    index = hnswlib.Index(space=space, dim=d)
    index.init_index(max_elements=n, M=12, ef_construction=200)
    index.add_items(X.astype(np.float32, copy=False), np.arange(n))
    index.set_ef(max(50, 4 * k))  # better speed/recall tradeoff for small k
    idx, dists = index.knn_query(X.astype(np.float32, copy=False), k+1)
    if space == 'l2':  # hnswlib returns squared L2; match previous behavior
        dists = np.sqrt(dists, out=dists)
    return idx, dists


def compute_sigmas_vectorized(dists, tol=1e-6, grid_points=48):
    """
    Vectorized replacement for per-point fsolve in find_sigma.
    Solves for sigma (per column) from sum_j exp(-max(d_j - rho, 0)/(sigma+tol)) = log2(k).

    Args:
        dists (np.ndarray): shape (k, n), sorted ascending per column (nearest on row 0)
        tol (float): numerical floor added to denominators
        grid_points (int): number of log-spaced sigma candidates

    Returns:
        sigmas (np.ndarray): shape (n,)
        weights (np.ndarray): shape (k, n), equals exp(-relu(dists - rho)/ (sigma + tol))
    """
    k, n = dists.shape
    target = np.log2(k)  # same target as original
    rhos = dists[0]      # (n,)
    A = np.maximum(dists - rhos[None, :], 0.0)  # (k, n)

    # Global, conservative sigma bracket (ensures monotonic coverage for all columns)
    amax = float(A.max())
    sigma_min = tol
    sigma_max = (amax if amax > 0.0 else 1.0) * 64.0
    sg = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), grid_points))  # (g,)

    # S(σ) = sum_j exp(-A_j/(σ+tol)), computed for all σ, all columns
    # Result shape: (n, g)
    S = np.exp(-A[..., None] / (sg[None, None, :] + tol)).sum(axis=0)

    # For each column, find first σ where S >= target (monotone in σ)
    mask = S >= target
    # Guarantee a valid bracket by forcing at least one True (σ_max -> S ≈ k >= target)
    first_true = mask.argmax(axis=1)                            # (n,)
    first_true = np.where(mask.any(axis=1), first_true, grid_points - 1)
    j_hi = np.maximum(first_true, 1)
    j_lo = j_hi - 1

    S_lo = S[np.arange(n), j_lo]
    S_hi = S[np.arange(n), j_hi]
    sig_lo = sg[j_lo]
    sig_hi = sg[j_hi]

    # Secant interpolation inside the bracket, clamped to stay positive
    sigmas = sig_lo + (target - S_lo) * (sig_hi - sig_lo) / (S_hi - S_lo + 1e-12)
    sigmas = np.maximum(sigmas, tol)

    # Compute final weights with the solved sigmas
    weights = np.exp(-A / (sigmas[None, :] + tol))
    return sigmas, weights


def simplex_neighbors(X, metric='euclidean', k=20, tol=1e-6):
    """
    Args:
        X (np.ndarray): (n, d)
        metric (str): 'euclidean' or 'cosine'
        k (int): number of neighbors used for weights
        tol (float): numerical tolerance

    Returns:
        wgts (np.ndarray): (k, n)
        idx  (np.ndarray): (k, n)
        sigmas (np.ndarray): (n,)
    """
    idx, dists = neighbors_hnswlib(X, metric, k)
    # drop self and transpose to (k, n) to match vectorized solver
    dists, idx = dists[:, 1:].T, idx[:, 1:].T
    sigmas, wgts = compute_sigmas_vectorized(dists, tol=tol)
    return wgts, idx, sigmas


def calculate_sigma(X0, d_embed=4, tol=1e-6, channelwise=True, verbose=False, cols_per_batch=1_000_000):
    """
    Streaming (low-RAM) version of `calculate_sigma`. Avoids constructing D with shape (k, m*ntx).

    Args:
        X0 (np.ndarray): (ntx, d) matrix of time series.
        d_embed (int): Embedding dimension.
        tol (float): Tolerance for the sigma solve.
        channelwise (bool): If True, embed each time series separately.
        verbose (bool): If True, prints progress every 10 channels.
        cols_per_batch (int): Number of columns of D (i.e., time points across channels)
            to solve per call to `compute_sigmas_vectorized`.

    Returns:
        np.ndarray: If channelwise, shape (m, ntx + d_embed - 1) after edge padding.
                    If False, shape (1, ntx).
    """
    X = X0.squeeze().copy()
    if channelwise:
        Xe = embed_ts(X, m=d_embed)  # (m, ntx, d_embed)
    else:
        Xe = X[None, ...]            # (1, ntx, d_embed)

    m, ntx, _ = Xe.shape
    k = min(ntx - 1, d_embed + 1)

    # Output (preallocate only the final result; never build the huge concatenation)
    all_sig = np.empty((m, ntx), dtype=np.float32)

    for i, Xe_i in enumerate(Xe):
        if verbose and (i % 10 == 0):
            print(f"Calculating sigma for channel {i} of {m}", flush=True)

        # neighbors for this channel only; drop self
        _, dists = neighbors_hnswlib(Xe_i, metric="euclidean", k=k)
        D_i = dists[:, 1:].T.astype(np.float32, copy=False)  # (k, ntx)

        # stream over time (columns) to cap peak memory
        for j0 in range(0, ntx, cols_per_batch):
            j1 = min(j0 + cols_per_batch, ntx)
            sig_block, _ = compute_sigmas_vectorized(D_i[:, j0:j1], tol=tol)
            all_sig[i, j0:j1] = sig_block

        # help GC early
        del dists, D_i

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
        min_library_size (int): Minimum library size to use for cross-mapping. Defaults to 1
        minibatch (bool): Whether to use minibatch cross-mapping. Used for large datasets. 
            Defaults to False
        minibatch_size (int): Size of minibatch to use for cross-mapping. Defaults to 1000
        store_intermediates (bool): Whether to store intermediate results
        neighbors (str): Type of neighbors to use for cross-mapping. Defaults to "simplex"
            which uses fuzzy simplicial set neighbors, which take longer but are more accurate  
            than classic 'knn' for K nearest neighbors.
        forecast (str): Type of forecast to use for cross-mapping, either "sum" or "smap".
            Defaults to "smap"
        prune_indirect (bool): Whether to prune indirect relationships due to causal
            transitivity. This helps reduce false positives. Defaults to False
        ensemble (bool): Whether to use ensemble-level cross-mapping. Defaults to False
        significance_threshold (float): Threshold for significance in cross-mapping. Defaults
            to None, in which case the causal matrix is not thresholded
        dilation_factor (float): Factor by which decimate the time series, in order to
            test for scaling of causal relationships with the number of timepoints. Defaults
            to 1.5
        sweep_d_embed (bool): Whether to sweep the embedding dimension. Defaults to False
    """
    def __init__(
            self, 
            d_embed=3, 
            k=None,
            verbose=True, 
            library_sizes=None, 
            max_library_size=None,
            min_library_size=1,
            minibatch=False,
            minibatch_size=1000,
            store_intermediates=False, 
            neighbors="simplex", 
            forecast="smap",
            prune_indirect=False,
            ensemble=True,
            signed=False,
            significance_threshold=None,
            dilation_factor=1.5,
            sweep_d_embed=False
        ):
        self.n_genes = None
        self.d_embed = d_embed
        self.causal_matrix = None
        self.all_causmat = None
        self.verbose = verbose
        self.library_sizes = library_sizes
        self.max_library_size = max_library_size
        self.min_library_size = min_library_size
        self.minibatch = minibatch
        self.minibatch_size = minibatch_size
        self.store_intermediates = store_intermediates
        self.k = k
        self.neighbors = neighbors
        self.forecast = forecast
        self.prune_indirect = prune_indirect
        self.ensemble = ensemble
        self.signed = signed
        self.significance_threshold = significance_threshold
        self.dilation_factor = dilation_factor
        self.sweep_d_embed = sweep_d_embed
        if self.k is None:
            self.k = self.d_embed + 1

        if self.forecast == "simplex" and self.neighbors == "knn":
            warnings.warn("Simplex neighbors and S-map forecast are not compatible, falling back to sum over neighbors")
            self.forecast = "sum"

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
                Ax = (Xe[:, idx.T, :1] * wgts.T[None, ..., None]).squeeze(-1).copy()  # (B, T, k)
                Ay = (Y[:, idx.T] * wgts.T[None, ..., None]).squeeze(-1).copy()  # (B, T, k)
                Cx = Xe[:, :idx.shape[1], 0].copy()  # (B, T)
                M = Ax.shape[2]
                lambda_reg = 0.5 * M
                I = np.eye(M)[None, :, :]
                AtA = np.einsum('btm,btn->bmn', Ax, Ax)
                AtC = np.einsum('btm,bt->bm', Ax, Cx)
                B_sol = np.linalg.solve(AtA + lambda_reg * I, AtC[:, :, None]).squeeze(-1)
                y_pred = np.einsum('btm,bm->bt', Ay, B_sol)
                y_target = Y[:, :y_pred.shape[1], 0].copy()
            else:
                if self.forecast != "sum":
                    warnings.warn("Forecast type not recognized, falling back to sum over neighbors")
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2).squeeze(-1)
                y_target = Y[:, :y_pred.shape[1], 0].copy()

            ## Score the prediction, weighted by the p-value
            rho, pval = batch_pearson(y_pred, y_target, pvalue=True)

            ## Set any non-significant CCM scores to zero
            if self.significance_threshold is not None:
                causal_matrix[pval > self.significance_threshold] = 0

            causal_matrix[i] = rho.copy() * (1 - pval)

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
        """
        m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
        nt = Y.shape[0]
        if len(Y.shape) < 3:
            Y = Y.T[..., None] # (n_genes, nt, 1)
        else:
            Y = Y.T

        ## If the prediction array would be larger than 500MB, use a memmap backed
        ## by a temporary file.  Unlinking immediately ensures the OS reclaims disk
        ## space when the process exits, even on crash or interrupt.
        _tmp_fd = None
        if 8 * m * m * ntx < 5e8:
            all_y_pred = np.zeros((m, m, ntx))
        else:
            _tmp_fd, _tmp_path = tempfile.mkstemp(suffix=".npy", prefix="rice_")
            os.close(_tmp_fd)
            if self.verbose: print(f"Large array detected, using temporary memmap", flush=True)
            all_y_pred = np.memmap(
                _tmp_path, dtype=np.float64, mode="w+", shape=(m, m, ntx)
            )
            os.unlink(_tmp_path)  # safe on Unix: data lives until memmap is GC'd

        k = min(ntx - 1, self.k)
        causal_matrix = np.zeros((m, m))
        I = np.eye(k)[None, :, :]
        lambda_reg = 0.5 * m * 100000

        y_target = Y[:, :ntx, 0].copy()
        for i in range(m):
            if self.neighbors == "simplex":
                wgts, idx, sig = simplex_neighbors(Xe[i], k=k, tol=tol)
            else:
                if self.neighbors != "knn":
                    warnings.warn("Neighbor type not recognized, falling back to K nearest neighbors")
                tree = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
                tree.fit(Xe[i])
                dists, idx  = tree.kneighbors(Xe[i])
                dists, idx = dists[:, 1:].T, idx[:, 1:].T # Remove self distance
                dmin = np.min(dists, axis=0) + 1e-8
                wgts = np.exp(-dists / dmin) # (k, nt) weights of k neighbors for each point

            if self.forecast == "smap":
                Ax = (Xe[:, idx.T, :1] * wgts.T[None, ..., None]).squeeze(-1)  # (B, T, k)
                Ay = (Y[:, idx.T] * wgts.T[None, ..., None]).squeeze(-1)  # (B, T, k)
                Cx = Xe[:, :idx.shape[1], 0].copy()  # (B, T)

                AtA = Ax.transpose(0, 2, 1) @ Ax
                AtC = (Ax.transpose(0, 2, 1) @ Cx[..., None]).squeeze(-1)
                B_sol = np.linalg.solve(AtA + lambda_reg * I, AtC[:, :, None]).squeeze(-1)
                y_pred = np.einsum('btm,bm->bt', Ay, B_sol)
            else:
                if self.forecast != "sum":
                    warnings.warn("Forecast type not recognized, falling back to sum over neighbors")
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2).squeeze(-1)
                y_target = Y[:, :y_pred.shape[1], 0].copy()

            all_y_pred[i] = y_pred

        # Score each response gene: fit covariance model across upstream predictions
        for i in range(m):
            # X_pred[:, t] are the cross-mapped predictions for response gene i
            # from each upstream gene, y_vec[t] is the actual value
            X_pred = all_y_pred[:, i, :]          # (m, ntx)
            y_vec = y_target[i]                    # (ntx,)

            mu_x = X_pred.mean(axis=1)             # (m,)
            mu_y = y_vec.mean()                    # scalar

            # A_j = cov(x_j, y) (unnormalized)
            A = X_pred @ y_vec - ntx * mu_x * mu_y  # (m,)

            # Variance of y
            var_y = np.dot(y_vec, y_vec) - ntx * mu_y**2

            # Predicted y_t = (x_t - mu_x)^T A for each timepoint
            X_cent = X_pred - mu_x[:, None]        # (m, ntx)
            y_hat = A @ X_cent                     # (ntx,)
            y_cent = y_vec - mu_y                  # (ntx,)

            # Pearson r between y_hat and y
            sum_pred2 = np.dot(y_hat, y_hat)
            sum_pred_y = np.dot(y_hat, y_cent)
            denom = np.sqrt(sum_pred2 * var_y)
            r = sum_pred_y / denom if denom > 0 else 0.0
            r = np.clip(r, -1.0, 1.0)

            a = ntx / 2.0 - 1.0
            pval = 2 * betainc(a, a, 0.5 * (1 - abs(r)))
            r2 = r * (1 - pval)
            if self.signed:
                causal_matrix[:, i] = A * r2
            else:
                causal_matrix[:, i] = np.abs(A) * r2

        del all_y_pred  # release memmap (if any) so OS can reclaim space

        np.fill_diagonal(causal_matrix, 0)
        return causal_matrix

    def _strided_op(self, X, op=lambda x: x):
        """
        Apply a function repeated to strided subsets of a time series
        """
        self.library_sizes = np.unique((self.dilation_factor ** np.arange(0, int(np.floor(np.log(X.shape[0]  / (self.d_embed + 1))/np.log(self.dilation_factor))))).astype(int))[::-1]
        for stride in self.library_sizes:
            yield op(X[..., ::stride])

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
                max_factor = int(np.floor(np.log(self.n  / (self.d_embed + 1))/np.log(self.dilation_factor)))
                min_factor = int(np.floor(np.log(self.min_library_size)/np.log(self.dilation_factor)))
                self.library_sizes = np.unique((
                    self.dilation_factor ** np.arange(min_factor, max_factor)
                ).astype(int))[::-1]
            else:
                self.library_sizes = np.unique(np.linspace(1, int(np.floor(self.n  / (self.d_embed + 1))), self.max_library_size).astype(int))[::-1]
        ## check that library sizes increase monotonically
        if not np.all(np.diff(self.library_sizes) <= 0):
            warnings.warn("Stride sizes must decrease monotonically. Sorting library sizes.")
            self.library_sizes = np.sort(self.library_sizes)[::-1]

        Xe = embed_ts(X, m=self.d_embed)
        Y_full = y[:-(self.d_embed - 1)]
        m = X.shape[1]
        n_strides = len(self.library_sizes)

        all_causmat = np.zeros((n_strides, m, m))

        if self.verbose:
            print(f"Fitting model with {n_strides} library sizes", flush=True)
            print(self.library_sizes, flush=True)
        for i, stride in enumerate(self.library_sizes):

            if self.verbose:
                progress_bar(i, n_strides)

            subset_inds = np.arange(0, Xe.shape[1], stride)
            if self.minibatch and len(subset_inds) > self.minibatch_size:
                subset_inds = subset_inds[:self.minibatch_size]

            if self.ensemble:
                all_causmat[i] = self.compute_crossmap_ensemble(Xe[:, subset_inds], Y_full[subset_inds])
            else:
                all_causmat[i] = self.compute_crossmap(Xe[:, subset_inds], Y_full[subset_inds])

        if self.store_intermediates:
            self.ac = all_causmat.copy()

        rho_mono = batch_spearman(all_causmat.T, pvalue=False)
        np.fill_diagonal(rho_mono, 0)
        cause_matrix = all_causmat[-1] * np.abs(rho_mono)

        ## Prune indirect connections
        if self.prune_indirect:
            cause_matrix = filter_loops(cause_matrix)

        return cause_matrix
