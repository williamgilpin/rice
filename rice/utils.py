"""
Utilities for processing and transforming time series datasets, with a particular
focus on vectorizing over batches of time series.

"""
import warnings
import numpy as np
from scipy.linalg import hankel

# try:
#     from numba import njit, prange
#     numba_flag = True
# except:
#     numba_flag = False
#     warnings.warn("Numba is not installed, some functions will be slower.")

# if not numba_flag:
#     njit = lambda x: x
#     prange = lambda x: range(x)

def mask_topk(arr, k=1):
    """
    Given an array, mask all but the top k elements
    
    Args:
        arr (np.ndarray): Array to mask
        k (int): Number of elements to keep unmasked

    Returns:
        np.ndarray: Masked array
    """
    mask = np.zeros_like(arr).ravel()
    idx = np.argsort(arr.ravel())[ -k:]
    mask[idx] = 1
    mask = mask.reshape(arr.shape)
    return mask

# def hankel_matrix(data, q, p=None):
#     """
#     Find the Hankel matrix dimensionwise for multiple multidimensional 
#     time series
    
#     Args:
#         data (ndarray): An array of shape (N, T, 1) or (N, T, D) corresponding to a 
#             collection of N time series of length T and dimensionality D
#         q (int): The width of the matrix (the number of features)
#         p (int): The height of the matrix (the number of samples)
        
#     Returns:
#         hmat (ndarray): An array of shape (N, T - p, D, q) containing the Hankel
#             matrices for each time series

#     """
    
#     if len(data.shape) == 3:
#         return np.stack([_hankel_matrix(item, q, p) for item in data])
    
#     if len(data.shape) == 1:
#         data = data[:, None]
#     hmat = _hankel_matrix(data, q, p)    
#     return hmat
    

# def _hankel_matrix(data, q, p=None):
#     """
#     Calculate the hankel matrix of a multivariate timeseries
    
#     Args:
#         data (ndarray): T x D multidimensional time series
#         q (int): The number of columns in the Hankel matrix
#         p (int): The number of rows in the Hankel matrix

#     Returns:
#         ndarray: The Hankel matrix of shape (T - p, D, q)
#     """
#     if len(data.shape) == 1:
#         data = data[:, None]

#     # Hankel parameters
#     if not p:
#         p = len(data) - q
#     all_hmats = list()
#     for row in data.T:
#         first, last = row[-(p + q) : -p], row[-p - 1 :]
#         out = hankel(first, last)
#         all_hmats.append(out)
#     out = np.dstack(all_hmats)
#     return np.transpose(out, (1, 0, 2))[:-1]

def batch_diag(array, axis=0):
    """Give a multidimensional array, create a diagonal matrix for each row along the 
    specified axis.

    Args:
        array (np.ndarray): The input array
        axis (int): The axis along which to create the diagonal matrices

    Returns:
        np.ndarray: The output array

    Example:
        >>> array = np.random.randn(3, 4, 5)
        >>> a = batch_diag(array, axis=0)
        >>> a.shape
        (3, 3, 4, 5)
    """
    input_shape = array.shape
    if axis < 0:
        axis = len(input_shape) + axis
    n = input_shape[axis]
    remaining_dims = np.delete(np.arange(len(input_shape)), axis)
    remaining_dims[remaining_dims > axis] += 1
    id = np.expand_dims(np.eye(n), tuple(remaining_dims))
    array = np.expand_dims(array, axis)
    out = array * id
    return out

def embed_ts(X, m):
    """
    Create a time delay embedding of a time series or a set of time series

    Args:
        X (array-like): A matrix of shape (n_timepoints, n_dims) or 
            of shape (n_timepoints)
        m (int): The number of dimensions

    Returns:
        Xp (array-like): A time-delay embedding
    """
    batch_shape = len(X.shape)
    if batch_shape == 1:
        X_out = np.lib.stride_tricks.sliding_window_view(X, m, axis=0).squeeze()
    if batch_shape == 2:
        X_out = np.lib.stride_tricks.sliding_window_view(X, m, axis=0).squeeze()
        X_out = np.swapaxes(X_out, 0, 1)
    if batch_shape == 3:
        X_out = np.lib.stride_tricks.sliding_window_view(X, m, axis=1).squeeze()
        X_out = np.swapaxes(X_out, 2, 3)
    return X_out

# def embed_ts(X, m, padding="constant"):
#     """
#     Create a time delay embedding of a time series or a set of time series

#     Args:
#         X (array-like): A matrix of shape (n_timepoints, n_dims) or 
#             of shape (n_timepoints)
#         m (int): The number of dimensions

#     Returns:
#         Xp (array-like): A time-delay embedding
#     """
#     X = X.copy()[::-1]
#     if padding:
#         if len(X.shape) == 1:
#             X = np.pad(X, [m, m], padding)
#         if len(X.shape) == 2:
#             X = np.pad(X, [[m, m], [0, 0]], padding)
#         if len(X.shape) == 3:
#             X = np.pad(X, [[0, 0], [m, m], [0, 0]], padding)
#     Xp = hankel_matrix(X, m)
#     Xp = np.moveaxis(Xp, (0, 1, 2), (1, 2, 0))
#     Xp = Xp[:, ::-1, ::-1]
#     Xp = Xp[:, m-1:-m]
#     return Xp


def multivariate_embed_ts(X, m, **kwargs):
    """
    Create a flattened time delay embedding of a multivariate time series

    Args:
        X (array-like): A matrix of shape (n_timepoints, n_dims)
        m (int): The number of dimensions

    Returns:
        Xp (array-like): A time-delay embedding of shape (n_timepoints, n_dims * m)
    """
    Xe = embed_ts(X, m=m, **kwargs)
    Xe = np.moveaxis(Xe, 1, 0)
    Xe_flat = np.reshape(Xe, (Xe.shape[0], -1))
    # Xe_flat = np.transpose(Xe, (1, 0, 2)).reshape(Xe.shape[1], -1, order="F")
    return Xe_flat


from scipy.sparse.linalg import svds
def project_pca(X, k=1):
    """
    Perform PCA on the dataset X and return the projection onto the top k principal components.

    Args:
        X (numpy.ndarray): The input dataset of shape (N, M).
        k (int): The number of principal components to project onto.

    Returns:
        numpy.ndarray: The dataset projected onto the top k principal components of 
            shape (N, k).
    """
    if X.shape[1] <= k:
        return X
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = svds(X_centered, k=k)
    
    # Sort the singular values and corresponding vectors in descending order
    idx = np.argsort(-S)
    U = U[:, idx]
    S = S[idx]
    Vt = Vt[idx, :]
    
    # Project the data onto the top k principal components
    X_projected = np.dot(U, np.diag(S))
    
    return X_projected

def batch_pca(X0):
    """
    Given a tensor of shape batch, N, M, perform PCA on the last two dimensions
    
    Args:
        X0 (ndarray): A tensor of shape (batch, N, M)

    Returns:
        eivals (ndarray): Eigenvalues of the covariance matrix, sorted in descending 
            order. Shape is (batch, M)
        eivecs (ndarray): Eigenvectors of the covariance matrix, sorted in descending
            order along last index. Shape is (batch, M, M), where the last index is the
            eigenvector index that pairs with the corresponding eigenvalue.
    
    """
    X = X0.copy()
    X -= np.mean(X, axis=-1, keepdims=True)
    cov = np.einsum('ijk,ijm->ikm', X, X) / (X.shape[1] - 1)

    eigsys = np.linalg.eigh(cov)
    eivals = eigsys[0][:, ::-1]
    eivecs = eigsys[1][..., ::-1]

    return eivals, eivecs



def embed_ts_pca(X, m=10, scaled=False):
    """
    Embed a univariate time series data using PCA

    Args:
        X (np.ndarray): Time series data
        m (int): Embedding dimension
        scaled (bool): Whether to scale transformed data features by the eigenvalues

    Returns:
        np.ndarray: Embedded time series data
    """
    Xe = embed_ts(X, m=m)
    eigvals, pca_vecs = batch_pca(Xe)
    Xe_proj = np.einsum('itm,imk->itk', Xe, pca_vecs)
    if scaled:
        eigvals = np.real(eigvals)
        eigvals[eigvals < 0] = 0
        scale_factors = batch_diag(np.sqrt(eigvals), axis=-1)
        Xe_proj = Xe_proj @ scale_factors
    return Xe_proj

# from scipy.linalg import eigh
from scipy.signal import savgol_filter
def batch_sfa(X, num_features, return_transform=False, savgol_settings=None):
    """
    Perform Slow Feature Analysis (SFA) on the data matrix X.
    
    Args:
        X (numpy.ndarray): The data matrix of shape (B, N, M).
        num_features (int): The number of slow features to extract.
        savgol_settings (dict): The settings for the Savitzky-Golay filter. These are 
            the window and polynomial order typically passed to 
            scipy.signal.savgol_filter. If this argument is None, then the 
            finite-difference time-derivatives are used instead of the Savitzky-Golay
            filter.
    
    Returns:
        S (numpy.ndarray): The slow features of shape (B, N, num_features).
        W (numpy.ndarray): The transformation matrix of shape (B, M, num_features).
        E (numpy.ndarray): The eigenvalues of the generalized eigenvalue problem of
            shape (B, num_features).
    """
    B, N, M = X.shape
    
    # Compute the finite-difference time-derivatives or use the Savitzky-Golay filter
    if savgol_settings is not None:
        dX = savgol_filter(X, **savgol_settings, axis=1, deriv=1)
    else:
        dX = np.diff(X, axis=1)
        dX = np.pad(dX, ((0, 0), (1, 0), (0, 0)), mode='constant')

    # Compute the covariance matrix of the original data and the time-derivatives
    C_X = np.einsum('bij,bik->bjk', X, X) / N
    C_dX = np.einsum('bij,bik->bjk', dX, dX) / (N - 1)

    ## Solve the generalized eigenvalue problem, and sort the eigenvectors by eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(C_X) @ C_dX)
    # print(eigenvalues.shape, eigenvectors.shape)
    idx = np.argsort(eigenvalues, axis=1)
    # W = np.array([eigenvectors[i, :, idx[i, :num_features]] for i in range(B)])
    batch_indices = np.arange(eigenvectors.shape[0])[:, None, None]
    vector_component_indices = np.arange(eigenvectors.shape[1])[None, :, None]
    W = eigenvectors[batch_indices,  vector_component_indices, idx[:, None, :num_features]] # (xx, xxx, )
    E = eigenvalues[batch_indices, idx[:, :num_features]]
    # print(W.shape, eigenvectors.shape, idx[:, None, :num_features].shape)


    # Project the original data onto the slow features
    # print(X.shape, W.shape)
    S = np.einsum('bij,bkj->bik', X, W)
    if return_transform:
        return S, W, E
    else:
        return S
    
def embed_ts_sfa(X, m=10, scaled=False):
    """
    Embed a univariate time series data using PCA

    Args:
        X (np.ndarray): Time series data
        m (int): Embedding dimension
        scaled (bool): Whether to scale transformed data features by the eigenvalues

    Returns:
        np.ndarray: Embedded time series data
    """
    Xe = embed_ts(X, m=m)
    # print(Xe.shape)
    Xe_sfa, W, E = batch_sfa(Xe, m, return_transform=True)
    if scaled:
        eigvals = np.real(E)
        # eigvals[eigvals < 0] = 0
        # scale_factors = batch_diag(np.sqrt(eigvals), axis=-1)
        # Xe_sfa = Xe_sfa @ scale_factors
    return Xe_sfa

from scipy.stats import t as t_dist
def batch_pearson_memmap(dtype, shape, y_path=None,
                         mode='r', chunk_size=10_000_000, pvalue=False, eps=1e-8):
    """
    Memory-mapped, chunk-wise Pearson correlation along the last axis.

    Args:
        dtype: data type of x
        shape (tuple): full shape of x
        y_path (str, optional): filename of .npy for y; if None, uses sorted(x)
        mode (str): numpy.memmap mode
        chunk_size (int): max elements to load per chunk
        pvalue (bool): if True, also return two-tailed p-value
        eps (float): stabilizer to avoid zero-division

    Returns:
        corr (ndarray[...,]) or (corr, pvalue)
    """
    # open x (and y) as memmaps
    x = np.memmap("temp_pearson.npy", dtype=dtype, mode=mode, shape=shape)
    if y_path is None:
        # if sorting required, do it in‐place in small blocks
        y = np.empty_like(x)
        # sort each slice along last axis chunk-wise
        for idx in np.ndindex(*shape[:-1]):
            y[idx] = np.sort(x[idx], axis=-1)
    else:
        y = np.memmap(y_path, dtype=dtype, mode=mode, shape=shape)

    n = shape[-1]
    out_shape = shape[:-1]
    sum_x  = np.zeros(out_shape, np.float64)
    sum_y  = np.zeros(out_shape, np.float64)
    sum_x2 = np.zeros(out_shape, np.float64)
    sum_y2 = np.zeros(out_shape, np.float64)
    sum_xy = np.zeros(out_shape, np.float64)

    # process in chunks along the last axis
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        xs = x[..., start:stop]
        ys = y[..., start:stop]

        sum_x  += xs.sum(axis=-1)
        sum_y  += ys.sum(axis=-1)
        sum_x2 += np.einsum('...i,...i->...', xs, xs)
        sum_y2 += np.einsum('...i,...i->...', ys, ys)
        sum_xy += np.einsum('...i,...i->...', xs, ys)

    cov   = sum_xy - sum_x * sum_y / n
    var_x = sum_x2  - sum_x**2  / n
    var_y = sum_y2  - sum_y**2  / n

    denom = np.sqrt(var_x * var_y) + eps
    corr = cov / denom

    if pvalue:
        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + eps))
        p      = 2 * t_dist.sf(np.abs(t_stat), df=n - 2)
        return corr, p

    return corr

def indices_to_adjacency(indices, nval=None):
    """
    Convert a list of indices to an adjacency matrix

    Args:
        indices (array): List of indices
        nval (int): Number of nodes in the graph

    Returns:
        array: Adjacency matrix
    """
    if nval is None:
        nval = np.max(indices) + 1
    assert nval < 1000, "Too many nodes to convert to adjacency matrix"

    amat = np.zeros((nval, nval))
    for i, j in indices:
        amat[i, j] = 1
    return amat

from scipy.stats import t as t_dist
def batch_pearson(x, y=None, pvalue=False, eps=1e-8):
    """
    Memory-efficient Pearson correlation along the last axis.

    Args:
        x (ndarray[..., M]): input tensor
        y (ndarray[..., M], optional): second tensor; if None, uses sorted(x)
        pvalue (bool): if True, also return two-tailed p-value
        eps (float): small constant to avoid division by zero

    Returns:
        corr (ndarray[...]): Pearson r
        (optional) p (ndarray[...]): two-tailed p-value
    """
    if y is None:
        # unavoidable O(...×M) cost of sorting, but no additional copy for centering
        y = np.sort(x, axis=-1)

    n = x.shape[-1]

    # compute sums and sums of squares without full-array temporaries
    sum_x  = np.sum(x,  axis=-1)
    sum_y  = np.sum(y,  axis=-1)
    sum_x2 = np.einsum('...i,...i->...', x, x)
    sum_y2 = np.einsum('...i,...i->...', y, y)
    sum_xy = np.einsum('...i,...i->...', x, y)

    # covariance and variances
    cov   = sum_xy - sum_x * sum_y / n
    var_x = sum_x2  - sum_x**2  / n
    var_y = sum_y2  - sum_y**2  / n

    # Pearson r
    denom = np.sqrt(var_x * var_y) + eps
    corr = cov / denom

    if pvalue:
        # Student’s t for Pearson r
        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + eps))
        p = 2 * t_dist.sf(np.abs(t_stat), df=n - 2)
        return corr, p

    return corr

# from scipy.stats import t as t_dist
# def batch_pearson(x, y=None, pvalue=False):
#     """
#     Calculate the Pearson correlation between two sets of time series along the 
#     last axis
    
#     Args:
#         x (ndarray): A tensor of shape (..., M)
#         y (ndarray): A tensor of shape (..., M). If None, the sorted values of the 
#             x time series are used and the Pearson correlation is calculated
#             relative to these sorted values
#         pvalue (bool): Whether to return the p-value of the correlation
    
#     Returns:
#         corr (ndarray): A tensor of shape (..., N) containing the Pearson correlation
#             between each pair of datasets contracted along the last axis
#     """
#     if y is None:
#         y = np.sort(x, axis=-1)
#     xc = x.copy() - np.mean(x, axis=-1, keepdims=True)
#     yc = y.copy() - np.mean(y, axis=-1, keepdims=True)
#     corr = np.sum(xc * yc, axis=-1) / np.sqrt(np.sum(xc ** 2, axis=-1) * np.sum(yc ** 2, axis=-1))
#     # corr = np.nan_to_num(corr, nan=0.0)
#     if pvalue:
#         n = x.shape[-1]
#         t_stat = corr * np.sqrt((n - 2) / (1e-6 + 1 - corr ** 2))
#         p_value = 2 * t_dist.sf(np.abs(t_stat), df=n-2)
#         return corr, p_value
#     return corr

def batch_spearman(x, y=None, pvalue=False):
    """
    Calculate the Spearman correlation between two sets of time series along the
    last axis

    Args:
        x (ndarray): A tensor of shape (batch, N, M)
        y (ndarray): A tensor of shape (batch, N, M). If None, the indices of the
            time series are used and the Spearman correlation is calculated
            relative to a monotonic function
        pvalue (bool): Whether to return the p-value of the correlation

    Returns:
        corr (ndarray): A tensor of shape (batch, N) containing the Spearman correlation
            between each pair of time series
    """
    if y is None:
        y = np.arange(x.shape[-1])[None, :]
    xrank = np.argsort(np.argsort(x, axis=-1), axis=-1).astype(np.float32)
    yrank = np.argsort(np.argsort(y, axis=-1), axis=-1).astype(np.float32)
    return batch_pearson(xrank, yrank, pvalue=pvalue)

def progress_bar(i, n, n_bar=20):
    """
    Print a progress bar to stdout

    Args:
        i (int): Current iteration
        n (int): Total number of iterations
        n_bar (int): Number of characters in the progress bar

    Returns:
        None
    """
    idots = int(i / n * n_bar)
    stars = '#' * idots
    spaces = ' ' * (n_bar - idots)
    bar_str = f"[{stars}{spaces}] "
    print(bar_str, end='\r')
    if i == n - 1:
        print("\n")
    return None

def flatten_along_axis(a, axes):
    """
    Flattens a NumPy array along the specified axes.

    Args:
        array (np.ndarray): The input array.
        axes (tuple or list): Axes to flatten.

    Returns:
        np.ndarray: A reshaped array with specified axes flattened.
    """
    # Ensure axes are sorted for consistent behavior
    axes = sorted(axes)
    if any(ax < 0 or ax >= a.ndim for ax in axes):
        raise ValueError(f"Axes {axes} are out of bounds for array with {a.ndim} dimensions.")

    # First we reorder the axes to flatten at the end
    unflattened_axes = [i for i in range(a.ndim) if i not in axes]
    new_order = unflattened_axes + axes
    reordered = np.moveaxis(a, new_order, range(a.ndim))
    flattened_shape = reordered.shape[:len(unflattened_axes)] + (-1,)
    return reordered.reshape(flattened_shape)

from sklearn.linear_model import Ridge
from scipy.stats import t
def max_linear_correlation_ridge(A, B, alpha=1e-3, return_pvalue=False):
    """
    Compute the maximum linear correlation between each column of B and any linear 
    combination of columns of A. A represents the cause(s) and B the effect.

    Args:
        A (np.ndarray): A matrix of shape (T, N)
        B (np.ndarray): A matrix of shape (T, M)
        alpha (float): Ridge regularization parameter. Defaults to 1e-3
        return_pvalue (bool): Whether to return the p-value of the correlation. 
            Defaults to False

    Returns:
        np.ndarray: An array of shape (M,) containing the maximum correlation
            between each column of B and a linear combination of columns of A
    """
    # ATA = A.T @ A
    # X = np.linalg.solve(ATA + alpha * np.eye(A.shape[1]), A.T @ B)  # shape (N, M)
    # Y = A @ X  # shape (T, M)
    # Y_mean = Y.mean(axis=0)
    # B_mean = B.mean(axis=0)
    # Y_std = Y.std(axis=0, ddof=1)
    # B_std = B.std(axis=0, ddof=1)
    # cov = np.sum((Y - Y_mean) * (B - B_mean), axis=0) / (A.shape[0] - 1)
    # return cov / (Y_std * B_std)

    # Fit the ridge model (no intercept needed for correlation)
    n_feats = A.shape[1]
    model = Ridge(alpha=alpha * n_feats, fit_intercept=False)
    model.fit(A, B)
    Y = model.predict(A) # T x N
    Y_mean = Y.mean(axis=0)
    B_mean = B.mean(axis=0)
    Y_centered = Y - Y_mean
    B_centered = B - B_mean
    numer = np.sum(Y_centered * B_centered, axis=0)
    denom = np.sqrt(np.sum(Y_centered**2, axis=0) * np.sum(B_centered**2, axis=0))
    corr = numer / denom
    ## correct for subset size
    # corr *= np.sqrt(A.shape[0] / (A.shape[0] - 2))
    if return_pvalue:
        n = A.shape[0]
        df = n - 2
        corr_clipped = np.clip(corr, -0.9999999999, 0.9999999999)
        t_stat = corr_clipped * np.sqrt(df / (1 - corr_clipped**2))
        pval = 2 * (1 - t.cdf(t_stat, df))
        return corr, pval
    else:
        return corr
    
from scipy.sparse.linalg import cg, LinearOperator
def approx_ridge_cg(A, b, lambda_reg, rtol=1e-6, atol=1e-8, maxiter=None, use_precond=True):
    """
    Solve min_x ||A x - b||^2 + λ||x||^2 approximately via CG,  without forming A^T A 
    explicitly.

    Args:
        A (np.ndarray): (T, M) design matrix
        b (np.ndarray): (T,) target vector
        lambda_reg (float): Ridge regularization parameter
        tol (float): CG tolerance
        maxiter (int): Maximum number of CG iterations

    Returns:
        x (np.ndarray): (M,) solution vector
    """
    M = A.shape[1]

    # Define the linear operator for (AᵀA + λI)
    def mv(v):
        return A.T @ (A @ v) + lambda_reg * v
    linop = LinearOperator((M, M), matvec=mv)

    # Build simple diagonal preconditioner M ≈ (AᵀA + λI)^(-1)
    M_inv = None
    if use_precond:
        # diag_j = ∑ₜ Aₜⱼ² + λ
        diag = np.einsum('tm,tm->m', A, A) + lambda_reg
        M_inv = LinearOperator((M, M), matvec=lambda v: v / diag)

    # Solve using CG with both rtol and atol
    x, info = cg(
        linop,
        A.T @ b,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter or M,
        M=M_inv
    )
    if info > 0:
        raise RuntimeError(f"CG failed to converge after {info} iterations")
    return x

from scipy.sparse.linalg import lsqr
def ridge_lsqr(A, b, lambda_reg, tol=1e-8):
    """
    Solve min ||A x - b||^2 + λ ||x||^2 via the damped LSQR algorithm.

    Args:
        A (np.ndarray): (T, M) design matrix
        b (np.ndarray): (T,) target vector
        lambda_reg (float): Ridge regularization parameter
        tol (float): Tolerance for the solution

    Returns:
        x (np.ndarray): (M,) solution vector
    """
    damp = np.sqrt(lambda_reg)
    sol = lsqr(A, b, damp=damp, atol=tol, btol=tol, iter_lim=A.shape[1])
    return sol[0]

def banded_matrix(n, r, m=None):
    """
    Create a banded matrix with a band radius of r around the main diagonal

    Args:
        n (int): Number of rows
        r (int): Band radius
        m (int): Number of columns. Defaults to None, in which case m = n

    Returns:
        np.ndarray: A banded matrix of shape (n, m)
    """
    if m is None:
        m = n
    a = np.ones((n, m))
        # Create a mask for elements within the radius r from the main diagonal
    row_indices = np.arange(n)[:, None]
    col_indices = np.arange(m)
    
    # Calculate the mask to determine which elements should remain as 1
    mask = np.abs(row_indices - col_indices) <= r
    
    # Set elements outside the band radius to zero
    a[~mask] = 0
    
    return a

def hollow_matrix(n, r=1, m=None):
    """
    Create a hollow matrix with a band radius of r around the main diagonal

    Args:
        n (int): Number of rows
        r (int): Band radius
        m (int): Number of columns. Defaults to None, in which case m = n

    Returns:
        np.ndarray: A hollow matrix of shape (n, m)
    """
    return 1 - banded_matrix(n, r, m=m)

import time
import os

class TimeTracker:
    def __init__(self):
        self.last_time = time.perf_counter()
time_tracker = TimeTracker()

def debug_print(i, name="dump.txt", include_time=True, reset=False):
    """
    Print a debug message to a file on disk. Creates the file if it doesn't exist.

    Args:
        i (int): The debug message to print
        name (str): The name of the file to write to
        include_time (bool): Whether to include the current time in the debug message
        reset (bool): Whether to reset the file. If True, the file is deleted if it exists.
    """
    if reset:
        if os.path.exists(name):
            os.remove(name)

    with open(name, "a") as f:
        if include_time:
            current_time = time.perf_counter()
            elapsed_time = current_time - time_tracker.last_time
            time_tracker.last_time = current_time
            f.write(f"{elapsed_time:.3f}s: {i}\n")
        else:
            f.write(f"{i}\n")

def unique_dict(dict_list, duplicate_keys):
    """
    Remove duplicates from a list of dictionaries based on specified keys, preserving order.
    Dictionaries are considered duplicates if they have the same values for all specified keys.

    Args:
        dict_list (list): A list of dictionaries
        duplicate_keys (list): A list of keys to check for duplicates. Dictionaries with
            the same values for these keys will be considered duplicates.

    Returns:
        list: A list of dictionaries with duplicates removed, preserving the original order
    """
    seen = set()
    unique_dicts = []
    
    for d in dict_list:
        # Create a tuple of values for the specified keys
        key_values = tuple(d.get(k) for k in duplicate_keys)
        
        # If we haven't seen these values before, add the dictionary
        if key_values not in seen:
            seen.add(key_values)
            unique_dicts.append(d)
            
    return unique_dicts



class StreamingCorrelation:
    """
    Compute the Pearson correlation between each entry of a streaming
    M x M array and a known monotonic function g(i) in one pass.

    Parameters:
        M (int): Spatial dimension of each frame (frames are M×M).
        g_func (callable): Monotonic function g(i) of the time index i (0-based or 1-based).
        dtype (data-type): Numeric type for accumulators.
        tol (float): Tolerance for the solution.
    
    While this yields Pearson, for a strictly increasing g the
    Pearson trend coefficient often closely tracks Spearman's rho.

    Example:
        >>> model = StreamingCorrelation(100, lambda i: i)
        >>> for i in range(1000):
        >>>     model.update(np.random.normal(size=(100, 100)))
        >>> rho = model.finalize()
        >>> print(rho) # (100, 100) array of Pearson trend correlations
    """

    def __init__(self, M, g_func=None, dtype=np.float64, tol=1e-10):
        self.M = M
        self.g = g_func
        # accumulators for sums
        self.Sx = np.zeros((M, M), dtype=dtype)
        self.Sxx = np.zeros((M, M), dtype=dtype)
        self.Sxg = np.zeros((M, M), dtype=dtype)
        # scalar sums for g
        self.g_sum = 0.0
        self.g2_sum = 0.0
        self.n = 0
        self.tol = tol

    def update(self, frame):
        """
        Incorporate the next M×M frame.

        Args:
            frame (np.ndarray, shape (M, M)): Next observation in the time series.
        """
        i = self.n
        # gi = self.g(i)
        gi = i # Assume a linear trend
        self.n += 1

        # update spatial sums
        self.Sx += frame
        self.Sxx += frame * frame
        self.Sxg += frame * gi

        # update scalar sums
        self.g_sum += gi
        self.g2_sum += gi * gi

    def finalize(self):
        """
        Compute the per-pixel correlation matrix.

        Returns: 
            rho (np.ndarray, shape (M, M)): Pearson trend correlation with g; 
            for strictly increasing g, often a good proxy for Spearman rho.
        """
        if self.n < 2:
            raise ValueError("Need at least two frames to compute correlation.")

        # covariance numerator
        num = self.Sxg - (self.Sx * (self.g_sum / self.n))

        # variance denominators
        var_x = self.Sxx - (self.Sx * self.Sx) / self.n
        var_g = self.g2_sum - (self.g_sum * self.g_sum) / self.n
        den = np.sqrt(var_x * var_g)
        pearson = num / (den + self.tol)

        ## Convert to Spearman via the Gaussian relation assuming bivariate normal
        spearman = (6 / np.pi) * np.arcsin(pearson / 2)

        return spearman