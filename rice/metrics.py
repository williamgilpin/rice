import numpy as np
import warnings

def minmax_scaler(data, axis=None, tolerance=1e-8):
    """
    Scale the data to the range [0, 1] along the specified axis

    Args:
        data (ndarray): The data to scale
        axis (int): The axis along which to scale the data
        tolerance (float): A small value to add to the denominator to prevent division by zero

    Returns:
        scaled_data (ndarray): The scaled data
    """
    numerator = data - np.min(data, axis=axis, keepdims=True)
    denominator = np.max(data, axis=axis, keepdims=True) - np.min(data, axis=axis, keepdims=True)
    return numerator / (denominator + tolerance)


def dense_score(atrue0, apred0, score_func, nan_policy="omit", check_transpose=True):
    """
    Compute the a score between two adjacency matrices, accounting for missing
    values in the ground truth.

    Args:
        atrue0 (ndarray): The true adjacency matrix
        apred0 (ndarray): The predicted adjacency matrix
        score_func (function): The scoring function to use
        nan_policy (str): The policy to use for handling NaN values. Options are "omit" 
            and "zero"
        check_transpose (bool): Whether to also score the correlation of the transpose of 
            the predicted matrix with the original matrix, as well as the predicted
            matrix against the transpose of the original matrix. 
            This helps account for different orientations of matrices returned
            by different methods

    Returns:
        score (float): The score between the two matrices
    """
    atrue, apred = atrue0.ravel().copy(), apred0.ravel().copy()

    if nan_policy == "omit":
        valid_indices = np.where(~np.isnan(atrue))[0]
        atrue, apred = atrue[valid_indices], apred[valid_indices]
    elif nan_policy == "zero":
        atrue[np.isnan(atrue)] = 0
        apred[np.isnan(apred)] = 0
    else:
        warnings.warn("Invalid NaN policy. Defaulting to 'omit'.")
        valid_indices = np.where(~np.isnan(atrue))[0]
        atrue, apred = atrue[valid_indices], apred[valid_indices]

    atrue = (atrue > 0).astype(int) # Ensure the ground truth is binary
    apred = minmax_scaler(apred)
    score = score_func(atrue, apred)
    if check_transpose:
        score = max(
            score, 
            dense_score(atrue0, apred0.T, score_func, nan_policy=nan_policy, check_transpose=False),
            dense_score(atrue0.T, apred0, score_func, nan_policy=nan_policy, check_transpose=False)
        )
    return score


from sklearn.metrics import average_precision_score
def auprc_score(atrue, apred, **kwargs):
    """
    Compute the AUPRC score between two adjacency matrices, omitting any NaN values
    from the calculation

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        check_transpose (bool): Whether to check the correlation of the transpose of
            the predicted matrix

    Returns:
        auprc_score (float): The AUPRC score between the two matrices
    """
    return dense_score(atrue, apred, average_precision_score, **kwargs)

from sklearn.metrics import roc_auc_score
def rocauc_score(atrue, apred, **kwargs):
    """
    Compute the ROC-AUC score between two adjacency matrices, omitting any NaN values
    from the calculation

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        check_transpose (bool): Whether to check the correlation of the transpose of
            the predicted matrix

    Returns:
        auc_score (float): The ROC-AUC score between the two matrices
    """
    return dense_score(atrue, apred, roc_auc_score, **kwargs)

def early_precision(y_true, y_pred, check_transpose=True, normalize=False):
    """
    Compute the early precision between two adjacency matrices, defined as the number of
    correct edges in the top-k predicted edges. k is set to the number of edges in the
    true adjacency matrix

    Args:
        y_true (ndarray): An array of true values
        y_pred (ndarray): An array of predicted values

    Returns:
        early_precision (float): The early precision between the two matrices
    """
    num_true_edges = np.sum(y_true)
    threshold = np.sort(y_pred)[-num_true_edges]
    y_pred_thresh = y_pred > threshold
    early_precision = int(np.sum(y_true[y_pred > threshold]))
    if normalize:
        early_precision /= num_true_edges
    return early_precision

def early_precision_score(atrue, apred, **kwargs):
    """
    Compute the early precision between two adjacency matrices, omitting any NaN values
    from the calculation

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        check_transpose (bool): Whether to check the correlation of the transpose of
            the predicted matrix

    Returns:
        early_precision (float): The early precision between the two matrices
    """
    return dense_score(atrue, apred, early_precision, **kwargs)


def top_k_accuracy(atrue, apred, k=10, check_transpose=True):
    """
    Compute the top-k accuracy between two adjacency matrices, defined as the number of
    top-k predicted edges that are correct

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        k (int): The number of top edges to consider

    Returns:
        accuracy (float): The top-k accuracy between the two matrices
    """
    apred_thresh = top_k_threshold(apred, k=k)
    num_incorrect = np.sum((atrue != apred) * apred_thresh)
    return 1 - num_incorrect / k

def topk_accuracy_score(atrue, apred, k=10, **kwargs):
    """
    Compute the top-k accuracy between two adjacency matrices, omitting any NaN values
    from the calculation

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        k (int): The number of top edges to consider
        check_transpose (bool): Whether to check the correlation of the transpose of
            the predicted matrix

    Returns:
        accuracy (float): The top-k accuracy between the two matrices
    """
    return dense_score(atrue, apred, top_k_accuracy, **kwargs)

def top_k_precision(atrue, apred, k=10, check_transpose=True):
    """
    Compute the top-k precision between two adjacency matrices

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        k (int): The number of top edges to consider

    Returns:
        precision (float): The top-k precision between the two matrices
    """
    atrue = atrue.copy()
    apred = apred.copy()
    atrue_thresh = atrue > np.percentile(atrue, 100 - (k / atrue.size) * 100)
    apred_thresh = apred > np.percentile(apred, 100 - (k / atrue.size) * 100)
    if check_transpose:
        return max(top_k_precision(atrue, apred, k=k, check_transpose=False),
                   top_k_precision(atrue, apred.T, k=k, check_transpose=False))

    return np.sum(np.logical_and(atrue_thresh, apred_thresh)) / np.sum(apred_thresh)


def top_k_threshold(arr, k=10):
    """
    Compute a thresholded array with ones in the top-k values

    Args:
        arr (ndarray): The array to compute the threshold for
        k (int): The number of top values to consider

    Returns:
        thresholded (ndarray): The thresholded array
    """
    thresholded = np.zeros_like(arr)
    threshold = np.percentile(arr, 100 - (k / arr.size) * 100)
    thresholded[arr > threshold] = 1
    return thresholded


def bootstrap_graph(amat, n_samples=1000, method="resample", seed=None):
    """
    Bootstrap a graph by resampling edges

    Args:
        amat (ndarray): The adjacency matrix to bootstrap
        n_samples (int): The number of samples to take
        method (str): The method to use for bootstrapping. Options are "resample", "permute", and "uniform". 
            "resample" resamples the edges with replacement, "permute" permutes the edges, and "uniform" samples
            uniformly from the range of values in the adjacency matrix
        seed (int): The random seed to use

    Returns:
        bootstrapped (ndarray): The bootstrapped adjacency matrices, shape (n_samples, *amat.shape)
    """
    np.random.seed(seed)
    n, m = amat.shape
    bootstrapped = np.zeros((n_samples, n, m))
    for i in range(n_samples):
        if method == "resample":
            bootstrapped[i] = np.random.choice(amat.ravel(), size=(n, m), replace=True).reshape(n, m)
        elif method == "permute":
            bootstrapped[i] = np.random.permutation(amat.ravel()).reshape(n, m)
        elif method == "uniform":
            bootstrapped[i] = np.random.uniform(np.min(amat), np.max(amat), size=(n, m))
        else:
            return bootstrap_graph(amat, n_samples=n_samples, method="resample", seed=seed)
    return bootstrapped


def empirical_percentile(arr, target):
    """
    Compute the empirical percentile of a target value in an array

    Args:
        arr (ndarray): The array to compute the percentile in
        target (float): The target value

    Returns:
        percentile (float): The percentile of the target value in the array
    """
    return np.sum(arr < target) / len(arr)

def reachable_nodes(adj_matrix, start_node):
    """
    Find all nodes downwards reachable from the start_node in a graph represented by 
    an adjacency matrix.
    
    Args:
        adj_matrix (np.ndarray): Binary adjacency matrix of a graph.
        start_node (int): The starting node for the search.
    
    Returns:
        List[int]: List of nodes reachable from start_node.

    """
    num_nodes = adj_matrix.shape[0]
    visited = np.zeros(num_nodes, dtype=bool)
    reachable = []

    def dfs(node):
        if visited[node]:
            return
        visited[node] = True
        reachable.append(node)
        for neighbor in range(num_nodes):
            if adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor)

    dfs(start_node)
    return reachable

def downstream_adjacency_graph(adj_matrix, max_depth=None):
    """
    Walk an adjacency matrix to find the connected components

    Args:
        amat (ndarray): The adjacency matrix to walk
        max_depth (int): The maximum depth to walk

    Returns:
        components (list): A list of connected components

    TODO: Can save compute with dynamic programming. When a node is visited, all
        downstream nodes

    """
    adj_matrix = (adj_matrix.copy() > 0.0).astype(float)
    adj_matrix_downstream = np.copy(adj_matrix)
    for i in range(adj_matrix.shape[0]):
        reachable = np.array(reachable_nodes(adj_matrix, i))
        adj_matrix_downstream[i][reachable] = 1
    np.fill_diagonal(adj_matrix_downstream, 0)
    return adj_matrix_downstream

def compute_metrics(atrue, apred, verbose=True, significance=False, walk_downstream=False, hollow=False, **kwargs):
    """
    Compute a set of metrics between two adjacency matrices

    Args:
        atrue (ndarray): The true adjacency matrix
        apred (ndarray): The predicted adjacency matrix
        verbose (bool): Whether to print the metrics after computing them
        check_transpose (bool): Whether to check the correlation of the transpose of 
            the predicted matrix
        nan_policy (str): The policy to use for handling NaN values. Options are "omit"
            and "zero"
        significance (bool): Whether to compute significance values for the metrics by
            bootstrapping
        walk_downstream (bool): Whether to walk the ground truth adjacency matrix to find
            downstream nodes
        hollow (bool): Whether to compute the metrics on the hollow version of the adjacency
            matrix

    Returns:
        metrics (dict): A dictionary containing the computed metrics
    """
    if walk_downstream:
        atrue = downstream_adjacency_graph(np.copy(atrue))

    if hollow:
        np.fill_diagonal(atrue, 0)
        np.fill_diagonal(apred, 0)
    
    auprc_value = auprc_score(atrue.copy(), apred.copy(), **kwargs)
    auprc_baseline = np.sum(atrue > 0) / np.sum(~np.isnan(atrue))
    
    auroc_value = rocauc_score(atrue.copy(), apred.copy(), **kwargs)
    auroc_baseline = 0.5

    early_precision_value = early_precision_score(atrue.copy(), apred.copy(), **kwargs)
    early_precision_baseline = np.sum(atrue > 0)**2 / np.sum(~np.isnan(atrue))
    
    kmax = np.prod(atrue.shape)
    metrics = {
        "AUPRC Multiplier": auprc_value / auprc_baseline,
        "ROC-AUC Multiplier": auroc_value / auroc_baseline,
        "AUPRC": auprc_value,
        "ROC-AUC": auroc_value,
        "Early Precision": early_precision_value,
        "Early Precision Rate": early_precision_value / np.sum(atrue > 0),
        "Early Precision Ratio": early_precision_value / early_precision_baseline,
    }

    if significance:
        bootstrapped = bootstrap_graph(apred.copy())
        all_metrics = [[] for _ in metrics.keys()]
        for apred_boot in bootstrapped:
            # if verbose:
                # progress_bar
            metrics_boot = compute_metrics(atrue, apred_boot, verbose=False, significance=False, **kwargs)
            for i, key in enumerate(metrics.keys()):
                all_metrics[i].append(metrics_boot[key])
        for i, key in enumerate(metrics.keys()):
            metrics[key] = (metrics[key], empirical_percentile(all_metrics[i], metrics[key]))

        if verbose:
            for row in metrics.keys():
                print(f"{row}: {metrics[row][0]:.3f} {metrics[row][1]:.3f}")

    else:
        if verbose:
            for row in metrics.keys(): # print with 3 decimal places
                print(f"{row}: {metrics[row]:.3f}")

    return metrics


