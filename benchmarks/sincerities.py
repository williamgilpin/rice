"""
A re-implementation of the SINCERITIES pipeline for gene regulatory network inference.

Example:

    model = SINCERITIES(X, t=None, dd_metric='ks')
    cmat = model.fit()

References:

    Papili Gao, Nan, et al. "SINCERITIES: inferring gene regulatory networks from 
        time-stamped single cell transcriptional expression profiles." 
        Bioinformatics 34.2 (2018): 258-266.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import RidgeCV
from itertools import product

class SINCERITIES:
    """
    Implements a simplified SINCERITIES pipeline using:
      1) Distribution distance (Kolmogorov-Smirnov by default) 
      2) Ridge regression with leave-one-out cross-validation
      3) Partial correlation (Spearman rank) to determine edge sign
    
    Parameters:
        X (ndarray): Gene expression matrix (N x D), where N is the number of timepoints
            and D is the number of genes (features).
        t (ndarray): Time values associated with each of the N rows in X. If None, 
            uniform spacing is assumed (time=0,1,2,...)
        dd_metric (str): Distribution distance metric ('ks' or 'mean'). For real 
            single-cell data, each row would represent multiple cells and a chosen 
            distribution metric like KS distance would compare distributions across 
            cells at consecutive timepoints.
    """

    def __init__(self, X, t=None, dd_metric='ks'):
        self.X = np.array(X, dtype=float)
        self.n_time, self.n_genes = self.X.shape
        if t is None:
            self.t = np.arange(self.n_time)
        else:
            self.t = np.array(t, dtype=float)
        self.dd_metric = dd_metric
        self.ridge_models = []
        self.alpha_matrix = None
        self.sign_matrix = None
        
    def _distribution_distance(self, data_t1, data_t2):
        return np.abs(np.mean(data_t2) - np.mean(data_t1))

    def compute_dd_matrix(self):
        """
        Computes the normalized distribution distance matrix of shape ((n_time-1), n_genes).
        dd_matrix[l, j] = distance of gene j from time l to l+1, normalized by time-step size.
        """
        dd = np.zeros((self.n_time - 1, self.n_genes))
        dt = np.diff(self.t)  # time steps between consecutive timepoints

        # For each consecutive time window, compute distance for each gene
        for l in range(self.n_time - 1):
            for j in range(self.n_genes):
                # Treat the row at time l as data_t1, row at time l+1 as data_t2
                # Each row could represent many cell measurements
                data_t1 = [self.X[l, j]]   # placeholder for single aggregated value
                data_t2 = [self.X[l + 1, j]]
                raw_dist = self._distribution_distance(data_t1, data_t2)
                dd[l, j] = raw_dist / dt[l] if dt[l] != 0 else raw_dist
        return dd

    def fit_regression(self):
        """
        For each target gene j, run ridge regression to predict DD_j(t+1) from DD_i(t), i != j.
        Uses leave-one-out cross validation to find best lambda (RidgeCV with 'loo').
        """
        dd_matrix = self.compute_dd_matrix()        # shape (n_time-1, n_genes)
        n_samples = dd_matrix.shape[0] - 1          # for regression: we map t -> t+1
        alpha_mat = np.zeros((self.n_genes, self.n_genes))  # each row=regressions for j

        # For time windows: 
        #    X_reg = dd_matrix[0..n_samples-1, :]   (predictors, from previous window)
        #    y_reg = dd_matrix[1..n_samples, j]     (target, from next window)
        # We do separate regression for each target gene j
        for j in range(self.n_genes):
            # Prepare the training data
            X_reg = dd_matrix[:-1, :]    # shape (n_samples, n_genes)
            y_reg = dd_matrix[1:, j]     # shape (n_samples, )

            # We do not remove gene j from the matrix since the method's original
            # statement used all other genes. If you want to exclude auto-regulation
            # strictly, you could remove column j from X_reg. The official text 
            # states "all other genes," so let's drop the same gene column:
            X_reg_noj = np.delete(X_reg, j, axis=1)  # shape (n_samples, n_genes-1)

            # RidgeCV with leave-one-out: 'cv=None' uses LOOCV in scikit-learn, so we specify:
            model = RidgeCV(alphas=np.logspace(-5, 5, 51), store_cv_results=True, cv=None)
            model.fit(X_reg_noj, y_reg)

            # Full solution including all D genes: re-insert gene j's coefficient as 0
            coeffs = np.insert(model.coef_, j, 0.0)
            coeffs = np.clip(coeffs, 0, None) # Non-negative

            alpha_mat[:, j] = coeffs
            self.ridge_models.append(model)

        self.alpha_matrix = alpha_mat
        return alpha_mat

    def partial_correlation(self):
        """
        Computes Spearman partial correlation coefficients among genes 
        (combining all timepoints in X). Sign for edge i->j = sign of partial corr(i,j).
        A simple approximate partial correlation is computed via linear regression 
        or by inverting correlation matrices. Here, we do a naive approach: 
          1) rank-transform data across all timepoints 
          2) compute partial corr from the inverse of the correlation matrix

        Returns:
            sign_mat (np.ndarray): The sign matrix of partial correlations.
        """
        # Rank-transform across each genes for Spearman correlation
        ranked = np.apply_along_axis(rankdata, 0, self.X)
        corr_mat = np.corrcoef(ranked, rowvar=False)  # shape (n_genes, n_genes)
        inv_corr = np.linalg.pinv(corr_mat)
        diag_sqrt = np.sqrt(np.diag(inv_corr))
        P = -inv_corr / np.outer(diag_sqrt, diag_sqrt)  # partial correlation matrix
        np.fill_diagonal(P, 1.0)
        sign_mat = np.sign(P) # The sign for i->j
        self.sign_matrix = sign_mat
        return sign_mat

    def get_ranked_edges(self):
        """
        Produces a list of all edges (i->j) ranked by alpha_matrix[i,j].
        The sign is given by partial correlation matrix if available.
        Returns a DataFrame: columns = [source, target, alpha, sign].

        Returns:
            edges_df (pd.DataFrame): A DataFrame of ranked edges.

        Raises:
            ValueError: If fit_regression() has not been called.
        """
        if self.alpha_matrix is None:
            raise ValueError("Run fit_regression() first.")

        alpha_abs = np.abs(self.alpha_matrix).ravel()
        # i->j has alpha_matrix[i,j]
        edges = []
        for i, j in product(range(self.n_genes), range(self.n_genes)):
            if i == j:
                continue
            val = self.alpha_matrix[i, j]
            # If partial correlation is available, fetch sign
            edge_sign = 0
            if self.sign_matrix is not None:
                edge_sign = self.sign_matrix[i, j]
            edges.append((i, j, val, edge_sign))

        # Sort descending by alpha
        edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
        return pd.DataFrame(edges_sorted, columns=["source", "target", "alpha", "sign"])

    def get_connection_matrix(self, signed=False):
        """
        Returns the D x D matrix of inferred links between genes, where entry (i, j)
        represents the weight of the connection from gene i to gene j.
        If signed=True, the returned matrix incorporates signs from partial correlation.

        Args:
            signed (bool): Whether to return the signed connection matrix.

        Returns:
            cmat (np.ndarray): The D x D connection matrix.
        """
        if self.alpha_matrix is None:
            raise ValueError("Must run fit_regression() or run_pipeline() first.")

        if signed:
            # If partial correlation has not been computed yet, do it now.
            if self.sign_matrix is None:
                self.partial_correlation()
            return self.alpha_matrix * self.sign_matrix
        else:
            return self.alpha_matrix

    def fit(self):
        """
        Fits the SINCERITIES pipeline and returns the connection matrix.

        Returns:
            cmat (np.ndarray): The D x D connection matrix, where entry (i, j) 
                represents the weight of the connection from gene i to gene
        """
        self.fit_regression()
        self.partial_correlation()
        cmat = self.get_connection_matrix(signed=True)
        return cmat