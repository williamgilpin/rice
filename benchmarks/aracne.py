"""
A re-implementation of the ARACNE algorithm for inferring gene regulatory networks.

This implementation is based on the original paper [1] and the open-source 
implementation available at
https://github.com/rugrag/ARACNE-gene-network-inference
However, it has been modified to use vectorized operations and faster KDE

References:
    [1] Margolin, Adam A., et al. "ARACNE: an algorithm for the reconstruction of gene 
        regulatory networks in a mammalian cellular context." BMC bioinformatics. 
        Vol. 7. BioMed Central, 2006.
"""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

class ARACNE:
    """
    Optimized ARACNE algorithm for inferring gene regulatory networks.
    
    Parameters:
        n_permutations (int): Number of permutations to use for the permutation test.
        random_state (int): Random seed.

    Attributes:
        count_matrix (np.ndarray): Count matrix of shape (n_samples, n_genes).
        n_samples (int): Number of samples.
        n_genes (int): Number of genes.
        mutual_info_matrix (np.ndarray): Mutual information matrix of shape 
            (n_genes, n_genes).
        mutual_info_matrix_alternative (np.ndarray): Null distribution of the 
            mutual information matrix of shape (n_permutations, n_genes, n_genes).
        mutual_info_matrix_filtered (np.ndarray): Filtered mutual information 
            matrix of shape (n_genes, n_genes).
        network (np.ndarray): Inferred network as an upper triangular affinity
            matrix of shape (n_genes, n_genes).

    References:
        [1] Margolin, Adam A., et al. BMC Bioinformatics. Vol. 7. 2006.
    """
    def __init__(self, n_permutations=5, random_state=None):
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(self.random_state)

    def p_kde(self, x, X, h):
        kde = gaussian_kde(X, bw_method=h)
        return kde(x)[0]

    def kernel_mi(self, X, Y):
        """
        Calculate the mutual information between two variables using kernel density 
        estimation.
        """
        d = 2
        nx = len(X)
        hx = (4 / (d + 2)) ** (1 / (d + 4)) * nx ** (-1 / (d + 4))

        # Compute KDE for X, Y, and their joint distribution
        px = np.array([self.p_kde(X[i], X, hx) for i in range(nx)])
        py = np.array([self.p_kde(Y[i], Y, hx) for i in range(nx)])
        joint_data = np.vstack((X, Y))
        pxy = np.array([self.p_kde(np.array([X[i], Y[i]]), joint_data, hx) for i in range(nx)])

        # Calculate mutual information
        mi_estimation = np.log(pxy / (px * py))
        return np.mean(mi_estimation)
    
    def kernel_mi(self, X, Y, d=2):
        nx = len(X)
        hx = (4 / (d + 2)) ** (1 / (d + 4)) * nx ** (-1 / (d + 4))

        # Compute KDE for X, Y, and their joint distribution
        kde_X = gaussian_kde(X, bw_method=hx)
        px = kde_X(X)
        kde_Y = gaussian_kde(Y, bw_method=hx)
        py = kde_Y(Y)
        joint_data = np.array([X, Y])
        kde_joint = gaussian_kde(joint_data, bw_method=hx)
        pxy = kde_joint(joint_data)

        # Calculate mutual information
        mi_estimation = np.log(pxy / (px * py))
        return np.mean(mi_estimation)

    def calculate_mutual_info_matrix(self, X):
        """
        Calculate the mutual information matrix for a given data matrix. This 
        calculation is symmetric, so only the upper triangular portion of the
        matrix is stored.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_genes).

        Returns:
            mutual_info_matrix (np.ndarray): Mutual information matrix of shape 
                (n_genes, n_genes).
        """
        d = X.shape[1]
        mutual_info_matrix = np.zeros((d, d))
        for ix in np.arange(d):
            for jx in np.arange(ix + 1, d):
                mutual_info_matrix[ix, jx] = self.kernel_mi(X[:, ix], X[:, jx])
        return mutual_info_matrix

    def permutation_test(self):
        """
        Perform a permutation test to estimate the null distribution of the
        mutual information matrix.
        """
        mutual_info_matrix_alternative = np.zeros((self.n_permutations, self.n_genes, self.n_genes))
        for n in range(self.n_permutations):
            shuffled_X = np.apply_along_axis(np.random.permutation, 1, self.count_matrix)
            mutual_info_matrix_alternative[n, ...] = self.calculate_mutual_info_matrix(shuffled_X)
        return np.mean(mutual_info_matrix_alternative, axis=0)

    def filter_mi(self):
        """
        Filter the mutual information matrix based on the alternative
        permutation test.

        Returns:
            mutual_info_matrix_filtered (np.ndarray): Filtered mutual information 
                matrix of shape (n_genes, n_genes).
        """
        mutual_info_matrix_filtered = np.copy(self.mutual_info_matrix)
        I_0 = np.amax(self.mutual_info_matrix_alternative)
        mutual_info_matrix_filtered[mutual_info_matrix_filtered < I_0] = 0
        return mutual_info_matrix_filtered

    def data_processing_inequality(self, M, i, j, k):
        """
        The Data Processing Inequality (DPI) can be used to filter out edges that
        result from indirect relationships, where X -> Y -> Z induces a non-zero
        mutual information between X and Z.

        The criterion is that if I[i, k] < min(I[i, j], I[j, k]),
        then the edge from i to k is filtered.
        """
        dic = {0: (i, j), 1: (i, k), 2: (j, k)}
        dpi_list = [M[i, j], M[i, k], M[j, k]]
        idx = np.argmin(dpi_list)
        return dic[idx]

    def filter_loops(self, M):
        """
        Filter out loops from the mutual information matrix based on the
        Data Processing Inequality (DPI).

        Returns:
            M (np.ndarray): Filtered mutual information matrix
        """
        set_to_zero = []
        for i in range(M.shape[0]):
            idx_j = np.where(M[i] != 0)[0]  # Indices where M[i] is non-zero
            for j in idx_j:
                idx_k = np.where(M[j] != 0)[0]  # Indices where M[j] is non-zero
                valid_k = idx_k[M[i, idx_k] != 0]
                if valid_k.size > 0:
                    set_to_zero.extend(
                        [self.data_processing_inequality(M, i, j, k) for k in valid_k]
                    )

        if set_to_zero:
            set_to_zero = np.array(set_to_zero).T
            M[tuple(set_to_zero)] = 0
        
        return M

    def fit_transform(self, count_matrix):
        """
        Fit the ARACNE algorithm to the count matrix and return the inferred
        network as an upper triangular affinity matrix.

        Args:
            count_matrix (np.ndarray): Count matrix of shape (n_samples, n_genes).

        Returns:
            network (np.ndarray): Inferred network as an upper triangular affinity
                matrix of shape (n_genes, n_genes).
        """
        self.count_matrix = count_matrix
        self.n_samples, self.n_genes = count_matrix.shape
        scaler = StandardScaler()
        count_matrix = scaler.fit_transform(count_matrix)
        self.mutual_info_matrix = self.calculate_mutual_info_matrix(count_matrix)
        self.mutual_info_matrix_alternative = self.permutation_test()
        self.mutual_info_matrix_filtered = self.filter_mi()
        self.network = self.filter_loops(self.mutual_info_matrix_filtered)
        return self.network
