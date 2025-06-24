import numpy as np

import warnings

class WeightedCorrelations:
    """
    Finds the weighted correlations between the columns of a given dataset.

    Attributes:
        method (str): The method to use for computing the correlations.
        beta (float): The exponent to use for the correlations.
        tau (int): The lag to use for the correlations.

    Methods:
        fit(X0, weights=None): Computes the correlation matrix for the given data.

    References:
        Langfelder, Peter, and Steve Horvath. "WGCNA: an R package for weighted 
        correlation network analysis." BMC bioinformatics 9 (2008): 1-13.
    """

    def __init__(self, method="pearson", beta=1, tau=0):
        self.method = method
        self.beta = beta
        self.tau = tau

    def fit(self, X0, weights=None):
        """
        Computes the correlation matrix for the given data.

        Args:
            X0 (numpy.ndarray): Input data of shape (n_times, n_features).
            weights (numpy.ndarray, optional): Weights for each observation of 
                shape (n_times,). If None, unweighted correlations are computed.

        Raises:
            ValueError: If the specified method is not supported.
        """

        if self.method not in ["pearson", "spearman"]:
            warnings.warn(
                "Only Pearson and Spearman correlations are supported. Falling back to Pearson."
            )
            self.method = "pearson"
            
        X = np.asarray(X0).copy()
        n = X.shape[1]
        if self.method == "spearman":
            X = np.argsort(np.argsort(X, axis=0), axis=0)

        if self.tau > 0:
            Y = X[self.tau:]
            X = X[:-self.tau]
        elif self.tau < 0:
            Y = X[:-np.abs(self.tau)]
            X = X[np.abs(self.tau):]
        else:
            Y = X
        
        self.corr_matrix = np.corrcoef(X, Y, rowvar=False)[:n, :n]
    
        # ## Compute the Pearson correlation matrix
        # Xmean = np.mean(X, axis=0, keepdims=True)
        # Ymean = np.mean(Y, axis=0, keepdims=True)
        # cov_matrix = np.dot((X - Xmean).T, (Y - Ymean))
        # Xvar = np.sum((X - Xmean) ** 2, axis=0, keepdims=True)
        # Yvar = np.sum((Y - Ymean) ** 2, axis=0, keepdims=True)
        # std_X = np.sqrt(Xvar)
        # std_Y = np.sqrt(Yvar)
        # std_X[std_X == 0] = 1e-10
        # std_Y[std_Y == 0] = 1e-10

        # self.corr_matrix = cov_matrix / np.outer(std_X, std_Y)
        # self.corr_matrix = self.corr_matrix ** self.beta

        return np.abs(self.corr_matrix)


class LaggedCorrelations:
    """
    A class to compute lagged correlations for a given dataset.

    Attributes:
        max_lag (float): The maximum lag to consider for the correlations.
        corr_kwargs (dict): Keyword arguments to pass to the WeightedCorrelations class.

    Methods:
        fit(X): Computes the lagged correlations for the given data.

    """

    def __init__(self, max_lag=0.1, **corr_kwargs):
        self.max_lag = max_lag
        self.corr_kwargs = corr_kwargs

    def fit(self, X):
        max_tau = int(self.max_lag * X.shape[0])
        
        all_corrs = list()
        for tau in range(-max_tau, max_tau):
            model = WeightedCorrelations(**self.corr_kwargs, tau=tau)
            corr = model.fit(X)
            all_corrs.append(corr.copy())
        all_corrs = np.array(all_corrs)
        top_ind = np.argmax(np.mean(np.abs(all_corrs), axis=(1, 2)))
        # print(np.mean(np.abs(all_corrs), axis=(1, 2)))
        return all_corrs[top_ind]

