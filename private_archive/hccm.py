## Trivial speediups: Pre-embed the time series
## Correlations: better to average or take total Pearson?
## Faster monotonic scaling: use decimation to sample rarer timepoints, also ensures logarithmic spacing
import itertools
import warnings
import numpy as np

from .utils import progress_bar
from .utils import FiniteDict

from .ccm import CausalDetection

## Split a set of indices into K nearly equal random groups
def split_indices(n, k=3, randomize=True):
    """
    Split a set of indices into K nearly equal random groups

    Args:
        n (int): The number of indices
        k (int): The number of splits to make
        randomize (bool): Whether to shuffle the indices

    Returns:
        list: A list of lists of indices
    """
    indices = np.arange(n)
    if randomize:
        np.random.shuffle(indices)
    out = np.array_split(indices, k)
    out = [item for item in out if len(item) > 0]

    # if len(out[-1]) == 1:
    #     out = out[:-2] + [np.concatenate(out[-2:])]

    ## if any singleton groups, split across the last two groups
    # if len(out[-1]) == 1:
    #     vals1 = np.copy(out[-2][:(len(out[-2]) // 2)])
    #     vals2 = np.copy(out[-2][(len(out[-2]) // 2):])
    #     out[-2] = vals1
    #     out[-1] = np.concatenate([vals2, out[-1]])
    return out


class HierarchicalCCM:
    """Identify causal relationships in a multivariate time series using hierarchical 
    convergent cross 

    Args:
        k (int): The number of splits to make per search iteraction
        m (int): The number of top links to track
        store_history (bool): Whether to store the history of the search
        verbose (bool): Whether to print progress
        history (list): The history of the search
        random_state (int): The random seed to use
        **kwargs: Additional keyword arguments passed to the CCM model


    Attributes:
        k (int): The number of splits to make per search iteraction
        m (int): The number of top links to track
        store_history (bool): Whether to store the history of the search
        verbose (bool): Whether to print progress
        history (list): The history of the search
        random_state (int): The random seed to use
        scores (np.ndarray): The scores of the causal graph

    """

    def __init__(self, k=3, m=100, d_embed=10, nval=None, store_history=False, verbose=False, random_state=None, **kwargs):
        self.k = k
        self.m = m
        self.d_embed = d_embed
        self.store_history = store_history
        self.verbose = verbose
        self.history = list()
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.ccm_kwargs = kwargs

        if store_history:
            self.history = list()
            if nval is None:
                warnings.warn(
                    "Storing history without specifying nval will lead to memory errors"
                )
            self.scores = np.ones((nval, nval)) * 1e-12

        self.top_links = FiniteDict(self.m)

    def get_top_links(self, return_scores=False):
        """
        Return a list of the top causal links, in descending order

        Args:
            return_scores (bool): Whether to return the scores of the links

        Returns:
            np.ndarray: The top causal links
        """
        sorted_index_list = sorted(self.top_links.data.items(), key=lambda item: item[1], reverse=True)
        sorted_index_list = np.array([item[0] for item in sorted_index_list])
        score_values = np.array([self.top_links.data[tuple(item)] for item in sorted_index_list])
        if return_scores:
            return sorted_index_list, score_values
        return sorted_index_list

    def fit(self, X, niter=50, ground_truth=None, seed=None):
        """
        Fit a causal graph to the data using a hierarchical search. This function can
        be called multiple times to refine the search.

        Args:
            X (np.ndarray): Data matrix of shape (n_timepoints, n_features)
            niter (int): The number of iterations to run
            ground_truth (np.ndarray): List of ground truth indices
            seed (int): The random seed to use

        Returns:
            None
        """
        np.random.seed(self.random_state)

        # all_index_values, all_scores = list(), list()
        for ind in range(niter):
            if self.verbose:
                progress_bar(ind, niter, n_bar=20)
            _, index_values, top_score = self.hierarchical_search(X.T)
            # _, index_values, top_score = self.hierarchical_search(X.T, random_state=seed+ind)
            if top_score == None:
                continue
            # all_index_values.append(index_values)
            # all_scores.append(top_score)
            
            
    def hierarchical_search(self, data, index_values=None, top_val=1e-16, random_state=None):
        """
        Perform a hierarchical search for causal links in the data

        Args:
            data (np.ndarray): The data matrix of shape (n_features, n_timepoints)
            index_values (np.ndarray): The true index values of the data
            top_val (float): The current value of the top link
            random_state (int): The random seed to use

        Returns:
            np.ndarray: The reduced data matrix
            np.ndarray: The reduced index values
            float: The top value
        """

        n = data.shape[0]
    
        if index_values is None:
            index_values = np.arange(n)

        if n <= 2:
            return data, index_values, top_val
        else:
            np.random.seed(random_state)
            indices = split_indices(n, self.k)
            data_groups = [data[idx].T for idx in indices]
            index_values_groups = [index_values[idx] for idx in indices]
            # track.append([index_values.copy(), indices.copy()])
            self.d_embed = np.random.randint(2, min(20, data.shape[1]))
            model = CausalDetection(verbose=False, d_embed=self.d_embed, **self.ccm_kwargs)
            cg = model.fit(data_groups, groupflag=True)

            # model = CausalDetection(verbose=False, d_embed=self.d_embed, **self.ccm_kwargs)
            # cg = model.fit(data_groups)

            if n // self.k < 2:
                np.fill_diagonal(cg, -np.inf)

            ## Find the index of the largest value in the causal graph and return the 
            ## corresponding data group
            top_ind = np.array(np.unravel_index(np.argmax(cg.ravel()), cg.shape))
            new_top_val = float(np.max(cg.ravel()))
            
            ## Update the top links
            if not np.isnan(new_top_val):

                idx2 = itertools.product(index_values_groups[top_ind[0]], index_values_groups[top_ind[1]])
                idx2 = np.array(list(idx2))

                if self.store_history:
                    ## Update score matrix of top channels to be the maximum of the current and observed CCM values
                    # self.scores[idx2[:, 0], idx2[:, 1]] = np.max(
                    #     np.array([
                    #         self.scores[idx2[:, 0], idx2[:, 1]], 
                    #         new_top_val * np.ones_like(self.scores[idx2[:, 0], idx2[:, 1]]) / len(idx2)
                    #     ]),
                    #     axis=0
                    # )
                    self.scores[idx2[:, 0], idx2[:, 1]] = np.sum(
                        np.array([
                            self.scores[idx2[:, 0], idx2[:, 1]], 
                            new_top_val * np.ones_like(self.scores[idx2[:, 0], idx2[:, 1]]) / len(idx2)
                        ]),
                        axis=0
                    )
                    self.history.append(self.scores.copy())

                self.top_links.update_batch(idx2, np.ones(len(idx2)) * new_top_val / len(idx2))
                # self.top_links.update_batch(idx2, np.ones(len(idx2)) * new_top_val)
                
            data = np.vstack([data_groups[ind].T for ind in top_ind])
            index_values = np.hstack([index_values_groups[ind] for ind in top_ind])

            return self.hierarchical_search(data, index_values=index_values, top_val=new_top_val)


    def iterative_search(self, data, index_values=None, num_iter=100, top_val=1e-16, random_state=None):
        """
        Perform an iterative search for causal links in the data, by repeatedly
        splitting the data into two groups and performing CCM and averaging the
        results

        Currently dense

        Args:
            data (np.ndarray): The data matrix
            index_values (np.ndarray): The true index values of the data
            num_iter (int): The number of iterations to run
            top_val (float): The current value of the top link
            random_state (int): The random seed to use

        Returns:
            np.ndarray: The reduced data matrix
            np.ndarray: The reduced index values
            float: The top value
        """
        n, m = data.shape 
        self.out = np.zeros((n, n))
        # print(n, m)

        np.random.seed(random_state)
        for i in range(num_iter):
            if self.verbose:
                progress_bar(i, 100, n_bar=20)
            
            # k = min(int(np.floor(n / 2)), k) # Don't split into singletons
            indices = split_indices(n, self.k)
            # print(indices)
            data_groups = [data[idx].T for idx in indices]
            # index_values_groups = [index_values[idx] for idx in indices]
            self.d_embed = np.random.randint(2, min(20, m))
            model = CausalDetection(verbose=False, d_embed=self.d_embed, **self.ccm_kwargs)
            cg = model.fit(data_groups, groupflag=True)
            cg = np.abs(cg)

            # print("tt", np.array(model.all_all_coeff[0]).shape) 
            ## (10, 10, 10, 10)

            ## from cg, create larger matrice using indices
            for i, idx1 in enumerate(indices):
                for j, idx2 in enumerate(indices):
                    all_idx = itertools.product(idx1, idx2)
                    all_idx = np.array(list(all_idx))
                    group_size = data_groups[0].shape[1]
                    # self.out[all_idx[:, 0], all_idx[:, 1]] = 0.5 * cg[i, j] + 0.5 * self.out[all_idx[:, 0], all_idx[:, 1]]
                    ## Use max instead
                    # self.out[all_idx[:, 0], all_idx[:, 1]] = np.maximum(cg[i, j], self.out[all_idx[:, 0], all_idx[:, 1]])
                    self.out[all_idx[:, 0], all_idx[:, 1]] += cg[i, j]

                    #model.self.compute_crossmap_grouped

            # out = 0.5 * out + 0.5 * cg
        return self.out


    def flat_search(self, data, index_values=None, num_iter=100, sweepk=False, random_state=None):
        """
        Perform an iterative search for causal links, by randomly splitting the data
        into groups and performing ensemble CCM within the groups.

        Currently dense

        Args:
            data (np.ndarray): The data matrix
            index_values (np.ndarray): The true index values of the data
            num_iter (int): The number of iterations to run
            sweepk (bool): Whether to sweep over the number of splits
            random_state (int): The random seed to use

        Returns:
            np.ndarray: The reduced data matrix
            np.ndarray: The reduced index values
            float: The top value
        """
        n, m = data.shape 
        self.out = np.zeros((n, n))
        # print(n, m)

        if not sweepk:
            kvals = [self.k]
        else:
            # kvals = range(2, n // 2)
            kvals = np.linspace(2, n // 2, 15).astype(int)
        
        kvals = np.linspace(2, n, 10).astype(int)
        # kvals = range(2, 10)
        # kvals = [100]

        all_cg = list()
        for k in kvals:
            print(k, flush=True)
            np.random.seed(random_state)
            for i in range(num_iter):
                
                if self.verbose:
                    progress_bar(i, 100, n_bar=20)
                
                # k = min(int(np.floor(n / 2)), k) # Don't split into singletons
                indices = split_indices(n, self.k)
                # print(len(indices[0]))
                data_groups = [data[idx].T for idx in indices]
                # self.d_embed = np.random.randint(2, min(20, m))
                self.d_embed = 3
                for idx1, idx2 in np.ndindex(len(indices), len(indices)):

                    ## Ensure equal length
                    min_idx_len = min(len(indices[idx1]), len(indices[idx2]))
                    all_idx = itertools.product(indices[idx1][:min_idx_len], indices[idx2][:min_idx_len])
                    all_idx = np.array(list(all_idx))
                    model = CausalDetection(verbose=False, d_embed=self.d_embed, **self.ccm_kwargs, ensemble=True)
                    # model = CausalDetection(verbose=False, d_embed=self.d_embed, **self.ccm_kwargs, ensemble=False)
                    cg = model.fit(data_groups[idx1][:, :min_idx_len], data_groups[idx2][:, :min_idx_len])
                    cg = np.abs(cg)
                    # print(len(all_idx), cg.shape)
                    # self.out[all_idx[:, 0], all_idx[:, 1]] += cg.ravel()

                    self.out[all_idx[:, 0], all_idx[:, 1]] = np.maximum(self.out[all_idx[:, 0], all_idx[:, 1]], cg.ravel())
                    
                    # for ii, jj in np.ndindex(len(indices[idx1]), len(indices[idx2])):
                    #     all_idx = itertools.product(indices[idx1][ii], indices[idx2][jj])
                    #     all_idx = np.array(list(all_idx))
                    #     self.out[all_idx[:, 0], all_idx[:, 1]] += cg[ii, jj]

                    all_cg.append(self.out.copy())
                    # print(self.out.shape, flush=True)
                self.all_cg = np.array(all_cg)
                    
        return all_cg
        # return self.out

                

                

        #     # index_values_groups = [index_values[idx] for idx in indices]
        #     self.d_embed = np.random.randint(2, min(20, m))
        #     model = CausalDetection(verbose=False, d_embed=self.d_embed, **self.ccm_kwargs)
        #     cg = model.fit(data_groups, groupflag=True)
        #     cg = np.abs(cg)

        #     # print("tt", np.array(model.all_all_coeff[0]).shape) 
        #     ## (10, 10, 10, 10)

        #     ## from cg, create larger matrice using indices
        #     for i, idx1 in enumerate(indices):
        #         for j, idx2 in enumerate(indices):
        #             all_idx = itertools.product(idx1, idx2)
        #             all_idx = np.array(list(all_idx))
        #             group_size = data_groups[0].shape[1]
        #             # self.out[all_idx[:, 0], all_idx[:, 1]] = 0.5 * cg[i, j] + 0.5 * self.out[all_idx[:, 0], all_idx[:, 1]]
        #             ## Use max instead
        #             # self.out[all_idx[:, 0], all_idx[:, 1]] = np.maximum(cg[i, j], self.out[all_idx[:, 0], all_idx[:, 1]])
        #             self.out[all_idx[:, 0], all_idx[:, 1]] += cg[i, j]

        #             #model.self.compute_crossmap_grouped

        #     # out = 0.5 * out + 0.5 * cg
        # return self.out