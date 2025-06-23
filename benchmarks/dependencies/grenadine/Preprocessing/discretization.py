"""
This module allows to discretize gene expression datasets.
It is mostly based on `scikit-learn`_ library.
Different discretization methods are available :
EWD (equal width, uniform), EFD (equal frequency, quantile), kmeans,
`bikmeans`_ (Li et al., 2010).

.. _scikit-learn:
    https://scikit-learn.org

.. _bikmeans:
    https://www.ncbi.nlm.nih.gov/pubmed/20955620
"""


import pandas as pd
import numpy as np
from math import pow
from sklearn.preprocessing import KBinsDiscretizer


def discretize_genexp(data, method, nb_bins=2, axis=0):
    """
    Discretize data into nb_bins intervals, with specified method, along
    specified axis.

    Args:
        data (pandas.DataFrame or pandas.Series): dataset to discretize
        method (str): method used for discretization, amongst: 'kmeans',
            'bikmeans', 'ewd', 'efd'
        nb_bins (int): (default 2) number of intervals in which to discretize
            data
        axis (int): (default 0) indicates if discretization should be done on
            each column (0) or each line (1) of data. Ignore this parameter if
            method is bikmeans

    Returns:
        pandas.DataFrame or pandas.Series: dataframe or series of discretized
        data, depending on the dimension of passed data

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(3, 5),
                                index=["gene1", "gene2", "gene3"],
                                columns=["c1", "c2", "c3", "c4", "c5"])
        >>> data
                     c1        c2        c3        c4        c5
        gene1  1.764052  0.400157  0.978738  2.240893  1.867558
        gene2 -0.977278  0.950088 -0.151357 -0.103219  0.410599
        gene3  0.144044  1.454274  0.761038  0.121675  0.443863
        >>> discr_data = discretize_genexp(data=data, method='efd')
        >>> discr_data
                c1   c2   c3   c4   c5
        gene1  1.0  0.0  1.0  1.0  1.0
        gene2  0.0  1.0  0.0  0.0  0.0
        gene3  1.0  1.0  1.0  1.0  1.0
    """
    # choose a discretization method
    if method == 'kmeans':
        discr = KBinsDiscretizer(n_bins=nb_bins, encode='ordinal', strategy='kmeans',subsample=None)
    elif method == 'ewd':
        discr = KBinsDiscretizer(n_bins=nb_bins, encode='ordinal', strategy='uniform',subsample=None)
    elif method == 'efd':
        discr = KBinsDiscretizer(n_bins=nb_bins, encode='ordinal', strategy='quantile',subsample=None)
    elif method == 'bikmeans':
        # in this case all discretization process is handled in bikmeans
        discr_data = bikmeans_simple(data, nb_bins)
        return discr_data
    else:
        print("Unvalid option. Methods are: 'kmeans','bikmeans','ewd','efd'")
        return
    # apply discretizer to the data
    if len(data.shape) < 2 :
        # if data is 1D
        discr_data = discr.fit_transform(pd.DataFrame(data).values)
        discr_data = discr_data.flatten()
        if type(data) == type(pd.Series()):
            discr_data = pd.Series(discr_data, index = data.index)
            discr_data.name = data.name
        return(discr_data)
    # if data is 2D
    if axis == 1: # discretize along lines
        discr_data = discr.fit_transform(data.transpose().values)
        discr_data = discr_data.transpose()
    elif axis == 0: # discretize along columns
        discr_data = discr.fit_transform(data.values)
    else:
        print("Unvalid axis argument. 0 for columns (default) and 1 for lines.")
        return
    discr_data = pd.DataFrame(discr_data)
    discr_data.index = data.index
    discr_data.columns = data.columns
    discr_data = discr_data.astype('int64')
    return discr_data


def bikmeans_original(data, nb_bins):
    """
    Discretize data into nb_bins intervals, with method `bikmeans`_, from
    the publication by Li et al, 2010.

    Args:
        data (pandas.DataFrame): dataset to discretize
        nb_bins (int): number of intervals in which to discretize data

    Returns:
        pandas.DataFrame: dataframe of discretized data

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(3, 5),
                                index=["gene1", "gene2", "gene3"],
                                columns=["c1", "c2", "c3", "c4", "c5"])
        >>> data
                     c1        c2        c3        c4        c5
        gene1  1.764052  0.400157  0.978738  2.240893  1.867558
        gene2 -0.977278  0.950088 -0.151357 -0.103219  0.410599
        gene3  0.144044  1.454274  0.761038  0.121675  0.443863
        >>> discr_data = bikmeans_original(data=data, nb_bins=2)
        >>> discr_data
                c1   c2   c3   c4   c5
        gene1  1.0  0.0  0.0  1.0  1.0
        gene2  0.0  1.0  0.0  0.0  0.0
        gene3  0.0  1.0  0.0  0.0  0.0
    """
    km = KBinsDiscretizer(n_bins=nb_bins, encode='ordinal', strategy='kmeans',subsample=None)
    # use kmeans along lines an columns of data separately
    discr1 = km.fit_transform(data.transpose().values).transpose()
    discr2 = km.fit_transform(data.values)
    X = discr1*discr2
    # initalize empty discretized data array
    discr_data = np.zeros(X.shape)
    # going through all values in X and filling in discr_data
    for i,j in np.ndindex(X.shape):
        x = X[i,j]
        k = 0
        while (k < nb_bins):
            if x >= pow(k,2) and x < pow(k+1, 2):
                # discretized value for data[i][j] is k
                discr_data[i][j] = k
                break
            else:
                k += 1
        if k == nb_bins: # error case : no suitable bin was found for x
            discr_data[i][j] = -1
    # cast discretized data into a pandas DataFrame
    discr_data = pd.DataFrame(discr_data)
    discr_data.index = data.index
    discr_data.columns = data.columns
    return discr_data

def bikmeans_simple(data, nb_bins):
    """
    Discretize data into nb_bins intervals, with method `bikmeans`_, simplified.
    From the publication by Li et al, 2010.
    See function bikmeans_original() for the full implementation of bikmeans as
    described in the paper.

    Args:
        data (pandas.DataFrame): dataset to discretize
        nb_bins (int): number of intervals in which to discretize data

    Returns:
        pandas.DataFrame: dataframe of discretized data

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(3, 5),
                                index=["gene1", "gene2", "gene3"],
                                columns=["c1", "c2", "c3", "c4", "c5"])
        >>> data
                     c1        c2        c3        c4        c5
        gene1  1.764052  0.400157  0.978738  2.240893  1.867558
        gene2 -0.977278  0.950088 -0.151357 -0.103219  0.410599
        gene3  0.144044  1.454274  0.761038  0.121675  0.443863
        >>> discr_data = bikmeans_simple(data=data, nb_bins=2)
        >>> discr_data
                c1   c2   c3   c4   c5
        gene1  2.0  1.0  1.0  2.0  2.0
        gene2  1.0  2.0  1.0  1.0  1.0
        gene3  1.0  2.0  1.0  1.0  1.0
    """

    km = KBinsDiscretizer(nb_bins, encode='ordinal', strategy='kmeans')
    # use kmeans along lines an columns of data separately
    discr1 = km.fit_transform(data.transpose().values).transpose()
    discr2 = km.fit_transform(data.values)
    X = (discr1+1)*(discr2+1)
    discr_data = np.floor(np.sqrt(X))
    # cast discretized data into a pandas DataFrame
    discr_data = pd.DataFrame(discr_data)
    discr_data.index = data.index
    discr_data.columns = data.columns
    return discr_data
