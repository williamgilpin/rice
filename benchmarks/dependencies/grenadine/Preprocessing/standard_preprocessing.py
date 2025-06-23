# -*- coding: utf-8 -*-
"""
This module allows to pre-process gene expression data.
"""
import numpy as np
import pandas as pd
try:
    import ot

except:
    print("Warning: could not import POT. columns_matrix_OT_norm() function cannot be called.")


def z_score(A, axis=0):
    """
    Compute the z-score along the specified axis.

    Args:
        A (pandas.DataFrame or numpy.array): matrix
        axis (int): 0 for columns and 1 for rows

    Returns:
        pandas.DataFrame or numpy.array: Normalized matrix

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["c1", "c2", "c3", "c4", "c5"],
                            columns=["gene1", "gene2", "gene3", "gene4", "gene5"])
        >>> norm_data = z_score(data)
        >>> norm_data
               gene1     gene2     gene3     gene4     gene5
        c1  1.254757 -1.222682  0.914682  1.672581  0.828015
        c2 -0.446591 -0.083589 -1.038607 -0.418644 -0.331945
        c3  0.249333  0.960749  0.538403 -0.218012 -0.305461
        c4  0.367024  1.043200 -1.131598 -0.047267 -1.338834
        c5 -1.424523 -0.697678  0.717120 -0.988659  1.148225

    """
    epsilon = 1e-5
    if axis==0:
        A = (A - A.mean(axis=0)) / (A.std(axis=0) + epsilon)
    if axis==1:
        A = A.T
        A = (A - A.mean(axis=0)) / (A.std(axis=0) + epsilon)
        A = A.T
    return(A)

def mean_std_polishing(A, nb_iterations=5):
    """
    Iterative z-score on rows and columns.

    Args:
        A (pandas.DataFrame or numpy.array): matrix
        nb_iterations (int): number of polishing iterations

    Returns:
        pandas.DataFrame or numpy.array: Polished matrix

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["c1", "c2", "c3", "c4", "c5"],
                            columns=["gene1", "gene2", "gene3", "gene4", "gene5"])
        >>> norm_data = mean_std_polishing(data)
        >>> norm_data
               gene1     gene2     gene3     gene4     gene5
        c1  0.336095 -1.618781  0.187436  1.109617 -0.014367
        c2 -0.321684  0.586608 -1.606905  0.484159  0.857821
        c3  0.139260  0.860934  0.976541 -1.395814 -0.580921
        c4  1.243263  0.421752 -0.585940  0.282319 -1.361394
        c5 -1.363323 -0.161066  0.826375 -0.421998  1.120013

    """
    epsilon = 1e-10
    for i in range(nb_iterations):
        A = A-A.mean(axis=0)
        # std polish the column
        A = A/(A.std(axis=0)+epsilon)
        # mean polish the row
        A = (A.T-A.T.mean(axis=0)).T
        # std polish the row
        A = (A.T/(A.T.std(axis=0) + epsilon)).T
    return(A)

def cat_gene_expression_dfs(gene_expression_dfs):
    """
    Concatenate different gene expression datasets, based on gene id (rows).

    Args:
        gene_expression_dfs (list of pandas.DataFrame): Expression datasets list

    Returns:
        pandas.DataFrame: concatenated gene expression datasets

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data1 = pd.DataFrame(np.random.randn(3, 3),
                            index=["gene1", "gene2", "gene3"],
                            columns=["c1", "c2", "c3"])
        >>> data1
                     c1        c2        c3
        gene1  1.764052  0.400157  0.978738
        gene2  2.240893  1.867558 -0.977278
        gene3  0.950088 -0.151357 -0.103219
        >>> data2 = pd.DataFrame(np.random.randn(3, 3),
                            index=["gene2", "gene3", "gene4"],
                            columns=["c4", "c5", "c6"])
        >>> data2
                     c4        c5        c6
        gene2  0.410599  0.144044  1.454274
        gene3  0.761038  0.121675  0.443863
        gene4  0.333674  1.494079 -0.205158
        >>> data=cat_gene_expression_dfs([data1, data2])
        >>> data
                     c1        c2        c3        c4        c5        c6
        gene1  1.764052  0.400157  0.978738       NaN       NaN       NaN
        gene2  2.240893  1.867558 -0.977278  0.410599  0.144044  1.454274
        gene3  0.950088 -0.151357 -0.103219  0.761038  0.121675  0.443863
        gene4       NaN       NaN       NaN  0.333674  1.494079 -0.205158

    """
    cat = pd.concat(gene_expression_dfs,axis=1,sort=True)
    return(cat)

def median_outliers_filter(X, threshold=3):
    """
    Ensures that all the values of data_set are within:
    :math:`median(X) \pm \\tau \\times MAD(X))`

    Args:
        X (pandas.DataFrame or numpy.array): gene expression matrix (for instance)
        threshold (float): :math:`\\tau` threshold

    Returns:
        pandas.DataFrame or numpy.array: X without outliers (outliers set to
        the extreme values allowed)

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["c1", "c2", "c3", "c4", "c5"],
                            columns=["gene1", "gene2", "gene3", "gene4", "gene5"])
        >>> median_outliers_filter(data)
               gene1     gene2     gene3     gene4     gene5
        c1  1.764052  0.400157  0.978738  0.674682  1.867558
        c2 -0.977278  0.950088 -0.653101 -0.103219  0.410599
        c3  0.144044  1.454274  0.761038  0.121675  0.443863
        c4  0.333674  1.494079 -0.653101  0.313068 -0.854096
        c5 -2.552990  0.653619  0.864436 -0.674682  2.269755

    """
    difference = np.abs(X-np.median(X,axis=0))
    median_difference = np.median(difference,axis=0)
    s = difference / median_difference
    mask = s > threshold
    mask_negative = X < np.median(X,axis=0)
    mask_positive = X > np.median(X,axis=0)
    X[np.logical_and(mask,mask_negative)] = -threshold*np.ones(X.shape)*median_difference
    X[np.logical_and(mask,mask_positive)] = threshold*np.ones(X.shape)*median_difference
    return(X)

def columns_matrix_OT_norm(X,reference=None,bins=None,**SinkhornTransport_para):
    """
    Use optimal transport in order to make all conditions disributions alike.

    Args:
        X (pandas.DataFrame): gene expression matrix
        r_percentile (numpy.array): reference distribution
        bins (numpy.array): bins for percentiles computation
        SinkhornTransport_para: ot.da.SinkhornTransport parameters

    Returns:
        pandas.DataFrame: Normalized matrix

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> a = pd.DataFrame(np.random.randn(10000,10))
        >>> b = pd.DataFrame(np.random.randn(10000,10)*3+4)
        >>> bins = list(range(1,100))
        >>> b_ = columns_matrix_OT_norm(b,a.iloc[:,0],bins,reg_e=5e-1)
    """
    if bins is None:
        bins = [0.01]+list(np.arange(0.1,100,0.1))+[99.99]
    if reference is None:
        reference = X.values[:,0]
    percentile = np.percentile(reference,bins)
    X_c = pd.DataFrame()
    if "reg_e" not in SinkhornTransport_para:
        SinkhornTransport_para["reg_e"] = 1e-3
    for c in X.columns:
        percentile_c = np.percentile(X[c].values.flatten(),bins)
        ot_sinkhorn = ot.da.SinkhornTransport(**SinkhornTransport_para)
        ot_sinkhorn.fit(Xs=percentile_c.reshape(-1,1),
                        Xt=percentile.reshape(-1,1))
        X_c_transf = ot_sinkhorn.transform(Xs=X[c].values.reshape(-1,1))
        X_c[c] = X_c_transf.flatten()
    X_c.index = X.index
    return(X_c)
