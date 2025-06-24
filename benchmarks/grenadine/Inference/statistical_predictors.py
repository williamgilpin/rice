"""
This module allows to infer co-expression Gene Regulatory Networks using
gene expression data (RNAseq or Microarray). This module implements severall
inference algorithms based on statistical predictors, using `scipy-stats`_ and
`scikit-learn`_.

.. _scipy-stats:
    https://docs.scipy.org/doc/scipy/reference/stats.html
.. _scikit-learn:
    https://scikit-learn.org
"""

from sklearn.feature_selection import f_regression as _sklearn_f_regression
from sklearn.feature_selection import mutual_info_regression as _sklearn_mutual_info_regression
from scipy.stats import spearmanr as _scipy_spearmanr
from scipy.stats import pearsonr as _scipy_pearsonr
from scipy.stats import wilcoxon as _scipy_wilcoxon
from scipy.stats import mannwhitneyu as _scipy_mannwhitneyu
from scipy.stats import theilslopes as _scipy_theilslopes
from scipy.stats import kendalltau as _scipy_kendalltau
from scipy.stats import rankdata as _scipy_rankdata
from scipy.stats import energy_distance as _scipy_energy_distance
from scipy.stats import wasserstein_distance as _scipy_wasserstein_distance
import numpy as np


def abs_pearsonr_coef(X,y):
    """
    Score predictor function based on the `scipy-stats`_
    absolute Pearson correlation.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the absolute value of the
        correlation between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = abs_pearsonr_coef(tfs,tg)
        >>> scores
        array([0.41724166, 0.02212467, 0.23708491])
    """
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = _scipy_pearsonr(y,x_tf)[0]
    scores = np.abs(scores)
    return(scores)

def abs_spearmanr_coef(X,y):
    """
    Score predictor function based on the `scipy-stats`_
    absolute Spearman correlation.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the absolute value of the
        correlation between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = abs_spearmanr_coef(tfs,tg)
        >>> scores
        array([0.5, 0.3, 0.3])
    """
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = _scipy_spearmanr(y,x_tf)[0]
    scores = np.abs(scores)
    return(scores)

def kendalltau_score(X,y,**kendalltau_parameters):
    """
    Score predictor function based on the `scipy-stats`_
    Kendall’s tau correlation measure.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **kendalltau_parameters: Named parameters for the scipy-stats kendall's
            tau correlation measure

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score of the
        score between target gene expression and the i-th transcription factor
        gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = kendalltau_score(tfs,tg)
        >>> scores
        array([0.8487997 , 1.30065214, 0.20467198])s
    """
    epsilon = 1e-300
    scores = np.zeros(X.shape[1])
    y = _scipy_rankdata(y)
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        x_tf_ranks = _scipy_rankdata(x_tf)
        scores[i] = _scipy_kendalltau(y,x_tf_ranks,**kendalltau_parameters)[1]
        scores[i] = -np.log10(scores[i]+epsilon)
    return(scores)

def f_regression_score(X,y):
    """
    Score predictor function based on the `scikit-learn`_ f_regression score.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score of the
        f_regression linear test between target gene expression and the
        i-th transcription factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = f_regression_score(tfs,tg)
        >>> scores
        array([0.63235967, 0.00146922, 0.17867071])
    """
    scores, p_values = _sklearn_f_regression(X, y, center=True)
    return(scores)

def CLR(X,y,**mi_parameters):
    """
    Score predictor function based on `scikit-learn`_ mutual_info_regression
    score.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **mi_parameters: Named parameters for sklearn mutual_info_regression

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score of the
        sklearn mutual_info_regression computation between target gene
        expression and the i-th transcription factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = CLR(tfs,tg)
        >>> scores
        array([6.66666667e-02, 1.16666667e-01, 2.22044605e-16])
    """
    scores = _sklearn_mutual_info_regression(X,y,**mi_parameters)
    return(scores)

def wilcoxon_score(X,y,**wilcoxon_parameters):
    """
    Score predictor function based on the `scipy-stats`_
    Wilcoxon signed-rank test.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **wilcoxon_parameters: Named parameters for the scipy-stats Wilcoxon
            signed-rank test

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score
        between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),index =["c1","c2","c3","c4","c5"],columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = wilcoxon_score(tfs,tg)
        >>> scores
        array([1.36537718, 0.64797987, 0.30086998])
    """
    epsilon = 1e-300
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = _scipy_wilcoxon(y,x_tf,**wilcoxon_parameters)[1]
        scores[i] = -np.log10(scores[i]+epsilon)
    return(scores)

def mannwhitneyu_score(X,y,**mannwhitneyu_parameters):
    """
    Score predictor function based on the `scipy-stats`_ Mann-Whitney rank test.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **mannwhitneyu_parameters: Named parameters for the scipy-stats
            Mann-Whitney rank test

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score
        between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = mannwhitneyu_score(tfs,tg)
        >>> scores
        array([1.52213525, 0.47101693, 0.3795872 ])
    """
    epsilon = 1e-300
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = _scipy_mannwhitneyu(y,x_tf,**mannwhitneyu_parameters)[1]
        scores[i] = -np.log10(scores[i]+epsilon)
    return(scores)

def theilslopes_score(X,y,**theilslopes_parameters):
    """
    Score predictor function based on the `scipy-stats`_
    Theil-Sen robust slope estimator.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **theilslopes_parameters: Named parameters for the scipy-stats
            Theil-Sen robust slope estimator

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score
        between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = theilslopes_score(tfs,tg)
        >>> scores
        array([0.92309299, 0.90933202, 0.26451817])
    """
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = np.abs(_scipy_theilslopes(y,x_tf,**theilslopes_parameters)[0])
    return(scores)


def energy_distance_score(X,y,**energy_distance_parameters):
    """
    Score predictor function based on the `scipy-stats`_ energy distance between
    1D distributions.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **energy_distance_parameters: Named parameters for the scipy-stats
            energy distance

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score
        between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = energy_distance_score(tfs,tg)
        >>> scores
        array([0.40613705, 0.6881455 , 0.72786711])
    """
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = np.exp(-_scipy_energy_distance(y,x_tf,**energy_distance_parameters))
    return(scores)

def wasserstein_distance_score(X,y,**wasserstein_distance_parameters):
    """
    Score predictor function based on the `scipy-stats`_ Wasserstein distance
    between 1D distributions.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **wasserstein_distance_parameters: Named parameters for the scipy-stats
            Wasserstein distance

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score
        between target gene expression and the i-th transcription
        factor gene expression.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randn(5),index=["c1","c2","c3","c4","c5"])
        >>> scores = wasserstein_distance_score(tfs,tg)
        >>> scores
        array([0.36457586, 0.72057084, 0.81207932])
    """
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x_tf = X[:,i]
        scores[i] = np.exp(-_scipy_wasserstein_distance(y,x_tf,**wasserstein_distance_parameters))
    return(scores)
