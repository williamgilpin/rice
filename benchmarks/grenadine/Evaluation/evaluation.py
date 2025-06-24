# -*- coding: utf-8 -*-
"""
This module allows to evaluate, compare and aggregate putative Gene Regulatory
Networks, using `scikit-learn`_.

.. _scikit-learn:
    https://scikit-learn.org
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx
from grenadine.Inference.inference import rank_GRN,clean_nan_inf_scores
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

__author__ = "Sergio Peignier, Pauline Schmitt"
__copyright__ = "Copyright 2019, The GReNaDIne Project"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Sergio Peignier"
__email__ = "sergio.peignier@insa-lyon.fr"
__status__ = "pre-alpha"


def jaccard_similarity(joined_ranks, k):
    """
    Compute the Jaccard similarity between GRN inference methods.

    Args:
        joined_ranks (pandas.DataFrame): joined ranks for different methods,
            where rows represent possible regulatory links, and columns
            represent each method. The value at row i and column j represents
            the rank or the score of edge i computed by method j.
        k (int): Top k number of top ranked links to be compared

    Returns:
        pandas.DataFrame: Square matrix with GRNI methods Jaccard similaties
            Value at row i and column j represents the Jaccard similarity
            between the top k edges found by methods i and j resp.

    Example:
        >>> import pandas as pd
        >>> joined_ranks = pd.DataFrame([[1, 1, 2],[2, 2, 3],[3, 3, 1]],
                                        columns=['method1','method2','method3'],
                                        index=['gene1_gene2',
                                               'gene1_gene3',
                                               'gene3_gene2'])
        >>> joined_ranks
                     method1  method2  method3
        gene1_gene2        1        1        2
        gene1_gene3        2        2        3
        gene3_gene2        3        3        1
        >>> similarity = jaccard_similarity(joined_ranks,2)
        >>> similarity
                  method1   method2   method3
        method1       NaN         1  0.333333
        method2         1       NaN  0.333333
        method3  0.333333  0.333333       NaN

    """
    jaccard_sim = pd.DataFrame(columns=joined_ranks.columns,
                               index=joined_ranks.columns)
    methods = list(jaccard_sim.columns)
    for i,c in enumerate(tqdm(methods[:-1])):
        top_k_c = list(joined_ranks[c].nsmallest(k).index)
        for j,b in enumerate(methods[i+1:]):
            top_k_i = list(joined_ranks[b].nsmallest(k).index)
            inter = len(set(top_k_i).intersection(set(top_k_c)))
            union = len(set(top_k_i).union(set(top_k_c)))
            jaccard_sim.loc[b,c] = inter/union
            jaccard_sim.loc[c,b] = jaccard_sim.loc[b,c]
    return(jaccard_sim)

def get_top_k_edges(method_ranks,k):
    """
    Return the top k edges for a given method.

    Args:
        method_ranks (pandas.Series): edges ranks for a given method, where each
            element represents the rank of a given edge
        k (int): top k edges to select

    Returns:
        list: list of the top k edges

    Example:
        >>> import pandas as pd
        >>> method_ranks = pd.Series([1,4,2,3], index=['gene1_gene2',
                                                       'gene1_gene3',
                                                       'gene3_gene1',
                                                       'gene3_gene2'])
        >>> edges = get_top_k_edges(method_ranks,2)
        >>> edges
        ['gene1_gene2', 'gene3_gene1']

    """
    return list(method_ranks.nsmallest(k).index)[:k]

def get_top_tfs_per_tg_from_scores(scores,k):
    """
    Return the top k tfs per tg, from a score matrix.

    Args:
        scores (pandas.DataFrame): scores assigend to each tf (columns)
            for each tg (rows)
        k (int): top k tfs to select

    Returns:
        pandas.DataFrame: top tfs per tg

    """
    top_tfs = pd.DataFrame(scores.columns.values[np.argsort(-scores.values, axis=1)[:, :k]],
                          index=scores.index,
                          columns = ["TF "+str(i) for i in range(k)])
    return top_tfs

def get_top_k_edges_per_node(method_ranks,k,tg=1):
    """
    Return the top k edges per node.

    Args:
        method_ranks (pandas.Series): edges ranks for a given method, where each
            element represents the rank of a given edge
        k (int): top k edges to select
        tg (bool): True for TGs and False for TFs

    Returns:
        list: list of top k edges

    Example:
        >>> import pandas as pd
        >>> method_ranks = pd.Series([1,4,2,3], index=['gene1_gene2',
                                                       'gene1_gene3',
                                                       'gene3_gene1',
                                                       'gene3_gene2'])
        >>> # Top 1 edges per target gene
        >>> get_top_k_edges_per_node(method_ranks,1)
        ['gene1_gene2', 'gene1_gene3', 'gene3_gene1']
        >>> # Top 1 edges per transcription factor
        >>> get_top_k_edges_per_node(method_ranks,1,tg=0)
        ['gene1_gene2', 'gene3_gene1']

    """
    get_tg = lambda x: x.split("_")[int(tg)]
    tg = method_ranks.index.map(get_tg)
    gb = method_ranks.groupby(tg)
    top_k_edges_per_tg_list = gb.apply(get_top_k_edges,k)
    top_k_edges_per_tg = []
    for g in tg:
        top_k_edges_per_tg += top_k_edges_per_tg_list[g]
    top_k_edges_per_tg = np.unique(top_k_edges_per_tg)
    return list(top_k_edges_per_tg)


def union_top_k_edges(joined_ranks,
                      k,
                      method_selection=get_top_k_edges,
                      **method_selection_args):
    """
    Compute the top k edges found by different GRN inference methods.

    Args:
        joined_ranks (pandas.DataFrame): joined ranks for different methods,
            where rows represent possible regulatory links, and columns
            represent each method. The value at row i and column j represents
            the rank or the score of edge i computed by method j.
        k (int): Top k number of top ranked links to be compared
        method_selection (function): Method used to select top k edges for
            each algorithm (e.g., top k, top k for each tg, top k for each tf)
            this function should receive as parameters a pandas.Series of ranks
            and k.
    Returns:
        list: union of the top k edges of the different methods

    Example:
        >>> import pandas as pd
        >>> from grenadine.Evaluation.evaluation import get_top_k_edges
        >>> joined_ranks = pd.DataFrame([[1,1,2],[2,2,3],[4,3,1],[3,4,4]],
                                        columns=['method1',
                                                 'method2',
                                                 'method3'],
                                        index=['gene1_gene2',
                                               'gene1_gene3',
                                               'gene3_gene2',
                                               'gene3_gene1'])
        >>> joined_ranks
                     method1  method2  method3
        gene1_gene2        1        1        2
        gene1_gene3        2        2        3
        gene3_gene2        4        3        1
        gene3_gene1        3        4        4
        >>> union_top_k_edges(joined_ranks, k=2, method_selection=get_top_k_edges)
        ['gene1_gene3', 'gene1_gene2', 'gene3_gene2']
        >>> union_top_k_edges(joined_ranks, k=1, method_selection=get_top_k_edges)
        ['gene1_gene2', 'gene3_gene2']

    """
    union_top_k = set()
    for i,c in enumerate(joined_ranks):
        top_k_c = method_selection(joined_ranks[c],k,**method_selection_args)
        union_top_k = union_top_k.union(set(top_k_c))
    return(list(union_top_k))

def score_top_k_edges(joined_ranks,
                      k,
                      method_selection=get_top_k_edges,
                      **method_selection_args):
    """
    Compute the number of methods that find each edge.

    Args:
        joined_ranks (pandas.DataFrame): joined ranks for different methods,
            where rows represent possible regulatory links, and columns
            represent each method. The value at row i and column j represents
            the rank or the score of edge i computed by method j.
        k (int): Top k number of top ranked links to be compared
        method_selection (function): Method used to select top k edges for
            each algorithm (e.g., top k, top k for each tg, top k for each tf)
            this function should receive as parameters a pandas.Series of ranks
            and k.
    Returns:
        pandas.Series: number of methods having detected each edge

    Example:
        >>> import pandas as pd
        >>> from grenadine.Evaluation.evaluation import get_top_k_edges
        >>> joined_ranks = pd.DataFrame([[1,1,2],[2,2,3],[4,3,1],[3,4,4]],
                                        columns=['method1',
                                                 'method2',
                                                 'method3'],
                                        index=['gene1_gene2',
                                               'gene1_gene3',
                                               'gene3_gene2',
                                               'gene3_gene1'])
        >>> score_top_k_edges(joined_ranks, k=2, method_selection=get_top_k_edges)
        gene1_gene2    3
        gene1_gene3    2
        gene3_gene2    1
        gene3_gene1    0
        >>> score_top_k_edges(joined_ranks, k=1, method_selection=get_top_k_edges)
        gene1_gene2    2
        gene1_gene3    0
        gene3_gene2    1
        gene3_gene1    0

    """
    union_top_k = pd.Series(0,index = joined_ranks.index)
    for i,c in enumerate(joined_ranks):
        top_k_c = method_selection(joined_ranks[c],k,**method_selection_args)
        union_top_k[top_k_c] += 1
    return(union_top_k)

def pca_representation(joined_ranks,k,**pca_parameters):
    """
    Map the method to the space of the union of top k edges and apply PCA.

    Args:
        joined_ranks (pandas.DataFrame): joined ranks for different methods,
            where rows represent possible regulatory links, and columns
            represent each method. The value at row i and column j represents
            the rank or the score of edge i computed by method j.
        k (int): Top k number of top ranked links to be compared
        pca_parameters: Named parameter for the sklearn PCA method

    Returns:
        pandas.DataFrame: methods coordinates along principal components, where
        rows represent methods and columns representt Principal Components.
        sklearn.decomposition.PCA: sklearn pca object

    Example:
        >>> import pandas as pd
        >>> joined_ranks = pd.DataFrame([[1,1,2],[2,2,3],[4,3,1],[3,4,4]],
                                        columns=['method1',
                                                 'method2',
                                                 'method3'],
                                        index=['gene1_gene2',
                                               'gene1_gene3',
                                               'gene3_gene2',
                                               'gene3_gene1'])
        >>> pca_X, pca = pca_representation(joined_ranks, k=2)
        [9.81125224e-01 1.88747757e-02 1.60366056e-32]
        >>> pca_X
                        0         1             2
        method1 -1.400804 -0.194292  1.790900e-16
        method2 -0.512730  0.265408  1.790900e-16
        method3  1.913533 -0.071116  1.790900e-16

    """
    # Select only edges that are in the top k of at least one method
    edges = union_top_k_edges(joined_ranks,k)
    rankings = joined_ranks.loc[edges]
    # transpose the rankings
    rankingsT = rankings.T
    # create PCA object
    pca = PCA(**pca_parameters)
    # Apply the PCA
    rankingsT_pca = pca.fit_transform(rankingsT)
    rankingsT_pca = pd.DataFrame(rankingsT_pca)
    rankingsT_pca.index = rankingsT.index
    return(rankingsT_pca,pca)

def edges2boolvec(total_edges, chosen_edges):
    """
    Build a boolean vector from a list of edges.

    Args:
        total_edges (list or numpy.array): total list of edges to be considered.
            Each element of the output boolean vector represents each link from
            this list
        chosen_edges (list or numpy.array): list of edges to be labeled as 1s

    Returns:
        pandas.Series: boolean list representing edges received as parameters.
            Each element of the output boolean vector represents each link from
            total_edges list, and the corresponding value is equal to 1 if the
            edge is also in chosen_edges and 0 otherwise

    Example:
        >>> edges = ['gene1_gene2', 'gene1_gene3', 'gene3_gene2', 'gene3_gene1']
        >>> chosen_edges = ['gene1_gene2','gene3_gene1']
        >>> edges2boolvec(edges, chosen_edges)
        gene1_gene2    1
        gene1_gene3    0
        gene3_gene2    0
        gene3_gene1    1

    """
    #print(np.asarray([e in total_edges for e in chosen_edges]).all())
    y = pd.Series(0,index=total_edges)
    y[chosen_edges] = 1
    return(y)

def KneeLocator(fpr, tpr):
    """
    Find the optimal ROC curve point (closest to (0,1)).

    Args:
        fpr (numpy.array): false positive rate array
        tpr (numpy.array): true positive rate array

    Returns:
        int: optimal point index

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(0)
        >>> fpr = np.sort(np.random.rand(10))
        >>> np.random.seed(1)
        >>> tpr = np.sort(np.random.rand(10))
        >>> index = KneeLocator(fpr, tpr)
        >>> print("Optimal ROC point is ( fpr =", fpr[index], "; tpr =", tpr[index], ")")
        Optimal ROC point is ( fpr = 0.3834415188257777 ; tpr = 0.538816734003357 )

    """
    distances = np.sqrt((fpr**2 + (tpr-1)**2))
    i = np.argmin(distances)
    return i

def get_y_targets(gold_std_grn, scores=None, ranks=None, n_links=100000):
    """
    Get ground truth and predictor's estimation of a GRN, in a format ready to
    be used in sklearn.metrics functions.

    Args:
        scores (pandas.DataFrame): co-expression score matrix, where rows are
            target genes and columns are transcription factors.
            The value at row i and column j represents the score assigned by a
            score_predictor to the regulatory relationship between target gene i
            and transcription factor j.
        ranks (pandas.DataFrame): ranking matrix. A ranking matrix contains a
            row for each possible regulatory link, it also contains 4 columns,
            namely the rank, the score, the transcription factor id, and the
            target gene id.
        gold_std_grn (pandas.DataFrame): reference GRN used as a gold standard,
            where rows are links with index of the type <TF symbol>_<TG symbol>,
            and columns are respectively 'TF':TF symbol, 'TG':TG symbol, and
            'IS_REGULATED': indicates whether the link exists (1) or not (0)
        n_links (int): number of highest scores to keep from the
            estimated scores matrix

    Returns:
        (pandas.Series, pandas.Series, pandas.Series): tuple containing:

             (y_true, y_pred, y_pred_binary). y_true, boolean values for the
             ground truth (correct target values) of the GRN.
             y_pred, continuous values of the estimated GRN.
             y_pred_binary, boolean values of the estimated GRN, found by
             computing the argmin of the euclidean distance to the optimal roc
             curve point (0,1)

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> scores = pd.DataFrame(np.random.randn(3, 2),
                                  index=["gene1", "gene2", "gene3"],
                                  columns=["gene1", "gene3"])
        >>> # scores associated to self loops are set to nan
        >>> scores.iloc[0,0]=np.nan
        >>> scores.iloc[2,1]=np.nan
        >>> scores
                  gene1     gene3
        gene1       NaN  0.400157
        gene2  0.978738  2.240893
        gene3  1.867558       NaN
        >>> grn = pd.DataFrame(np.array([['gene1', 'gene2', 1],
                                         ['gene1', 'gene3', 0],
                                         ['gene3', 'gene2', 1],
                                         ['gene3', 'gene1', 0]]),
                                         columns=['TF', 'TG', 'IS_REGULATED'])
        >>> grn.index=grn['TF']+'_'+grn['TG']
        >>> grn["IS_REGULATED"] = grn["IS_REGULATED"].astype(int)
        >>> grn
                        TF     TG  IS_REGULATED
        gene1_gene2  gene1  gene2             1
        gene1_gene3  gene1  gene3             0
        gene3_gene2  gene3  gene2             1
        gene3_gene1  gene3  gene1             0
        >>> y_true, y_pred, y_pred_binary = get_y_targets(scores=scores, grn, n_links=3)
        >>> y_true
        gene1_gene3    0
        gene1_gene2    1
        gene3_gene2    1
        >>> y_pred
        gene1_gene3    1.867558
        gene1_gene2    0.978738
        gene3_gene2    2.240893
        >>> y_pred_binary
        gene1_gene3    0
        gene1_gene2    0
        gene3_gene2    1

    """
    if scores is None and ranks is None:
        return None
    if scores is not None and ranks is None:
        scores = clean_nan_inf_scores(scores) 
        # print("Passed array:", np.mean(scores.values))
        ranks = rank_GRN(coexpression_scores_matrix=scores,take_abs_score=False)
    if(len(ranks.shape)==1):
        ranks = pd.DataFrame(ranks,columns=["rank"])# the best method is ranked 1
        ranks["score"] = ranks["rank"].max() - ranks["rank"]
    ranks_top = ranks.iloc[:n_links,:] # taking only n_links best links
    mutual_edges = set(ranks_top.index).intersection(set(gold_std_grn.index)) # taking links that are both in ranks_top and in the golden standard
    if not len(mutual_edges):
        print("Warning: no common edges between gold standard and top-k predicted links")
        return(None,None,None)
    # Amongst n_links best links, we select the ones that are in the golden standard
    ranks_top_in_golden = ranks_top.loc[list(mutual_edges)]
    # Amongst links in the golden standard, we select the ones that appear in the n_links best links
    golden = gold_std_grn.loc[list(mutual_edges)]

    y_pred = ranks_top_in_golden["score"]
    y_true = golden["IS_REGULATED"]
    # print(np.unique(y_true))
    y_true = y_true.astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    i = KneeLocator(fpr, tpr)
    thr = thresholds[i]
    y_pred_binary = np.zeros(y_pred.shape)
    y_pred_binary[y_pred >= thr] = 1
    y_pred_binary = pd.Series(y_pred_binary.astype(int))
    y_pred_binary.index = y_pred.index
    return y_true, y_pred, y_pred_binary


def evaluate_result(scores, gold_std_grn, n_links=100000):
    """
    Evaluate the performance of a GRN predictor based on the estimated scores,
    compared with the gold standard. Uses metrics from `scikit-learn`_.

    Args:
        scores (pandas.DataFrame): co-expression score matrix, where rows are
            target genes and columns are transcription factors.
            The value at row i and column j represents the score assigned by a
            score_predictor to the regulatory relationship between target gene i
            and transcription factor j.
        gold_std_grn (pandas.DataFrame): reference GRN used as a gold standard,
            where rows are links with index of the type <TF symbol>_<TG symbol>,
            and columns are respectively 'TF':TF symbol, 'TG':TG symbol, and
            'IS_REGULATED': indicates whether the link exists (1) or not (0)
        n_links (int): number of highest scores to keep from the estimated
            scores matrix

    Returns:
        pandas.Series: values of evaluation metrics for the estimated GRN

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> scores = pd.DataFrame(np.random.randn(3, 2),
                                  index=["gene1", "gene2", "gene3"],
                                  columns=["gene1", "gene3"])
        >>> # scores associated to self loops are set to nan
        >>> scores.iloc[0,0]=np.nan
        >>> scores.iloc[2,1]=np.nan
        >>> scores
                  gene1     gene3
        gene1       NaN  0.400157
        gene2  0.978738  2.240893
        gene3  1.867558       NaN
        >>> grn = pd.DataFrame(np.array([['gene1', 'gene2', 1],
                                         ['gene1', 'gene3', 0],
                                         ['gene3', 'gene2', 1],
                                         ['gene3', 'gene1', 0]]),
                                         columns=['TF', 'TG', 'IS_REGULATED'])
        >>> grn.index=grn['TF']+'_'+grn['TG']
        >>> grn["IS_REGULATED"] = grn["IS_REGULATED"].astype(int)
        >>> grn
                        TF     TG  IS_REGULATED
        gene1_gene2  gene1  gene2             1
        gene1_gene3  gene1  gene3             0
        gene3_gene2  gene3  gene2             1
        gene3_gene1  gene3  gene1             0
        >>> metrics = evaluate_result(scores, grn, n_links=3)
        >>> metrics
        AUROC        0.500000
        AUPR         0.791667
        Precision    1.000000
        Recall       0.500000
        Accuracy     0.666667
        F1           0.666667

    """

    y_true, y_pred, y_pred_binary = get_y_targets(gold_std_grn,
                                                  scores=scores,
                                                  n_links=n_links)
    classes = np.unique(y_true)
    if len(classes) == 1:
        if classes[0] is None:
            metrics_dict = {"AUROC": 0,
                            "AUPR": 0,
                            "Precision": 0,
                            "Recall": 0,
                            "Accuracy": 0,
                            "F1": 0}
        if classes[0] == 0:
            metrics_dict = {"AUROC": 0,
                            "AUPR": 0,
                            "Precision": 0,
                            "Recall": 0,
                            "Accuracy": 0,
                            "F1": 0}
        if classes[0] == 1:
            metrics_dict = {"AUROC": 1,
                            "AUPR": 1,
                            "Precision": 1,
                            "Recall": 1,
                            "Accuracy": 1,
                            "F1": 1}
    else:
        metrics_dict = {"AUROC": roc_auc_score(y_true,y_pred),
                        "AUPR": pr_auc_score(y_true,y_pred),
                        "Precision": precision_score(y_true, y_pred_binary),
                        "Recall": recall_score(y_true, y_pred_binary),
                        "Accuracy": accuracy_score(y_true, y_pred_binary),
                        "F1": f1_score(y_true, y_pred_binary)}
    return pd.Series(metrics_dict)


def fit_beta_pdf(raw_values,xmin=0,xmax=1,nb_points=int(1e5)):
    """
    Fit a beta function to the given raw values.

    Args:
        raw_values (numpy.array): array of raw values between xmin and xmax
        xmin (int): minimum x value of beta function
        xmax (int): maximum x value of beta function
        nb_points (int): number of x points of beta function

    Returns:
        numpy.array: X values of fitted pdf
        numpy.array: y values of fitted pdf

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> values = np.random.rand(100)
        >>> X, y = fit_beta_pdf(values)

    """
    X = np.linspace(xmin, xmax, nb_points)
    ab,bb,cb,db = stats.beta.fit(raw_values)
    pdf_beta = stats.beta.pdf(X, ab, bb,cb, db)
    return X, pdf_beta


def create_random_distribution_scores(tgs, tfs):
    """
    Create random scores for the given target genes and transcription factors.
    This function is useful to evaluate the estimated GRNs (comparing to the
    GRNs of a random predictor).

    Args:
        tgs (list): list of target genes
        tfs (list): list of transcription factors

    Returns:
        pandas.DataFrame: randomly generated scores (rows = TGs ; columns = TFs)

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tgs = ["gene1", "gene2", "gene3", "gene4", "gene5"]
        >>> tfs = ["gene1", "gene3", "gene4"]
        >>> rand_scores = create_random_distribution_scores(tgs, tfs)
        >>> rand_scores
                  gene1     gene3     gene4
        gene1       NaN  0.400157  0.978738
        gene2  2.240893  1.867558 -0.977278
        gene3  0.950088       NaN -0.103219
        gene4  0.410599  0.144044       NaN
        gene5  0.761038  0.121675  0.443863

    """
    # generating random scores for the tfs-tgs interactions
    rand = np.random.randn(len(tgs),len(tfs))
    scores_rand = pd.DataFrame(rand, index=tgs, columns=tfs)
    # deleting the self loops
    for tf in tfs:
        scores_rand[tf][tf] = None
    return scores_rand

# to be used only once - saves a file that can be used as a reference to evaluate all GRNs
def generate_rand_metrics(tgs,
                          tfs,
                          gold_std_grn,
                          n_iterations=25000,
                          path=None,
                          **eval_res_params):
    """
    Generates evaluation scores from the function evaluate_result()
    (default: AUROC, AUPR) for randomly generated GRNs.
    The random scores are given to all possible interactions tgs-tfs.
    Then this GRN is evaluated using the gold_std_grn.
    This process is repeated for n_iterations.

    Args:
        tgs (list): list of target genes
        tfs (list): list of transcription factors
        gold_std_grn (pandas.DataFrame): reference GRN used as a gold standard,
            where rows are links with names of the type <TF symbol>_<TG symbol>,
            and columns are respectively 'TF':TF symbol, 'TG':TG symbol, and
            'IS_REGULATED': indicates whether the link exists (1) or not (0)
        n_iterations (int): (default 25000) number of random GRNs to generate
        path (str): (default None) if specified, folder where to store the .csv
            file with the return matrix
        eval_res_params: (optional) parameters of called evaluate_result()
            function

    Returns:
        pandas.DataFrame: matrix
        (rows = iterations, columns = metrics of evaluated random GRN)

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tgs = ['gene1', 'gene2', 'gene3']
        >>> tfs = ['gene1', 'gene3']
        >>> grn = pd.DataFrame(np.array([['gene1', 'gene2', 1],
                                         ['gene1', 'gene3', 0],
                                         ['gene3', 'gene2', 1],
                                         ['gene3', 'gene1', 0]]),
                                         columns=['TF', 'TG', 'IS_REGULATED'])
        >>> grn.index=grn['TF']+'_'+grn['TG']
        >>> grn["IS_REGULATED"] = grn["IS_REGULATED"].astype(int)
        >>> random_metrics = generate_rand_metrics(tgs,
                                                   tfs,
                                                   grn,
                                                   n_iterations=5)
        >>> random_metrics
           AUROC      AUPR  Precision  Recall  Accuracy        F1
        0   0.75  0.791667   1.000000     0.5      0.75  0.666667
        1   0.50  0.416667   0.666667     1.0      0.75  0.800000
        2   0.50  0.708333   1.000000     0.5      0.75  0.666667
        3   1.00  1.000000   1.000000     1.0      1.00  1.000000
        4   0.75  0.791667   1.000000     0.5      0.75  0.666667

    """

    res = []
    for i in tqdm(range(0, n_iterations)):
        scores_rand = create_random_distribution_scores(tgs, tfs)
        eval_res_params["scores"] = scores_rand
        eval_res_params["gold_std_grn"] = gold_std_grn
        metrics = evaluate_result(**eval_res_params)
        res.append(metrics)
    res = pd.DataFrame(res)
    if path:
        res.to_csv(path+".csv")
    return(res)

def compute_p_values_from_pdf(x, X, Y):
    """
    Compute the p_value of a given metric with respect to a randomized
    reference probability density function.

    Args:
        x (float): metric value to test
        X (numpy.ndarray): metric randomized reference (probability density
            function X)
        Y (numpy.ndarray): frequency randomized reference (probability density
            function Y)

    Returns:
        float: p_value of x

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> index = np.random.randint(0, 10)
        >>> X = np.random.standard_normal(10)
        >>> x = X[index]
        >>> np.random.seed(1)
        >>> Y = np.random.standard_normal(10)
        >>> p_value = compute_p_values_from_pdf(x, X, Y)
        >>> p_value
        0.008318328172202086

    """
    dx = X[1]-X[0]
    p_val = ((dx*Y)[X>=x]).sum()
    return(p_val)

def compute_p_values_from_raw_distribution(x,X_rand):
    """
    Compute the p_value of a given metric with respect to a randomized
    reference distribution (raw values).

    Args:
        x (float): metric value to test
        X_rand (numpy.ndarray): randomized reference raw values

    Returns:
        float: p_value of x

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> x = np.random.rand()
        >>> X = np.random.standard_normal(1000)
        >>> p_value = compute_p_values_from_raw_distribution(x, X)
        >>> p_value
        0.273

    """
    p_val = (X_rand >=x).sum()/X_rand.shape[0]
    return p_val


def pvalue2score(p_value):
    """
    Compute the -log10 score of a p_value.

    Args:
        p_value (float): p value

    Returns:
        float: -log10 score

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> p_value = np.random.random()
        >>> pvalue2score(p_value)
        0.2605752110604732

    """
    epsilon = 1e-300
    return(-np.log10(p_value+epsilon))


def pr_auc_score(y_true, y_pred):
    """
    Compute area under the Precision-Recall curve.

    Args:
        y_true (pandas.Series): Boolean values for the ground truth
            (correct target values)
        y_pred (pandas.Series): Continuous values from predictor

    Returns:
        float: area under the Precision-Recall curve

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> y_true = pd.Series(np.random.randint(0,2,10))
        >>> y_pred = pd.Series(np.random.randn(10))
        >>> aupr = pr_auc_score(y_true, y_pred)
        >>> aupr
        0.4755555555555555

    """
    pr_curve = precision_recall_curve(y_true,y_pred)
    aupr = auc(pr_curve[1], pr_curve[0])
    return aupr

def shuffle_matrix(A, axis=0):
    """
    Shuffle the values of a matrix, in order to compute scores under shuffled
    expression condition, to break relations between genes (null hypothesis)

    Args:
        A (pandas.DataFrame or numpy.array): matrix
        axis (int): 0 for columns and 1 for rows

    Returns:
        pandas.DataFrame or numpy.array: shuffled matrix

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["c1", "c2", "c3", "c4", "c5"],
                            columns=["gene1", "gene2", "gene3", "gene4", "gene5"])
        >>> data
               gene1     gene2     gene3     gene4     gene5
        c1  1.764052  0.400157  0.978738  2.240893  1.867558
        c2 -0.977278  0.950088 -0.151357 -0.103219  0.410599
        c3  0.144044  1.454274  0.761038  0.121675  0.443863
        c4  0.333674  1.494079 -0.205158  0.313068 -0.854096
        c5 -2.552990  0.653619  0.864436 -0.742165  2.269755

        >>> shuffle_matrix(data,0)
               gene1     gene2     gene3     gene4     gene5
        c1  0.333674  1.494079 -0.205158 -0.742165  1.867558
        c2  0.144044  1.454274  0.978738  0.121675 -0.854096
        c3 -0.977278  0.950088  0.761038  0.313068  0.443863
        c4 -2.552990  0.653619 -0.151357 -0.103219  2.269755
        c5  1.764052  0.400157  0.864436  2.240893  0.410599
    """
    A = A.copy()
    A_ = np.apply_along_axis(np.random.permutation, axis, A)
    A_ = pd.DataFrame(A_,index=A.index,columns=A.columns)
    return A_


def grn_to_networkx(GRN,genes=None,to_undirected=True,add_isolated_nodes=True):
    """
    Convert a data frame containing benchmark GRN links into a networkx graph.

    Args:
        GRN (pandas.DataFrame): column "TF" represent the regulators and "TG" the target genes
        colmun IS_REGULATED indicates whether the TF regulates the TG (1) or not (0)
        genes (list): list of genes to extract sub-graph

    Returns:
        networkx.DiGraph: GRN as networkx.Digraph

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"TF":["tf1","tf2","tf1","tf2"],
                              "TG":["tf2","tf1","tg1","tg2"],
                              "IS_REGULATED":[1,1,1,1]})
        >>> grn = grn_to_networkx(X,["tf1","tf2"])
        >>> grn.edges()
        EdgeView([('tf1', 'tf2')])
    """
    GRN_true = GRN[GRN["IS_REGULATED"].astype(bool)]
    if genes is not None:
        GRN_true = GRN_true[np.logical_and(GRN_true["TF"].isin(genes),GRN_true["TG"].isin(genes))]
    if to_undirected:
        grn = nx.from_pandas_edgelist(GRN_true, 'TF', 'TG', 'IS_REGULATED')
    else:
        grn = nx.from_pandas_edgelist(GRN_true, 'TF', 'TG', 'IS_REGULATED',create_using=nx.DiGraph())
    if add_isolated_nodes and genes is not None:
        grn.add_nodes_from([g for g in genes if g not in grn.nodes])
    return grn

def rank_to_networkx(GRN,top_n=1000,genes=None,to_undirected=False):
    """
    Convert a data frame containing ranked GRN link scores into a networkx graph.

    Args:
        GRN (pandas.DataFrame): column "TF" represent the regulators and "TG" the target genes column "rank" indicates the importance of the link
        top_n (int): Number of top edges to include in the graph
        genes (list): list of genes to extract sub-graph
        to_undirected (bool): convert to undirected?

    Returns:
        networkx.DiGraph: GRN as networkx.Digraph

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"TF":["tf1","tf2","tf1","tf2"],
                              "TG":["tf2","tf1","tg1","tg2"],
                              "rank":[1,2,3,4],
                              "score":[1,0.9,0.8,0.7]})
        >>> grn = rank_to_networkx(X,4,["tf1","tf2"])
        >>> grn.edges()
        EdgeView([('tf1', 'tf2')])
    """
    GRN_true = GRN[GRN["rank"]<=top_n]
    if genes is not None:
        GRN_true = GRN_true[np.logical_and(GRN_true["TF"].isin(genes),
                                           GRN_true["TG"].isin(genes))]
    if to_undirected:
        grn = nx.from_pandas_edgelist(GRN_true,'TF', 'TG', 'rank')
    else:
        grn = nx.from_pandas_edgelist(GRN_true,'TF', 'TG', 'rank',create_using=nx.DiGraph())
    return grn

def pairwise_grn_distance(GRN,genes=None,to_undirected=True):
    """
    Compute the pairwise shortest-path distance between genes within a GRN.

    Args:
        GRN (pandas.DataFrame): column "TF" represent the regulators and "TG" the target genes
        column IS_REGULATED indicates whether the TF regulates the TG (1) or not (0)
        genes (list): list of genes to extract sub-graph
        to_undirected (bool): make undirected GRN graph

    Returns:
        numpy.array: matrix of distance between genes

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"TF":["tf1","tf2","tf1","tf2"],
                              "TG":["tf2","tf1","tg1","tg2"],
                              "IS_REGULATED":[1,1,1,1]})
        >>> pairwise_grn_distance(X,["tf1","tf2"])
             tf1  tf2
        tf1  0.0  1.0
        tf2  1.0  0.0
    """
    grn = grn_to_networkx(GRN,genes,to_undirected)
    distances = nx.floyd_warshall_numpy(grn,grn.nodes)
    distances = pd.DataFrame(distances,index=grn.nodes,columns=grn.nodes)
    return distances

def best_short_path_pairwise_distances(short_path_dist_matrix, A, B):
    """
    Let A and B be two sets of genes, for each gene a in A, the shortest distance
    with genes in B in a precomputed shortest-path distance matrix
    short_path_dist_matrix is retrieved.
    In practice this can be used to compare the distance between the list of
    predicted TFs for a given TG (set of genes A) and the list of true TFs for
    the same TG (set of genes B)

    Args:
        short_path_dist_matrix (pandas.DataFrame): matrix of distance between genes
        A (list): list of genes (e.g., list of predicted TFs for a given TG)
        B (list): list of genes (e.g., list of true TFs for a given TG)

    Returns:
        pandas.Series: distance to closest gene in B for each gene in A

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"TF":["tf1","tf2","tf1","tf2"],
                              "TG":["tf2","tf1","tg1","tg2"],
                              "IS_REGULATED":[1,1,1,1]})
        >>> D = pairwise_grn_distance(X,["tf1","tf2"])
        >>> best_short_path_pairwise_distances(D, ["tf1"], ["tf1"])
        tf1    0.0
        dtype: float64
        >>> best_short_path_pairwise_distances(D, ["tf1"], ["tf2"])
        tf2    1.0
        dtype: float64
    """
    if not len(A) or not len(B):
        return pd.Series()
    distances = {}
    for i,a in enumerate(A):
        top_b = short_path_dist_matrix[a][B].idxmin()
        distances[a+"_"+top_b] = short_path_dist_matrix[a][top_b]
    distances = pd.Series(distances)
    return distances

def prediction_truth_distance(prediction_ranks,grn_truth,n_top,tfs):
    """
    Given a gold-standard GRN pandas DataFrame grn_truth, and a pandas DataFrame
    containing the score ranks predicted prediction_ranks, a top number of links
    to study n_top and a list of transcription factors tfs, this function
    computes the undirected shortest-path pairwise distance matrix between tfs in
    the gold-standard GRN, if two genes are not connected in the gold-standard,
    their shortest-path distance is inf. Both dataframes are transformed into
    networkx graphs. For each node n in the set of common nodes (intersection of nodes
    between both graphs), the tfs of n are computed in both networks, and for each
    predicted tf of n, we compute the shortest path to the closest true tf.


    Args:
        prediction_ranks (pandas.DataFrame): column "TF" represent the regulators and "TG" the target genes
        grn_truth (pandas.DataFrame): column "TF" represent the regulators and "TG" the target genes
        top_n (int): Number of top edges to include in the graph
        tfs (list): list of transcription factors

    Returns:
        pandas.Series: distance to closest true TF of each predicted TF

    Example:
        >>> import pandas as pd
        >>> G = pd.DataFrame({"TF":["tf1","tf2","tf1","tf2"],
                              "TG":["tf2","tf1","tg1","tg2"],
                              "IS_REGULATED":[1,1,1,1]})
        >>> X = pd.DataFrame({"TF":["tf1","tf2","tf1","tf2"],
                              "TG":["tf2","tf1","tg1","tg2"],
                              "rank":[1,2,3,4],
                              "score":[1,0.9,0.8,0.7]})
        >>> n_top = 4
        >>> tfs = ["tf1","tf2"]
        >>> prediction_truth_distance(X,G,n_top,tfs)
    """
    D = pairwise_grn_distance(grn_truth,genes=tfs,to_undirected=True)
    directed_tf_grn = grn_to_networkx(grn_truth,to_undirected=False)
    predicted_grn = rank_to_networkx(prediction_ranks,n_top)
    results = []
    common_nodes = set(predicted_grn.nodes).intersection(set(directed_tf_grn.nodes))
    for n in common_nodes:
        tfs_pred = list(predicted_grn.predecessors(n))
        tfs_true = list(directed_tf_grn.predecessors(n))
        results.append(best_short_path_pairwise_distances(D, tfs_pred, tfs_true))
    return pd.concat(results)
