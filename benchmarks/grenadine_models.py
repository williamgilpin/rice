import pandas as pd
import numpy as np
from grenadine.Inference.inference import score_links

from grenadine.Inference.regression_predictors import (GENIE3, BayesianRidgeScore, 
                                                       AdaBoost_regressor, SVR_score, 
                                                       TIGRESS, Elastica, GRNBoost2, XGENIE3)
from grenadine.Inference.statistical_predictors import (CLR, wasserstein_distance_score, 
                                                        energy_distance_score, wilcoxon_score)
from grenadine.Inference.classification_predictors import (SVM_classifier_score, 
                                                           ComplementNB_classifier_score)


def process_grenadine(X, method=GENIE3):
    """
    Convert a data matrix into a format that Grenadine can process, and run the 
    specified method on the data.

    Args:
        X (np.ndarray): A data matrix with shape (n_genes, n_samples)
         method (function): A Grenadine method to use for scoring links

    Returns:
        np.ndarray: A matrix of scores for each gene-gene pair
    """ 
    Xpd = pd.DataFrame(X.T)
    gene_names = ["G" + str(i).zfill(4) for i in range(X.shape[1])]
    # set row index to gene names
    Xpd.index = gene_names
    tf = gene_names.copy()
    score_matrix = score_links(Xpd, method, tf) 
    score_matrix = score_matrix[tf]
    score_matrix = score_matrix.T[tf].T
    cmat = score_matrix.values.copy()
    np.fill_diagonal(cmat, 0)
    return cmat

genie3 = lambda X: process_grenadine(X, GENIE3)
bayesian_ridge = lambda X: process_grenadine(X, BayesianRidgeScore)
svr = lambda X: process_grenadine(X, SVR_score)
tigress = lambda X: process_grenadine(X, TIGRESS)
elastica = lambda X: process_grenadine(X, Elastica)
clr = lambda X: process_grenadine(X, CLR)
adaboost = lambda X: process_grenadine(X, AdaBoost_regressor)
grnboost2 = lambda X: process_grenadine(X, GRNBoost2)
xgenie3 = lambda X: process_grenadine(X, XGENIE3)
wasserstein_gren = lambda X: process_grenadine(X, wasserstein_distance_score)
svm = lambda X: process_grenadine(X, SVM_classifier_score)
complement_nb = lambda X: process_grenadine(X, ComplementNB_classifier_score)
energy = lambda X: process_grenadine(X, energy_distance_score)
wilcoxon = lambda X: process_grenadine(X, wilcoxon_score)
