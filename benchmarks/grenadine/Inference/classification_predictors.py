"""
This module allows to infer Gene Regulatory Networks using
gene expresion data (RNAseq or Microarray). This module implements several
inference algorithms based on classification, using `scikit-learn`_.

.. _scikit-learn:
    https://scikit-learn.org
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as _sklearn_RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as _sklearn_ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier as _sklearn_AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier as _sklearn_GradientBoostingClassifier
from sklearn.svm import LinearSVC as _sklearn_SVC
from sklearn.naive_bayes import MultinomialNB as _sklearn_MultinomialNB
from sklearn.naive_bayes import ComplementNB as _sklearn_ComplementNB
from sklearn.ensemble import BaggingClassifier as _sklearn_BaggingClassifier
from collections import Counter

def RF_classifier_score(X, y, **rf_parameters):
    """
    Random Forest Classifier, score predictor function based on `scikit-learn`_
    RandomForestClassifier.

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **rf_parameters: Named parameters for the sklearn _sklearn_RandomForestClassifier

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        RandomForestClassifier to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = RF_classifier_score(tfs,tg)
        >>> scores
        array([0.21071429, 0.4       , 0.28928571])
    """
    classifier = _sklearn_RandomForestClassifier(**rf_parameters)
    classifier.fit(X, y)
    scores = classifier.feature_importances_
    return scores

def XRF_classifier_score(X, y, **xrf_parameters):
    """
    Randomized decision trees Classifier, score predictor function based on
    `scikit-learn`_ ExtraTreesClassifier.

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **xrf_parameters: Named parameters for the sklearn _sklearn_ExtraTreesClassifier

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        ExtraTreesClassifier to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = XRF_classifier_score(tfs,tg)
        >>> scores
        array([0.31354167, 0.35520833, 0.33125   ])
    """
    classifier = _sklearn_ExtraTreesClassifier(**xrf_parameters)
    classifier.fit(X, y)
    scores = classifier.feature_importances_
    return(scores)

def AdaBoost_classifier_score(X, y, **adab_parameters):
    """
    AdaBoost Classifier, score predictor function based on `scikit-learn`_
    AdaBoostClassifier.

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **adab_parameters: Named parameters for the sklearn AdaBoostClassifier

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        AdaBoostClassifier to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = AdaBoost_classifier_score(tfs,tg)
        >>> scores
        array([0.24, 0.44, 0.32])
    """
    classifier = _sklearn_AdaBoostClassifier(**adab_parameters)
    classifier.fit(X, y)
    scores = classifier.feature_importances_
    return scores

def GB_classifier_score(X, y, **gb_parameters):
    """
    Gradient Boosting Classifier, score predictor function based on
    `scikit-learn`_ GradientBoostingClassifier.

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **gb_parameters: Named parameters for the sklearn _sklearn_ExtraTreesClassifier

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        GradientBoostingClassifier to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = GB_classifier_score(tfs,tg)
        >>> scores
         array([0.33959125, 0.21147015, 0.4489386 ])
    """
    classifier = _sklearn_GradientBoostingClassifier(**gb_parameters)
    classifier.fit(X, y)
    scores = classifier.feature_importances_
    return scores

def SVM_classifier_score(X, y, **svm_parameters):
    """
    SVM Classifier, score predictor function based on `scikit-learn`_ SVC
    (Support Vector Classifier).

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **svm_parameters: Named parameters for the sklearn SVC

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        SVC to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = SVM_classifier_score(tfs,tg)
        >>> scores
        array([0.58413783, 0.5448345 , 0.31764191])
    """
    # consider changing kernel function: SVC(kernel = 'linear'/'poly'/'sigmoid'/'precomputed')
    # default is rbf
    # also consider changing polynomial degree : SVC(kernel = 'poly', degree = int value)
    # default is 3
    # but linear is the only kernel that has coef_ attribute
    #svm_parameters["kernel"] = 'linear'
    svm_parameters["multi_class"] = 'ovr'
    svm_parameters["dual"]=False
    classifier = _sklearn_SVC(**svm_parameters)
    classifier.fit(X, y)
    # coef_ = array of shape (nb_classes, nb_TFs)
    # replacing each class value (= discretized value) of y with corresponding TFs weights from attribute coef_
    # then taking the mean along each column (= mean importance of TF over all conditions)
    if classifier.coef_.shape[0] > 1:
        scores = np.abs(classifier.coef_).mean(axis=0)
        #coef = pd.DataFrame(classifier.coef_,index=classifier.classes_)
        #nb_classes = pd.Series(Counter(y))
        #scores = np.abs((coef.T*nb_classes/nb_classes.sum()).T).mean(axis=0)
    else:
        scores = np.abs(classifier.coef_[0,:])
    return scores


def MultinomialNB_classifier_score(X, y, **nb_parameters):
    """
    Multinomial Naive Bayes Classifier, score predictor function based on
    `scikit-learn`_ MultinomialNB.

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **nb_parameters: Named parameters for the sklearn MultinomialNB

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        MultinomialNB to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = MultinomialNB_classifier_score(tfs,tg)
        >>> scores
        array([0.3010284 , 0.41871716, 0.4272386 ])
    """
    x_shift = 0
    x_min = X.min().min()
    if x_min<0:
        x_shift = -x_min
    classifier = _sklearn_MultinomialNB(**nb_parameters)
    classifier.fit(X+x_shift, y)
    # coef_ = array of shape (nb_classes, nb_TFs)
    # replacing each class value (= discretized value) of y with corresponding TFs weights from attribute coef_
    # then taking the mean along each column (= mean importance of TF over all conditions)
    if classifier.coef_.shape[0] > 1:
        coef = pd.DataFrame(classifier.feature_log_prob_,index=classifier.classes_)
        scores = np.abs(coef)
        scores = coef.fillna(0)
        scores = scores.mean(axis=0)
    else:
        scores = np.abs(classifier.feature_log_prob_[0,:])
        scores = np.nan_to_num(scores)
        #scores = scores.fillna(0)
    return np.array(scores)


def ComplementNB_classifier_score(X, y, **nb_parameters):
    """
    Complement Naive Bayes Classifier, score predictor function based on
    `scikit-learn`_ ComplementtNB.

    Args:
        X (pandas.DataFrame): Transcription factor gene expressions (discretized
            or not) where rows are experimental conditions and columns are
            transcription factors
        y (pandas.Series): Target gene expression vector (discretized) where
            rows are experimental conditions
        **nb_parameters: Named parameters for the sklearn MultinomialNB

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the score assigned by the
        ComplementNB to the regulatory relationship between the target
        gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> tfs = pd.DataFrame(np.random.randn(5,3),
                               index =["c1","c2","c3","c4","c5"],
                               columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,3,size=5), index=["c1","c2","c3","c4","c5"])
        >>> scores = ComplementNB_classifier_score(tfs,tg)
        >>> scores
        array([0.28113447, 0.39096368, 0.45629413])
    """
    x_shift = 0
    x_min = X.min().min()
    if x_min<0:
        x_shift = -x_min
    classifier = _sklearn_ComplementNB(**nb_parameters)
    classifier.fit(X+x_shift, y)
    # coef_ = array of shape (nb_classes, nb_TFs)
    # replacing each class value (= discretized value) of y with corresponding
    # TFs weights from attribute coef_
    # then taking the mean along each column (= mean importance of TF over all
    # conditions)
    if classifier.coef_.shape[0] > 1:
        coef = pd.DataFrame(classifier.feature_log_prob_,index=classifier.classes_)
        scores = np.abs(coef)
        scores = coef.fillna(0)
        scores = scores.mean(axis=0)
    else:
        scores = np.abs(classifier.feature_log_prob_[0,:])
        scores = np.nan_to_num(scores)
        #scores = scores.fillna(0)
    return np.array(scores)


def bagging_classifier_score(X, y, **bagging_parameters):
    """
    Apply the bagging technique to a regression algorithm, based on
    `scikit-learn`_ BaggingClassifier.

    Args:
        X (pandas.DataFrame): Transcriptor factor gene expressions where rows
            are experimental conditions and columns are transcription factors
        y (pandas.Series): Target gene expression vector where rows are
            experimental conditions
        **adab_parameters: Named parameters for the sklearn AdaBoostRegressor

    Returns:
        numpy.array: co-regulation scores.

        The i-th element of the score array represents the average score
        assigned by the Base Regressor to the regulatory relationship
        between the target gene and transcription factor i.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.svm import SVR
        >>> np.random.seed(0)
        >>> svc = SVC(kernel="linear",decision_function_shape='ovr')
        >>> nb_conditions = 10
        >>> tfs = pd.DataFrame(np.random.randn(nb_conditions,3),
                       index =["c"+str(i) for i in range(nb_conditions)],
                       columns=["tf1","tf2","tf3"])
        >>> tg = pd.Series(np.random.randint(0,2,size=nb_conditions),
                           index =["c"+str(i) for i in range(nb_conditions)])
        >>> bagging_parameters = {"base_estimator":svc,
                                  "n_estimators":5,
                                  "max_samples":0.9}
        >>> scores = bagging_classifier_score(tfs,tg,**bagging_parameters)
        >>> scores
        array([0.269231,0.412219,0.299806])
    """
    def get_score(classifier):
        scores=None
        if hasattr(classifier, 'feature_importances_'):
            scores = pd.DataFrame(feature_importances).mean(axis=0)
        elif hasattr(classifier,"coef_"):
            if classifier.coef_.shape[0] > 1:
                scores = np.abs(classifier.coef_).mean(axis=0)
            else:
                scores = np.abs(classifier.coef_[0,:])
        return(scores)
    bc = _sklearn_BaggingClassifier(**bagging_parameters)#BaggingClassifier
    bc.fit(X,y)
    coefs = [get_score(e) for e in bc.estimators_]
    coefs = [c for c in coefs if c is not None]
    if len(coefs):
        scores = pd.DataFrame(coefs).mean(axis=0)
    else:
        print("Base regressor has not a feature_importance of coef_ attribute")
        return(0)
    return(scores)
