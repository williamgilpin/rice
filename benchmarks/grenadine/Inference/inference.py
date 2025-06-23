# -*- coding: utf-8 -*-
"""
This module allows to infer co-expression  Gene Regulatory Networks using
gene expression data (RNAseq or Microarray).
"""
from scipy import cluster
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from grenadine.Preprocessing.standard_preprocessing import z_score
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.metrics import pairwise_distances


from collections import Counter

class gene_model:
    """
    Make a gene-level self expressive optimized model

    Args:
        params_gridsearch (dict): parameters for sklearn GridSearchCV to perform 
        a meta-parameter optimization for the gene-level self-expressive model

    Examples:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.model_selection import StratifiedKFold
        >>> model = DecisionTreeClassifier(min_samples_leaf=3)
        >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                      "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                     "criterion":["gini","entropy",]
                    }
        >>> params_gridsearch = {"estimator":model,
                                 "param_grid":params,
                                 "return_train_score":True,
                                 "scoring":["balanced_accuracy",
                                             "accuracy",
                                             "f1_weighted",
                                             "f1_macro",
                                             "recall_weighted",
                                             "recall_macro",
                                             "roc_auc_ovr_weighted"],
                                 "refit":"roc_auc_ovr_weighted",
                                 "cv":StratifiedKFold(5),
                                 "n_jobs":-1,
                                }
        >>> gm = gene_model(params_gridsearch)


    """
    def __init__(self,params_gridsearch):
        self.params_gridsearch = params_gridsearch
        self.gscv = GridSearchCV(**params_gridsearch)
    
    def fit(self,X,y):
        """
        Fit the self expressive gene-level model

        Args:
            X (pandas.DataFrame): Transcription factor gene expressions (discretized
                or not) where rows are experimental conditions and columns are transcription factors
            y (pandas.Series): Target gene expression vector (discretized) where
                rows are experimental conditions

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gm = gene_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> tg = pd.Series(np.random.randint(0,3,size=50))
            >>> gm.fit(X,tg)
        """
        self.gscv.fit(X,y)

    def predict(self, X, y=None):
        """
        Predict the expression level of a target gene given the expression of the regulators

        Args:
            X (pandas.DataFrame): Transcription factor gene expressions (discretized
                or not) where rows are experimental conditions and columns are transcription factors
            y (pandas.Series): Target gene expression vector (discretized) where
                rows are experimental conditions (not used for prediction)

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gm = gene_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> tg = pd.Series(np.random.randint(0,3,size=50))
            >>> gm.fit(X,tg)
            >>> gm.predict(X,tg)
        """
        return self.gscv.predict(X)
    
    def regulators_importance(self):
        """
        Compute the feature importance of regulators using the feature_importance_ or the score_ attribute from sklearn model

        Args:
            X (pandas.DataFrame): Transcription factor gene expressions (discretized
                or not) where rows are experimental conditions and columns are transcription factors
            y (pandas.Series): Target gene expression vector (discretized) where
                rows are experimental conditions

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gm = gene_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> tg = pd.Series(np.random.randint(0,3,size=50))
            >>> gm.fit(X,tg)
            >>> gm.regulators_importance()
            tf1    0.407930
            tf2    0.166075
            tf3    0.425995
            dtype: float64
        """
        features = self.gscv.best_estimator_.feature_names_in_
        importance = None
        if "feature_importances_" in self.gscv.best_estimator_.__dir__():
            importance = self.gscv.best_estimator_.feature_importances_
        elif "coef_" in self.gscv.best_estimator_.__dir__():
            if len(self.gscv.best_estimator_.coef_.shape) > 1:
                importance = self.gscv.best_estimator_.coef_.mean(axis=0)
            else:
                importance = self.gscv.best_estimator_.coef_.flatten()
        if importance is not None:
            return pd.Series(importance, index=features)
    
    def score(self,X=None, y=None, metrics=["balanced_accuracy","roc_auc_ovr_weighted"]):
        """
        Return the scores for the best self-expressive gene-level model. If X or y are None, returns the validation scores

        Args:
            X (pandas.DataFrame): Transcription factor gene expressions (discretized
                or not) where rows are experimental conditions and columns are transcription factors
            y (pandas.Series): Target gene expression vector (discretized) where
                rows are experimental conditions
            metrics (list): list of sklearn evaluation metrics

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gm = gene_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> tg = pd.Series(np.random.randint(0,3,size=50))
            >>> gm.fit(X,tg)
            >>> gm.get_scores()
            mean_fit_time                        0.001037
            std_fit_time                         0.000429
            mean_score_time                      0.003107
            std_score_time                        0.00008
            param_ccp_alpha                          0.01
                                                   ...   
            split2_train_roc_auc_ovr_weighted    0.882115
            split3_train_roc_auc_ovr_weighted    0.894333
            split4_train_roc_auc_ovr_weighted    0.896417
            mean_train_roc_auc_ovr_weighted      0.905028
            std_train_roc_auc_ovr_weighted       0.019257
            Name: 16, Length: 113, dtype: object
        """
        if X is None or y is None:
            best = self.gscv.best_index_
            return pd.DataFrame(self.gscv.cv_results_).iloc[best,:]
        else:
            y_pred = self.predict(X)
            scores = pd.Series({m: get_scorer(m)._score_func(y, y_pred)for m in metrics})
            return scores
         

class gxn_model:
    """
    Make a gene-level self expressive optimized model

    Args:
        params_gridsearch (dict): parameters for sklearn GridSearchCV to perform 
        a meta-parameter optimization for each gene-level self-expressive model
        verbose (bool): print extra information 
        clustering (sklearn.cluster.*): the model clusters genes via the input clustering method 
        and uses the center-most gene of every cluster to get the hyperparameters 
        for the gene-level self-expressive model. If None, no clustering is performed.

    Examples:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.model_selection import StratifiedKFold
        >>> model = DecisionTreeClassifier(min_samples_leaf=3)
        >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                      "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                     "criterion":["gini","entropy",]
                    }
        >>> params_gridsearch = {"estimator":model,
                                 "param_grid":params,
                                 "return_train_score":True,
                                 "scoring":["balanced_accuracy",
                                             "accuracy",
                                             "f1_weighted",
                                             "f1_macro",
                                             "recall_weighted",
                                             "recall_macro",
                                             "roc_auc_ovr_weighted"],
                                 "refit":"roc_auc_ovr_weighted",
                                 "cv":StratifiedKFold(5),
                                 "n_jobs":-1,
                                }
        >>> gm = gene_model(params_gridsearch)


    """
    def __init__(self,params_gridsearch,verbose=True,clustering=None):
        self.params_gridsearch = params_gridsearch
        self.verbose = verbose
        self.clustering = clustering
        self.C_ = None
        self.scores_ = None
        self.gene_models_ = None

        self.clustering_model_ = None
        self.clustering_hyperparameters_ = None
        self.clustering_inertia_ = None
        self.nb_clusters_ = None
        
    def fit(self,X,Y,regulators,to_inspect=None):  
        """
        Fit the self expressive gene regulatory model

        Args:
            X (pandas.DataFrame): Predictor gene expression matrix where 
                rows are experimental conditions and columns are genes
            Y (pandas.DataFrame): Target gene expression matrix (discretized if classification model used) 
                where rows are experimental conditions
            regulators (list): list of columns in X used as predictors (i.e., regulators)
            to_inspect (list): list of columns in Y to model, if None all genes in Y are analyzed

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gxn = gxn_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> from grenadine.Preprocessing.discretization import discretize_genexp
            >>> Y = discretize_genexp(X,method="kmeans",nb_bins=2,axis=0)
            >>> regulators = ["tf1","tf2"]
            >>> gxn.fit(X,Y,regulators)
        """
        self.check_duplicated_columns_X(X)
        regulators = self.check_duplicated_regulators(regulators)
        self.gene_models_ = {}
        if self.clustering is not None:
            self.clustering_hyperparameters_search(X,Y,regulators)
        if to_inspect is None:
            to_inspect = Y.columns
        for i,gene in tqdm(enumerate(to_inspect)):
            if self.verbose and i%100 == 0:
                _ = os.write(1,bytes(f"processed: {i} genes \n",'utf-8'))
            if self.clustering:
                params_gridsearch = self.params_gridsearch.copy()
                params_gridsearch["param_grid"] = self.clustering_hyperparameters_[self.clustering_model_.labels_[i]]
            else:
                params_gridsearch = self.params_gridsearch
            self.gene_models_[gene] = gene_model(params_gridsearch)
            X_train = X[[e for e in regulators if e!=gene]]
            y_train = Y[gene]
            self.gene_models_[gene].fit(X_train,y_train)
    
    def check_duplicated_columns_X(self,X):
        """
        Check the columns of matrix X for duplicates and inform the user if any are found
        """
        column_counts = Counter(X.columns)
        for column_name, count in column_counts.most_common():
            if count > 1:
                raise ValueError(f"Column '{column_name}' appears {count} times in the expression matrix")
    
    def check_duplicated_regulators(self,regulators):
        """
        Check the list of regulators and inform the user if there are any duplicates
        If duplicates are found, modify the list to remove them
        """
        cnt_reg = Counter(regulators).most_common()
        modified = False
        for reg, count in cnt_reg:
            if count > 1:
                modified = True
                if self.verbose:
                    _ = os.write(1,bytes(f"The gene '{reg}' appears {count} times in the list of regulators. \n",'utf-8'))
               
        if(modified):
            regulators = list(set(regulators))
            if self.verbose:
                _ = os.write(1,bytes(f"Modified the list of regulators to only have unique values. \n",'utf-8'))
        
        return regulators
            
    
    def clustering_hyperparameters_search(self,X,Y,regulators):
        """
        Search for the optimal hyperparameters for the gene-level self-expressive model using clustering methods

        Args:
            X (pandas.DataFrame): Predictor gene expression matrix where
                rows are experimental conditions and columns are genes
            Y (pandas.DataFrame): Target gene expression matrix (discretized if classification model used)
                where rows are experimental conditions
            regulators (list): list of columns in X used as predictors (i.e., regulators)

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> from sklearn.cluster import KMeans
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                        "param_grid":params,
                                        "return_train_score":True,
                                        "scoring":["balanced_accuracy",
                                                    "accuracy",
                                                    "f1_weighted",
                                                    "f1_macro",
                                                    "recall_weighted",
                                                    "recall_macro",
                                                    "roc_auc_ovr_weighted"],
                                        "refit":"roc_auc_ovr_weighted",
                                        "cv":StratifiedKFold(5),
                                        "n_jobs":-1,
                                        }
            >>> gxn = gxn_model(params_gridsearch, clustering=KMeans(random_state=0))
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                    columns=["tf1","tf2","tf3"])
            >>> from grenadine.Preprocessing.discretization import discretize_genexp
            >>> Y = discretize_genexp(X,method="kmeans",nb_bins=2,axis=0)
            >>> regulators = ["tf1","tf2"]
            >>> gxn.fit(X,Y,regulators)
        """
        model = kelbow_visualizer(self.clustering, X.T, k=20, show=False) 
        self.clustering.set_params(n_clusters = model.elbow_value_)

        self.clustering_model_ = self.clustering.fit(X.T)
        self.clustering_inertia_ = self.clustering_model_.inertia_

        closest_genes_idx = list(np.argmin(pairwise_distances(self.clustering_model_.cluster_centers_,X.T), axis=1))
        closest_genes = [list(X.T.index)[i] for i in closest_genes_idx]

        self.clustering_hyperparameters_ = []
        for gene in closest_genes:
            if self.verbose:
                _ = os.write(1,bytes(f"Getting hyperparameters for gene: {gene} \n",'utf-8'))

            self.gene_models_[gene] = gene_model(self.params_gridsearch)
            X_train = X[[e for e in regulators if e!=gene]]
            y_train = Y[gene]
            self.gene_models_[gene].fit(X_train,y_train)

            parameters = self.gene_models_[gene].gscv.best_estimator_.get_params()
            for param in parameters:
                parameters[param] = [parameters[param]]
            self.clustering_hyperparameters_.append(parameters)
            
    def transform(self):
        """
        Build the gene self expressive network coefficients

        Returns:
            pandas.DataFrame: self expressive network coefficients matrix.
            Rows are target genes and columns are regulators.
            The value at row i and column j represents the feature importance score
            assigned by the self-expressive model to the regulatory relationship 
            between target gene i and regulator j.   

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gxn = gxn_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> from grenadine.Preprocessing.discretization import discretize_genexp
            >>> Y = discretize_genexp(X,method="kmeans",nb_bins=2,axis=0)
            >>> regulators = ["tf1","tf2"]
            >>> gxn.fit(X,Y,regulators)
            >>> gxn.transform()
        """
        self.C_ = {m: self.gene_models_[m].regulators_importance() for m in self.gene_models_} 
        self.C_ = pd.DataFrame(self.C_)
        self.C_ = self.C_.T
        self.C_.fillna(0,inplace=True)
        return self.C_
    
    def score(self,X=None,Y=None):
        """
        Return the scores for the best self-expressive model for each gene

        Args:
            X

        Returns:
            pandas.DataFrame: Evaluation scores for each gene, rows represent genes and columns metrics

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gxn = gxn_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> from grenadine.Preprocessing.discretization import discretize_genexp
            >>> Y = discretize_genexp(X,method="kmeans",nb_bins=2,axis=0)
            >>> regulators = ["tf1","tf2"]
            >>> gxn.fit(X,Y,regulators)
            >>> gxn.score()
            
        """
        if X is None:
            self.scores_ = pd.DataFrame({m: self.gene_models_[m].score() for m in self.gene_models_})
            self.scores_ = self.scores_.T
            self.scores_.fillna(0,inplace=True)
            return self.scores_
        else:
            self.scores_ = {}
            for i,gene in tqdm(enumerate(Y.columns)):
                X_train = X[[e for e in X.columns if e!=gene]]
                y_train = Y[gene]
                self.scores_[gene] = self.gene_models_[gene].score(X_train,y_train)
            self.scores_ = pd.DataFrame(self.scores_)
            return self.scores_
    
    def purify_model(self,score_condition,inplace=False):
        """
        Return a gxn_model purified keeping only genes with score following a given criteria

        Returns:
            gxn_model : purified model

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> from sklearn.model_selection import StratifiedKFold
            >>> model = DecisionTreeClassifier(min_samples_leaf=3,random_state=0)
            >>> params = {"min_impurity_decrease":[1e-2,5e-2,1e-1,5e-1],
                          "ccp_alpha":[1e-1,5e-2,1e-2,5e-3,1e-3],
                         "criterion":["gini","entropy",]
                        }
            >>> params_gridsearch = {"estimator":model,
                                     "param_grid":params,
                                     "return_train_score":True,
                                     "scoring":["balanced_accuracy",
                                                 "accuracy",
                                                 "f1_weighted",
                                                 "f1_macro",
                                                 "recall_weighted",
                                                 "recall_macro",
                                                 "roc_auc_ovr_weighted"],
                                     "refit":"roc_auc_ovr_weighted",
                                     "cv":StratifiedKFold(5),
                                     "n_jobs":-1,
                                    }
            >>> gxn = gxn_model(params_gridsearch)
            >>> import pandas as pd
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = pd.DataFrame(np.random.randn(50,3),
                                 columns=["tf1","tf2","tf3"])
            >>> from grenadine.Preprocessing.discretization import discretize_genexp
            >>> Y = discretize_genexp(X,method="kmeans",nb_bins=2,axis=0)
            >>> regulators = ["tf1","tf2"]
            >>> gxn.fit(X,Y,regulators)
            >>> def score_criteria(score):
                    score["mean_train_roc_auc_ovr_weighted"]>0.6 and score["mean_test_average_precision"]>0.6
            >>> gxn.purify_model(score_criteria)

        """
        mask = [i for i in self.scores_.index if score_condition(self.scores_.loc[i])]
        gene_models_ = {i:self.gene_models_[i] for i in mask}
        if self.C_ is not None:
            C_ = self.C_.loc[mask]
        else:
            C_ = None
        if self.scores_ is not None:
            scores_ = self.scores_.loc[mask]
        else:
            scores_ = None
        if inplace:
            self.gene_models_ = gene_models_
            self.C_ = C_
            self.scores_ = scores_
            return self
        else:
            purified_model = gxn_model(self.params_gridsearch)
            purified_model.gene_models_ = gene_models_
            purified_model.C_ = C_
            purified_model.scores_ = scores_
            return purified_model


def score_links(gene_expression_matrix,
                score_predictor,
                tf_list = None,
                tg_list = None,
                normalize = False,
                discr_method=None,
                progress_bar=False,
                **predictor_parameters):
    """
    Scores transcription factors-target gene co-expressions using a predictor.

    Args:
        gene_expression_matrix (pandas.DataFrame):  gene expression matrix where
            rows are genes and  columns ares samples (conditions).
            The value at row i and column j represents the expression of gene i
            in condition j.
        score_predictor (function): function that receives a pandas.DataFrame X
            containing the transcriptor factor expressions and a pandas.Series
            y containing the expression of a target gene, and scores the
            co-expression level between each transcription factor and the target
            gene.
        tf_list (list or numpy.array): list of transcription factors ids.
        tg_list (list or numpy.array): list of target genes ids.
        normalize (boolean): If True the gene expression of genes is z-scored
        discr_method : discretization method to use, if discretization of target
            gene expression is desired
        progress_bar: bool, if true include progress bar
        **predictor_parameters: Named parameters for the score predictor

    Returns:
        pandas.DataFrame: co-regulation score matrix.

        Rows are target genes and columns are transcription factors.
        The value at row i and column j represents the score assigned by the
        score_predictor to the regulatory relationship between target gene i
        and transcription factor j.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                            columns=["c1", "c2", "c3", "c4", "c5"])
        >>> tf_list = ["gene1", "gene2", "gene5"]

        >>> # Example with a regression method
        >>> from grenadine.Inference.regression_predictors import GENIE3
        >>> scores1 = score_links(gene_expression_matrix=data,
                                  score_predictor=GENIE3,
                                  tf_list=tf_list)
        >>> scores1
                  gene2     gene5     gene1
        gene1  0.484081  0.515919       NaN
        gene2       NaN  0.653471  0.346529
        gene3  0.245136  0.301229  0.453634
        gene4  0.309982  0.306964  0.383054
        gene5  0.529839       NaN  0.470161

        >>> # Example with a classification method
        >>> from grenadine.Inference.classification_predictors import RF_classifier_score
        >>> from grenadine.Preprocessing.discretization import discretize_genexp
        >>> discr_method = lambda X: discretize_genexp (X, "efd", 5, axis=1)
        >>> scores2 = score_links(gene_expression_matrix=data,
                                        score_predictor=RF_classifier_score,
                                        tf_list=tf_list,
                                        discr_method=discr_method)
        >>> scores2
                  gene2     gene5     gene1
        gene1  0.512659  0.487341       NaN
        gene2       NaN  0.463122  0.536878
        gene3  0.368175  0.317341  0.314484
        gene4  0.302738  0.346799  0.350463
        gene5  0.524815       NaN  0.475185

    """

    warnings.warn("score_links is deprecated and no longer mantained, please build a gxn_model instead", DeprecationWarning)
    gene_expression_matrix = gene_expression_matrix.T
    # Set the list of TFs and TGs if necessary
    if tg_list is None:
        tg_list = gene_expression_matrix.columns
    if tf_list is None:
        tf_list = gene_expression_matrix.columns
    tf_list_present = set(gene_expression_matrix.columns).intersection(tf_list)
    tg_list_present = list(set(gene_expression_matrix.columns).intersection(tg_list))
    if not len(tf_list_present):
        Exception('None of the tfs in '+str(tf_list)+\
        " is present in the gene_expression_matrix genes list"+\
        str(gene_expression_matrix.columns))
    if not len(tg_list_present):
        Exception('None of the tgs in '+str(tg_list)+\
        " is present in the gene_expression_matrix genes list"+\
        str(gene_expression_matrix.columns))
    tg_list_present.sort()
    #  Normalize expression data for each gene
    if normalize:
        gene_expression_matrix = z_score(gene_expression_matrix,axis=0)
    # compute tf scores for each gene
    scores_tf_per_gene = []
    for gene in tqdm(tg_list_present,disable=not progress_bar):
        # Exclude the current gene from the tfs list
        tfs2test = list(tf_list_present.difference(set([gene])))
        tfs2test.sort()
        X = gene_expression_matrix[tfs2test].values
        y = gene_expression_matrix[gene].values
        if discr_method is not None :
            y = discr_method(y)
        if len(np.unique(y)) <= 1:
            # handle the case when only one class was detected in y
            score = np.zeros(len(tfs2test))
        else:
            score = score_predictor(X, y, **predictor_parameters)
        # Get the features importance (score for each TF -> Gene)
        scores = pd.Series(score)
        scores.index = tfs2test
        scores_tf_per_gene.append(scores)
    df_results = pd.DataFrame(scores_tf_per_gene, index=tg_list_present)
    return(df_results)



def ensemble_score_links(score_links_matrices, score_links_weights=None):
    """
    Makes an ensemble co-regulation score matrix from a list of co-regulation
    score matrices obtained using different methods, and possibly a list of
    weights for each method

    Args:
        score_links_matrices (list): list of co-regulation score matrices (pandas
            DataFrames)
        score_links_weights (list): list of weights for each method (the higher
            the more confidence on the method). If no value is provided each
            method as a unitary weight

    Returns:
        pandas.DataFrame: co-regulation score matrix.

        Rows are target genes and columns are transcription factors.
        The value at row i and column j represents the score assigned by the
        score_predictor to the regulatory relationship between target gene i
        and transcription factor j.
    """
    if score_links_weights is None:
        score_links_weights = np.asarray([1 for s in score_links_matrices])

    for i,score in enumerate(score_links_matrices):
        score_links_matrices[i] -= score_links_matrices[i].values.flatten().mean()
        score_links_matrices[i] /= score_links_matrices[i].values.flatten().std()
        score_links_matrices[i] *= score_links_weights[i]
        if i == 0:
            score_links = score_links_matrices[i]
        else:
            score_links += score_links_matrices[i]
    return score_links/score_links_weights.sum()


def clean_nan_inf_scores(scores):
    """
    Replaces nan and -inf scores by the (minimum_score - 1), and inf scores by
    (maximum_score + 1)

    Args:
        scores (pandas.DataFrame): co-regulation score matrix.
        Rows are target genes and columns are transcription factors.
        The value at row i and column j represents the score assigned by the
        score_predictor to the regulatory relationship between target gene i
        and transcription factor j.

    Returns:
        pandas.DataFrame: co-regulation score matrix.

        Rows are target genes and columns are transcription factors.
        The value at row i and column j represents the score assigned by the
        score_predictor to the regulatory relationship between target gene i
        and transcription factor j.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(5, 5),
                            index=["gene1", "gene2", "gene3", "gene4", "gene5"],
                            columns=["c1", "c2", "c3", "c4", "c5"])
        >>> tf_list = ["gene1", "gene2", "gene5"]

        >>> # Example with a regression method
        >>> from grenadine.Inference.regression_predictors import GENIE3
        >>> scores1 = score_links(gene_expression_matrix=data,
                                  score_predictor=GENIE3,
                                  tf_list=tf_list)
        >>> scores1
                  gene2     gene5     gene1
        gene1  0.484081  0.515919       NaN
        gene2       NaN  0.653471  0.346529
        gene3  0.245136  0.301229  0.453634
        gene4  0.309982  0.306964  0.383054
        gene5  0.529839       NaN  0.470161
        >>> clean_nan_inf_scores(scores1)
                  gene2     gene5     gene1
        gene1  0.484081  0.515919  0.245126
        gene2  0.245126  0.653471  0.346529
        gene3  0.245136  0.301229  0.453634
        gene4  0.309982  0.306964  0.383054
        gene5  0.529839  0.245126  0.470161

    """
    # replacing NaN scores with minimum score - epsilon
    epsilon = 1e-10
    nan_scores = np.isnan(scores)
    min_score =  scores[np.logical_not(nan_scores)].min().min()
    scores[nan_scores] = min_score-epsilon
    # replacing infinite scores with maximum score + epsilon
    finite_scores = np.isfinite(scores)
    max_score =  scores[finite_scores].max().max()
    positive_inf = np.logical_and(np.logical_not(finite_scores),finite_scores>0)
    negative_inf = np.logical_and(np.logical_not(finite_scores),finite_scores<0)
    scores[positive_inf] = max_score+epsilon
    scores[positive_inf] = min_score-epsilon
    return scores


def rank_GRN(coexpression_scores_matrix, take_abs_score=False, clean_scores=True, pyscenic_format=False):
    """
    Ranks the co-regulation scores between transcription factors and target genes.

    Args:
        coexpression_scores_matrix (pandas.DataFrame):co-expression score matrix
            where rows are target genes and columns are transcription factors.
            The value at row i and column j represents the score assigned by a
            score_predictor to the regulatory relationship between target gene i
            and transcription factor j.
        take_abs_score (bool): take the absolute value of the score instead of
            taking scores themselves
    Returns:
        pandas.DataFrame: ranking matrix.

        A ranking matrix contains a row for each possible regulatory link, it
        also contains 4 columns, namely the rank, the score, the transcription
        factor id, and the target gene id.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(np.random.randn(3, 2),
                            index=["gene1", "gene2", "gene3"],
                            columns=["gene1", "gene3"])
        >>> # scores associated to self loops are set to nan
        >>> data.iloc[0,0]=np.nan
        >>> data.iloc[2,1]=np.nan
        >>> ranking_matrix = rank_GRN(data)
        >>> ranking_matrix
                     rank     score     TF     TG
        gene3_gene2   1.0  2.240893  gene3  gene2
        gene1_gene3   2.0  1.867558  gene1  gene3
        gene1_gene2   3.0  0.978738  gene1  gene2
        gene3_gene1   4.0  0.400157  gene3  gene1

    """
    # print("G", np.mean(coexpression_scores_matrix.values), coexpression_scores_matrix.shape)
    # print(coexpression_scores_matrix[:5])
    np.random.seed(0)
    epsilon = 1e-10
    rand_values = np.random.rand(*coexpression_scores_matrix.shape)*epsilon
    coexpression_scores_matrix += rand_values
    # print("After rand:", np.mean(coexpression_scores_matrix.values))
    if clean_scores:
        coexpression_scores_matrix = clean_nan_inf_scores(coexpression_scores_matrix)
    if take_abs_score:
        coexpression_scores_matrix = np.abs(coexpression_scores_matrix)
    coexpression_unstack = coexpression_scores_matrix.unstack(level=0)
    ranking = pd.DataFrame()
    ranking["rank"] = coexpression_unstack.rank(method="dense",
                                                ascending=False,
                                                na_option="bottom")
    # print("Ranking:", np.mean(coexpression_unstack))
    ranking["score"] = coexpression_unstack
    ranking["TF"] = list(ranking.index.get_level_values(level=0))
    ranking["TG"] = list(ranking.index.get_level_values(level=1))
    ranking = ranking.sort_values("score",ascending=False)
    ranking.index = ranking["TF"].astype(str)+"_"+ranking["TG"].astype(str)
    ranking = ranking.dropna()
    if pyscenic_format:
        ranking.rename(columns={'score':'importance',
                'TG':'target'},
                inplace=True)
        ranking = ranking.reset_index()
        ranking = ranking.drop(["rank","index"],axis=1)
        cols = ranking.columns.tolist()
        cols = ["TF","target","importance"]
        ranking = ranking[cols]
    return(ranking)

def join_rankings_scores_df(**rank_scores):
    """
    Join rankings and scores data frames generated by different methods.

    Args:
        **rank_scores: Named parameters, where arguments names should be the
            methods names and arguments values correspond to pandas.DataFrame
            output of rank_GRN

    Returns:
        (pandas.DataFrame, pandas.DataFrame): joined ranks and joined scores
            where rows represent possible regulatory links and columns represent
            each method.
            Values at row i and column j represent resp. the rank or the score
            of edge i computed by method j.

    Examples:
        >>> import pandas as pd
        >>> method1_rank = pd.DataFrame([[1,1.3, "gene1", "gene2"],
                                         [2,1.1, "gene1", "gene3"],
                                         [3,0.9, "gene3", "gene2"]],
                                         columns=['rank', 'score', 'TF', 'TG'])
        >>> method1_rank.index = method1_rank['TF']+'_'+method1_rank['TG']
        >>> method2_rank = pd.DataFrame([[1,1.4, "gene1", "gene3"],
                                         [2,1.0, "gene1", "gene2"],
                                         [3,0.9, "gene3", "gene2"]],
                                         columns=['rank', 'score', 'TF', 'TG'])
        >>> method2_rank.index = method2_rank['TF']+'_'+method2_rank['TG']
        >>> ranks, scores = join_rankings_scores_df(method1=method1_rank, method2=method2_rank)
        >>> ranks
                     method1  method2
        gene1_gene2        1        2
        gene1_gene3        2        1
        gene3_gene2        3        3
        >>> scores
                     method1  method2
        gene1_gene2      1.3      1.0
        gene1_gene3      1.1      1.4
        gene3_gene2      0.9      0.9

    """
    ranks_df = {method:rank_scores[method]["rank"] for method in rank_scores}
    ranks_df = pd.DataFrame(ranks_df)
    scores_df = {method:rank_scores[method]["score"] for method in rank_scores}
    scores_df = pd.DataFrame(scores_df)
    return(ranks_df,scores_df)