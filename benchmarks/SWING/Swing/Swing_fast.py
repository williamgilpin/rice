import random
import pandas as pd
import numpy as np
import warnings
import functools
from scipy import stats

from .Window import Window
from .RFRWindow import RandomForestRegressionWindow
from .DionesusWindow import DionesusWindow
from .LassoWindow import LassoWindow
from .util.Evaluator import Evaluator
from .util import utility_module as util


class Swing(object):
    """
    An object that grabs different timepoints of data, can set window and step size, and makes an edge-list.

    """

    def __init__(self, file_path, gene_start=None, gene_end=None, time_label="Time", separator="\t",
                 window_type="RandomForest", step_size=1, min_lag=0, max_lag=0, window_width=3, sub_dict = None):
        """
        Initialize the SWING object. Read the file and put it into a pandas dataframe
        :param file_path: string
            File to read
        :param gene_start: int
        :param gene_end: int
        :param time_label: str
        :param separator: str
        :param window_type: str
        :param step_size: int
        :param min_lag: int
        :param max_lag: int or None
        :param window_width: int
        :return:
        """
        # Precompute boolean masks for each timepoint to avoid repeated DataFrame filtering
        self.time_masks = {
            t: (self.norm_data[self.time_label] == t).values
            for t in self.time_vec
        }
        # Read the raw data into a pandas dataframe object
        self.raw_data = pd.read_csv(file_path, sep=separator)
        self.raw_data = self.raw_data.dropna(axis=0, how='all')
        if sub_dict is not None:
            valid_genes = sub_dict['genes']
            new_cols = [time_label] + list(valid_genes)
            self.raw_data = self.raw_data[new_cols]

        self.file_path = file_path
        self.window_type = window_type

        # Set SWING defaults
        self.current_step = 0
        self.window_width = window_width
        self.step_size = step_size
        self.time_label = time_label

        self.alpha = None
        self.tf_list = None

        # Get overall width of the time-course
        self.time_vec = self.raw_data[self.time_label].unique()
        self.overall_width = len(self.time_vec)

        # Set lag defaults
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.check_lags()

        if gene_end is not None:
            self.gene_end = gene_end
        else:
            self.gene_end = len(self.raw_data.columns)
        if gene_start is not None:
            self.gene_start = gene_start
        else:
            self.gene_start = 0

        self.gene_list = self.raw_data.columns.values[self.gene_start:self.gene_end]
        self.window_list = []

        # assign norm data for window creation.
        # by default, norm_data is raw_data and is later modified by other functions.
        self.norm_data = self.raw_data

        self.full_edge_list = None
        self.edge_dict = None
        self.lag_set = None

    def get_n_windows(self):
        """
        Calculate the number of windows

        Called by:
            create_windows
            get_window_stats

        :return: int
        """
        total_windows = int((self.overall_width - self.window_width + 1.0) / self.step_size)
        return int(total_windows)
    # def get_window_raw(self, start_index, random_time=False):
    #     """
    #     Select a window from the full data set. This is fancy data-frame slicing

    #     Called by:
    #         create_windows
    #         get_window_stats
    #         get_window

    #     :param start_index: int
    #         The start of the window
    #     :param random_time: bool, optional
    #     :return: data-frame
    #     """
    #     if random_time:
    #         # select three random timepoints
    #         time_window = self.time_vec[start_index]
    #         choices = self.time_vec
    #         choices = np.delete(choices, start_index)
    #         for x in range(0, self.window_width - 1):
    #             chosen_time = random.choice(choices)
    #             time_window = np.append(time_window, chosen_time)
    #             chosen_index = np.where(choices == chosen_time)
    #             choices = np.delete(choices, chosen_index)
    #     else:
    #         end_index = start_index + self.window_width
    #         time_window = self.time_vec[start_index:end_index]
    #     data = self.norm_data[self.norm_data[self.time_label].isin(time_window)]
    #     return data
    

    def get_window_raw(self, start_index, random_time=False):
        # Build the window of timepoints
        if random_time:
            # select three random timepoints
            time_window = self.time_vec[start_index]
            choices = self.time_vec
            choices = np.delete(choices, start_index)
            for x in range(0, self.window_width - 1):
                chosen_time = random.choice(choices)
                time_window = np.append(time_window, chosen_time)
                chosen_index = np.where(choices == chosen_time)
                choices = np.delete(choices, chosen_index)
        else:
            end_index = start_index + self.window_width
            time_window = self.time_vec[start_index:end_index]

        # Combine masks once, then slice underlying numpy arrays directly
        mask = functools.reduce(
            np.logical_or, (self.time_masks[t] for t in time_window)
        )
        df = self.norm_data.iloc[mask]
        return df

    # need to do something about this method. keep for now, but currently need a "preprocess" method.
    def remove_blank_rows(self):
        """
        Removes a row if the sum of that row is NaN

        Called by:


        :return:
        """
        """calculates sum of rows. if sum is NAN, then remove row"""
        coln = len(self.raw_data.columns)
        sums = [self.raw_data.iloc[:, x].sum() for x in range(0, coln)]
        ind = np.where(np.isnan(sums))[0]
        self.raw_data.iloc[:, ind] = 0

    def _build_window(self, index, random_time, td_window):
        if index + self.window_width > self.overall_width:
            raise Exception("Window out of bounds")
        explanatory_indices = self.get_explanatory_indices(index)
        if explanatory_indices is None:
            return None

        raw = self.get_window_raw(index, random_time)
        exp_dict, resp_dict = self.get_window_data(index, explanatory_indices)
        info = {
            "time_label": self.time_label,
            "gene_start": self.gene_start,
            "gene_end":   self.gene_end,
            "nth_window": index
        }
        return self.get_window_object(raw, info, td_window, exp_dict, resp_dict)

    def create_windows(self, random_time=False, n_jobs=1):
        """
        Parallel window creation using joblib.
        """
        self.check_lags()
        td_window = not (self.min_lag == self.max_lag == 0)

        # Dispatch each window to a worker
        windows = Parallel(n_jobs=n_jobs)(
            delayed(self._build_window)(idx, random_time, td_window)
            for idx in range(self.get_n_windows())
        )
        # Filter out None windows (e.g. left-censored)
        self.window_list = [w for w in windows if w is not None]
    
    def create_custom_windows(self, tf_list,random_time=False):
        """
        Create window objects for SWING to use, with set explanatory variables (such as TFs)

        Called by:
            pipeline

        :return:
        """
        # Initialize empty lists
        window_list = []
        self.tf_list=tf_list

        # Check to make sure lags are valid if parameters have been changed
        self.check_lags()

        # If min_lag is 0 and max_lag is 0 then you don't need a tdWindow
        if self.min_lag == 0 and self.max_lag == 0:
            td_window = False
        else:
            td_window = True

        # Generate possible windows using specified SWING parameters
        for index in range(0, self.get_n_windows()):

            # Confirm that the window will not be out of bounds
            if (index + self.window_width) > self.overall_width:
                raise Exception('Window created that is out of bounds based on parameters')

            explanatory_indices = self.get_explanatory_indices(index)
            raw_window = self.get_window_raw(index, random_time)
            if explanatory_indices is not None:
                explanatory_dict, response_dict = self.get_window_data(index, explanatory_indices)

                # remove information from explanatory window
                to_remove = list(set(explanatory_dict['explanatory_labels'])-set(tf_list))
                for removed_tf in to_remove:
                    # remove from explanatory_labels
                    removed_index = np.where(explanatory_dict['explanatory_labels'] == removed_tf)[0][0]
                    explanatory_dict['explanatory_labels'] = np.delete(explanatory_dict['explanatory_labels'],
                                                                       removed_index)

                    # explanatory_window
                    explanatory_dict['explanatory_window'] = np.delete(explanatory_dict['explanatory_window'],
                                                                       removed_index)

                    # explanatory_data
                    explanatory_dict['explanatory_data'] = np.delete(explanatory_dict['explanatory_data'],
                                                                     removed_index, axis=1)
                    # not explanatory_times

                window_info = {"time_label": self.time_label, "gene_start": self.gene_start, "gene_end": self.gene_end,
                               "nth_window": index}
                window_object = self.get_window_object(raw_window, window_info, td_window, explanatory_dict,
                                                       response_dict)
                window_list.append(window_object)

        self.window_list = window_list

    def check_lags(self):
        """
        Make sure the user specified lags meet necessary criteria
        :return:
        """
        if self.min_lag > self.max_lag and self.max_lag is not None:
            raise ValueError('The minimum lag {} cannot be greater than the maximum lag {}'.format(self.min_lag,
                                                                                                   self.max_lag))

        if self.min_lag < 0:
            raise ValueError('The minimum lag {} cannot be negative'.format(self.min_lag))

        if self.min_lag > self.get_n_windows():
            raise ValueError('The minimum lag {} cannot be greater '
                             'than the number of windows {}'.format(self.min_lag, self.get_n_windows()))

        if self.max_lag >= self.get_n_windows():
            raise ValueError('The maximum lag {} cannot be greater '
                             'than or equal to the number of windows {}'.format(self.max_lag, self.get_n_windows()))

    def strip_dataframe(self, dataframe):
        """
        Split dataframe object components into relevant numpy arrays
        :param dataframe:
        :return:
        """
        df = dataframe.copy()
        df_times = df[self.time_label].values
        df.drop(self.time_label, axis=1, inplace=True)
        data = df.values
        labels = df.columns.values
        return df_times, data, labels

    def get_window_data(self, index, explanatory_indices):
        """
        Get the appropriate data for the window
        :param index:
        :param explanatory_indices:
        :return:
        """
        # Get the data for the current window
        response_df = self.get_window_raw(index)
        response_times, response_data, response_labels = self.strip_dataframe(response_df)
        response_window = np.array([index]*len(response_labels))
        response_dict = {'response_times': response_times, 'response_data': response_data,
                         'response_labels': response_labels, 'response_window': response_window}

        explanatory_times, explanatory_data, explanatory_labels, explanatory_window = None, None, None, None

        # Get the data for each lagged window
        for ii, idx in enumerate(explanatory_indices):
            current_df = self.get_window_raw(idx)
            current_times, current_data, current_labels = self.strip_dataframe(current_df)
            current_window = np.array([idx]*len(current_labels))
            if ii == 0:
                # Initialize values
                explanatory_times = current_times.copy()
                explanatory_data = current_data.copy()
                explanatory_labels = current_labels.copy()
                explanatory_window = current_window.copy()
            else:
                # concatenate relevant windows horizontally.
                explanatory_data = np.hstack((explanatory_data, current_data))
                explanatory_times = np.append(explanatory_times, current_times)
                explanatory_labels = np.append(explanatory_labels, current_labels)
                explanatory_window = np.append(explanatory_window, current_window)

        explanatory_dict = {'explanatory_times': explanatory_times, 'explanatory_data': explanatory_data,
                            'explanatory_labels': explanatory_labels, 'explanatory_window': explanatory_window}

        return explanatory_dict, response_dict

    def get_window_object(self, dataframe, window_info_dict, td_window, explanatory_dict, response_dict):
        """
        Return a window object from a data-frame

        Called by:
            create_windows

        :param dataframe: data-frame
        :param window_info_dict: dict
            Dictionary containing information needed for window initialization
        :return:
        """
        window_obj = None

        if self.window_type == "Lasso":
            window_obj = LassoWindow(dataframe, window_info_dict, td_window, explanatory_dict,
                                     response_dict)
        elif self.window_type == "RandomForest":
            window_obj = RandomForestRegressionWindow(dataframe, window_info_dict, td_window,
                                                      explanatory_dict, response_dict)
        elif self.window_type == "Dionesus":
            window_obj = DionesusWindow(dataframe, window_info_dict, td_window, explanatory_dict,
                                        response_dict)

        return window_obj

    def rank_windows(self, n_permutes=10, n_bootstraps=10, n_alphas=20, noise=0.2):
        """
        Run tests to score and rank windows

        Called by:


        :param n_permutes: int, optional
            Number of permutes to run. Default is 1,000
        :param n_bootstraps: int, optional
            Number of bootstraps to run. Default is 1,000
        :param n_alphas: int, optional
            Number of alpha values to test if using Lasso. Default is 20
        :param noise: float ([0,1]), optional
            The amount of noise to add to bootstrapped windows. Default is 0.2

        :return:
        """
        for window in self.window_list:
            window.run_permutation_test(n_permutes)
            window.run_bootstrap(n_bootstraps, n_alphas, noise)
            window.make_edge_table()

    def optimize_params(self):
        """
        Optimize window fit parameters

        Called by:
            pipeline


        :return:
        """
        if self.window_type is "Lasso":
            for window in self.window_list:
                window.initialize_params(alpha=self.alpha)
        else:
            for window in self.window_list:
                window.initialize_params()

        return self.window_list

    def fit_windows(self, pcs=None, alpha=None, n_trees=None, n_jobs=None, show_progress=True):
        """
        Fit each window in the list

        Called by:
            pipeline

        :param alpha:
        :param n_trees:
        :return:
        """

        for i, window in enumerate(self.window_list):
            if self.window_type == "Lasso":
                if alpha is not None:
                    window.alpha = alpha
            if self.window_type == "RandomForest":
                if n_trees is not None:
                    window.n_trees = n_trees
                if n_jobs is not None:
                    window.n_jobs = n_jobs
            if self.window_type == "Dionesus":
                if pcs is not None:
                    window.num_pcs = pcs
            if show_progress:
                print(f"Fitting window {i} of {self.get_n_windows()}")
                if window.td_window:
                    print(f"Fitting window index {window.nth_window} against the following window indices: {window.earlier_windows}")
                    
            window.fit_window()

        return self.window_list

    def rank_edges(self, n_bootstraps=1000, permutation_n=1000):
        """
        Run tests to rank edges in windows

        Called by:
            pipeline

        :param n_bootstraps:
        :param permutation_n:
        :return:
        """
        if self.window_type == "Dionesus":
            for window in self.window_list:
                window.make_edge_table()

        if self.window_type == "Lasso":
            for window in self.window_list:
                window.run_permutation_test(n_permutations=permutation_n)
                print("Running bootstrap...")
                window.run_bootstrap(n_bootstraps=n_bootstraps)
                window.make_edge_table()
        if self.window_type == "RandomForest":
            for window in self.window_list:
                window.make_edge_table()
        return self.window_list

    def zscore_all_data(self):
        """
        Zscore the data in a data-frame

        Called by:
            pipeline

        :return: z-scored dataframe
        """

        zscored_dataset = pd.DataFrame(stats.zscore(self.raw_data.values, axis=0, ddof=1),
                                       index=self.raw_data.index,
                                       columns=self.raw_data.columns)

        zscored_dataset[self.time_label] = self.raw_data[self.time_label]
        self.norm_data = zscored_dataset

        return zscored_dataset

    def get_window_stats(self):
        """
        Generate a dictionary of relevant information from a window
            N : the number of data points in this window,
            time_labels: the names of the time points
            step_size: the step-size of the current model
            window_size: the size of the window of the current model
            total_windows: the number of windows total
            window_index: the index of the window. counts start at 0. ie if the window index is 0 it is the 1st window.
            If the window index is 12, it is the 12th window in the series.

        Called by:


        :return: dict
        """
        """for each window, get a dict:
            N : the number of datapoints in this window,
            time_labels: the names of the timepoints
            step_size: the step-size of the current model
            window_size: the size of the window of the current model
            total_windows: the number of windows total
            window_index: the index of the window. counts start at 0. ie if the window index is 0 it is the 1st window. if the window index is 12, it is the 12th window in the series."""
        current_window = self.get_window_raw()

        # calculate the window index
        min_time = np.amin(current_window[self.time_label])
        window_index = np.where(self.time_vec == min_time) / self.step_size

        window_stats = {'N': len(current_window.index),
                        'time_labels': current_window[self.time_label].unique(),
                        'step_size': self.step_size,
                        'window_size': self.window_width,
                        'total_windows': self.get_n_windows(),
                        'window_index': window_index}
        return window_stats

    def compile_edges(self, self_edges=False):
        """
        Edges across all windows will be compiled into a single edge list
        :return:
        """
        import numpy as np

        print("Compiling all model edges...", end='')
        df = None
        for ww, window in enumerate(self.window_list):
            # Get the edges and associated values in table form
            current_df = window.make_edge_table()

            current_df['adj_imp'] = np.abs(current_df['Importance'])

            if self.window_type == "Dionesus":  # Fixed comparison with "is" to "=="
                current_df['adj_imp'] = np.abs(current_df['Importance'])
            elif self.window_type == "Lasso":
                current_df['adj_imp'] = np.abs(current_df['Stability'])
            current_df.sort_values(['adj_imp'], ascending=False, inplace=True)

            current_df['Rank'] = np.arange(0, len(current_df))

            if df is None:
                df = current_df.copy()
            else:
                df = pd.concat([df, current_df.copy()], ignore_index=True)  # Updated to use pd.concat

        if not self_edges:
            df = df[df['Parent'] != df['Child']]  # Updated syntax for column access

        df['Edge'] = list(zip(df['Parent'], df['Child']))
        df['Lag'] = df['C_window'] - df['P_window']
        self.full_edge_list = df.copy()
        print("[DONE]")

        return


    def make_static_edge_dict(self, self_edges=False, lag_method='max_median'):
        """
        Make a dictionary of edges
        :return:
        """
        print("Lumping edges...", end='')
        df = self.full_edge_list.copy()

        # Only keep edges with importance > 0. Values below 0 are not helpful for model building
        df = df[df['Importance'] > 0]

        # Ignore self edges if desired
        if not self_edges:
            df = df[df.Parent != df.Child]
        edge_set = set(df.Edge)

        # Calculate the full set of potential edges with TF list if it is provided.

        if self.tf_list is not None:
            full_edge_set = set(util.make_possible_edge_list(np.array(self.tf_list), self.gene_list, self_edges=self_edges))
        else:
            full_edge_set = set(util.make_possible_edge_list(self.gene_list, self.gene_list, self_edges=self_edges))

        # Identify edges that could exist, but do not appear in the inferred list
        edge_diff = full_edge_set.difference(edge_set)

        self.edge_dict = {}
        lag_importance_score, lag_lump_method = lag_method.split('_')
        score_method = eval('np.'+lag_importance_score)
        lump_method = eval('np.'+lag_lump_method)
        for idx, edge in enumerate(full_edge_set):
            if idx % 1000 == 0:
                print(str(idx)+" out of "+ str(len(full_edge_set)), end='')
            if edge in edge_diff:
                self.edge_dict[edge] = {"dataframe": None, "mean_importance": 0,
                                        "max_importance": 0, 'max_edge': None, 'lag_importance': 0,
                                        'lag_method': lag_method, 'rank_importance': np.nan, 'adj_importance': 0}
                continue

            current_df = df[df['Edge'] == edge]
            max_idx = current_df['Importance'].idxmax()
            lag_set = list(set(current_df.Lag))
            lag_imp = score_method([lump_method(current_df.Importance[current_df.Lag == lag]) for lag in lag_set])
            lag_adj_imp = score_method([lump_method(current_df.adj_imp[current_df.Lag == lag]) for lag in lag_set])
            lag_rank = score_method([lump_method(current_df.Rank[current_df.Lag == lag]) for lag in lag_set])
            self.edge_dict[edge] = {"dataframe":current_df, "mean_importance":np.mean(current_df.Importance), "max_importance":current_df.Importance[max_idx],
                                    'max_edge':(current_df.P_window[max_idx], current_df.C_window[max_idx]),
                                    'lag_importance': lag_imp, 'lag_method':lag_method,
                                    'rank_importance': lag_rank, 'adj_importance':lag_adj_imp}
        print("...[DONE]")
        if edge_diff:
            message = 'The last %i edges had no meaningful importance score' \
                      ' and were placed at the bottom of the list' %len(edge_diff)
            warnings.warn(message)
        return

    def _make_possible_edge_list(self,parents, children, self_edges=True):
        """
        Create a list of all the possible edges between parents and children

        :param parents: array
            labels for parents
        :param children: array
            labels for children
        :param self_edges:
        :return: array, length = parents * children
            array of parent, child combinations for all possible edges
        """
        parent_index = range(len(parents))
        child_index = range(len(children))

        a, b = np.meshgrid(parent_index, child_index)
        parent_list = list(parents[a.flatten()])
        child_list = list(children[b.flatten()])
        possible_edge_list = None
        if self_edges:
            possible_edge_list = list(zip(parent_list, child_list))

        elif not self_edges:
            possible_edge_list = [x for x in zip(parent_list, child_list) if x[0] != x[1]]

        return possible_edge_list

    def make_sort_df(self, df, sort_by='mean'):
        """
        Calculate the mean for each edge
        :param df: dataframe
        :return: dataframe
        """

        sort_field = sort_by+"_importance"

        print("Calculating {} edge importance...".format(sort_by), end='')
        temp_dict = {edge: df[edge][sort_field] for edge in df.keys()}
        sort_df = pd.DataFrame.from_dict(temp_dict, orient='index')
        sort_df.columns = [sort_field]
        if sort_by.lower() == 'rank':
            sort_df.sort_values(sort_field, ascending=True, inplace=True)
        else:
            sort_df.sort_values(sort_field, ascending=False, inplace=True)

        sort_df.index.name = 'regulator-target'
        sort_df = sort_df.reset_index()
        print("[DONE]")
        return sort_df

    def get_samples(self):
        df = pd.read_csv(self.file_path,sep='\t')
        node_list = df.columns.tolist()
        node_list.pop(0)
        return node_list

    def get_explanatory_indices(self,index):
        # In append mode, the start index can always be 0
        if self.max_lag is None:
            start_idx = 0
        else:
            start_idx = max(index-self.max_lag, 0)
        end_index = max(index-self.min_lag+1, 0)

        explanatory_indices = range(start_idx, end_index)

        # If the maximum lag required is greater than the index, this window must be left censored
        if len(explanatory_indices) == 0 or self.max_lag > index:
            explanatory_indices = None

        return explanatory_indices

    def score(self, sorted_edge_list, gold_standard_file=None):
        """
        Scores some stuff, I think...
        Called by:
            pipeline
        :param sorted_edge_list:
        :param gold_standard_file:
        :return:
        """
        print("Scoring model...", end='')
        current_gold_standard = gold_standard_file

        evaluator = Evaluator(current_gold_standard, '\t', node_list=self.get_samples())
        tpr, fpr, auroc = evaluator.calc_roc(sorted_edge_list)
        auroc_dict = {'tpr':np.array(tpr), 'fpr':np.array(fpr), 'auroc': np.array(auroc)}
        precision, recall, aupr = evaluator.calc_pr(sorted_edge_list)
        aupr_random = [len(evaluator.gs_flat)/float(len(evaluator.full_list))]*len(recall)
        aupr_dict = {"precision": np.array(precision), "recall": np.array(recall), "aupr": np.array(aupr),
                     "aupr_random": np.array(aupr_random)}
        print("[DONE]")
        return auroc_dict, aupr_dict

