import time
import numpy as np
# from operator import itemgetter
from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

def compute_feature_importances(estimator):
    """
    Compute raw feature importances for a (forest or single) tree-based estimator

    Args:
        estimator (sklearn.tree.BaseDecisionTree or 
            sklearn.ensemble.RandomForestRegressor or 
            sklearn.ensemble.ExtraTreesRegressor): 
            The tree-based estimator to compute feature importances for.

    Returns:
        np.ndarray: Feature importances.
    """
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    imps = np.array([t.tree_.compute_feature_importances(normalize=False)
                     for t in estimator.estimators_])
    return imps.mean(axis=0)

def get_link_list(VIM, gene_names=None, regulators='all', maxcount='all', file_name=None):
    """
    Get the link list from the VIM matrix.

    Args:
        VIM (np.ndarray): VIM matrix.
        gene_names (list of str): Gene names.
        regulators (str or list of str): List of regulators.
        maxcount (str or int): Maximum number of links to return.
        file_name (str): File name to save the link list.

    Returns:
        list of str: Link list.
    """
    VIM = np.asarray(VIM)
    if VIM.ndim != 2 or VIM.shape[0] != VIM.shape[1]:
        raise ValueError("VIM must be a square array")
    p = VIM.shape[0]

    # regulator indices
    if regulators == 'all':
        regs = np.arange(p)
    else:
        if gene_names is None:
            raise ValueError("gene_names must be specified when regulators is a list")
        name_to_idx = {g: i for i, g in enumerate(gene_names)}
        regs = np.array([name_to_idx[g] for g in regulators if g in name_to_idx], dtype=int)
        if regs.size == 0:
            raise ValueError("No valid regulators found in gene_names")

    # build and sort edges
    i_idx, j_idx = np.where(~np.eye(p, dtype=bool))
    mask = np.isin(i_idx, regs)
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    scores = VIM[i_idx, j_idx]
    order = np.argsort(-scores)
    i_idx, j_idx, scores = i_idx[order], j_idx[order], scores[order]

    # shuffle zero‐score tail
    zero_start = np.searchsorted(scores[::-1], 0, side='left')
    if zero_start < scores.size:
        tail = np.random.permutation(np.arange(scores.size - zero_start)) + zero_start
        perm = np.concatenate([np.arange(zero_start), tail])
        i_idx, j_idx, scores = i_idx[perm], j_idx[perm], scores[perm]

    n = scores.size if maxcount == 'all' else min(int(maxcount), scores.size)
    labels = (gene_names if gene_names is not None
              else [f"G{i+1}" for i in range(p)])
    lines = [f"{labels[i]}\t{labels[j]}\t{s:.6f}"
             for i, j, s in zip(i_idx[:n], j_idx[:n], scores[:n])]

    if file_name:
        with open(file_name, 'w') as f:
            f.write("\n".join(lines))
    else:
        print("\n".join(lines))


def estimate_degradation_rates(TS_data, time_points):
    """
    Estimate degradation rates for each gene in the timeseries data.

    Args:
        TS_data (list of np.ndarray): Timeseries data arrays.
        time_points (list of np.ndarray): Time points for each timeseries.

    Returns:
        alphas (np.ndarray): Array of degradation rates for each gene.
    """
    TS = [np.asarray(ts) for ts in TS_data]
    mins = [ts.min() for ts in TS]
    C_min = min(mins)

    nexp, p = len(TS), TS[0].shape[1]
    alphas = np.zeros((nexp, p), dtype=float)

    for i, (ts, tp) in enumerate(zip(TS, time_points)):
        t = np.asarray(tp)
        idx_min = np.argmin(ts, axis=0)
        idx_max = np.argmax(ts, axis=0)
        xmin = ts[idx_min, np.arange(p)] - C_min
        xmax = ts[idx_max, np.arange(p)] - C_min
        ok = xmax != xmin
        dt = np.abs(t[idx_max] - t[idx_min])
        alphas[i, ok] = (np.log(np.maximum(xmax[ok], 1e-6))
                         - np.log(np.maximum(xmin[ok], 1e-6))) / dt[ok]

    return alphas.max(axis=0)


def dynGENIE3(TS_data, time_points, alpha='from_data', SS_data=None,
              gene_names=None, regulators='all', tree_method='RF',
              K='sqrt', ntrees=1000, compute_quality_scores=False,
              save_models=False, nthreads=1):
    """
    Run dynGENIE3 to infer the VIM and degradation rates.

    Args:
        TS_data (list of np.ndarray): Timeseries data arrays.
        time_points (list of np.ndarray): Time points for each timeseries.
        alpha (str or float): Degradation rate parameter.
        SS_data (np.ndarray): Static state data.
        gene_names (list of str): Gene names.
        regulators (str or list of str): List of regulators.
        tree_method (str): Tree method.
        K (str or int): Number of features to consider.
        ntrees (int): Number of trees.
        compute_quality_scores (bool): Whether to compute quality scores.
        save_models (bool): Whether to save models.
        nthreads (int): Number of threads.

    Returns:
        VIM (np.ndarray): VIM matrix.
        alphas (np.ndarray): Degradation rates.
        pred_scores (np.ndarray): Predictive scores.
        stab_scores (np.ndarray): Stability scores.
        models (list of sklearn.ensemble.RandomForestRegressor or sklearn.ensemble.ExtraTreesRegressor): Models.
    """
    time0 = time.time()

    # validate inputs
    TS = [np.asarray(x) for x in TS_data]
    p = TS[0].shape[1]
    for x in TS[1:]:
        if x.shape[1] != p:
            raise ValueError("All TS_data arrays must have the same number of columns")

    if len(time_points) != len(TS):
        raise ValueError("time_points must match TS_data length")

    # sort each timeseries
    for i, (x, tp) in enumerate(zip(TS, time_points)):
        tp = np.asarray(tp, dtype=float)
        idx = np.argsort(tp)
        TS[i] = x[idx]
        time_points[i] = tp[idx]

    # compute or assign α
    if alpha == 'from_data':
        alphas = estimate_degradation_rates(TS, time_points)
    elif isinstance(alpha, (int, float)):
        alphas = np.full(p, float(alpha))
    else:
        alphas = np.asarray(alpha, float)
        if alphas.shape != (p,):
            raise ValueError("alpha vector length must match number of genes")

    # regulator indices
    if regulators == 'all':
        regs = list(range(p))
    else:
        regs = [i for i, g in enumerate(gene_names) if g in regulators]

    VIM = np.zeros((p, p), float)
    pred_scores = [] if not compute_quality_scores or tree_method != 'RF' else np.zeros(p)
    stab_scores = [] if not compute_quality_scores else np.zeros(p)
    models = [None] * p if save_models else []

    def worker(i):
        return dynGENIE3_single(TS, time_points, SS_data, i, alphas[i],
                                regs, tree_method, K, ntrees,
                                compute_quality_scores, save_models)

    if nthreads > 1:
        with Pool(nthreads) as pool:
            results = pool.map(worker, range(p))
    else:
        results = map(worker, range(p))

    for i, (vi, ps, ss, model) in enumerate(results):
        VIM[i, :] = vi
        if compute_quality_scores:
            if tree_method == 'RF': pred_scores[i] = ps
            stab_scores[i] = ss
        if save_models:
            models[i] = model

    VIM = VIM.T
    if compute_quality_scores:
        if tree_method == 'RF':
            pred_scores = np.mean(pred_scores)
        stab_scores = np.mean(stab_scores)

    print(f"Elapsed time: {time.time() - time0:.2f}s")
    return VIM, alphas, pred_scores, stab_scores, models


def dynGENIE3_single(TS, time_points, SS_data, out_idx, alpha, regs,
                     tree_method, K, ntrees, compute_quality_scores, save_models):
    h = 1
    ntop = 5
    p = TS[0].shape[1]

    # build design matrix
    X_list, y_list = [], []
    present, future, dt_list = [], [], []
    for ts, tp in zip(TS, time_points):
        tp = np.asarray(tp)
        dts = tp[h:] - tp[:-h]
        X = ts[:-h, regs]
        dy = (ts[h:, out_idx] - ts[:-h, out_idx]) / dts + alpha * ts[:-h, out_idx]
        X_list.append(X)
        y_list.append(dy)
        if compute_quality_scores and tree_method == 'RF':
            present.append(ts[:-h, out_idx])
            future.append(ts[h:, out_idx])
            dt_list.append(dts)

    X_time = np.vstack(X_list)
    y_time = np.concatenate(y_list)
    if SS_data is not None:
        SS = np.asarray(SS_data)
        X_time = np.vstack((SS[:, regs], X_time))
        y_time = np.concatenate((alpha * SS[:, out_idx], y_time))

    # random forest params
    oob = compute_quality_scores and tree_method == 'RF'
    max_feat = ("auto" if K == 'all' or (isinstance(K, int) and K >= len(regs))
                else K)

    model = (RandomForestRegressor(ntrees, max_features=max_feat, oob_score=oob)
             if tree_method == 'RF'
             else ExtraTreesRegressor(ntrees, max_features=max_feat))

    model.fit(X_time, y_time)
    fi = compute_feature_importances(model)
    vi = np.zeros(p, float)
    vi[regs] = fi
    vi[out_idx] = 0
    if vi.sum() > 0:
        vi /= vi.sum()

    ps, ss = None, None
    if compute_quality_scores:
        if tree_method == 'RF':
            # OOB predictions
            preds = model.oob_prediction_
            if SS_data is not None:
                nss = SS.shape[0]
                oob_SS = preds[:nss] / alpha
                oob_TS = (preds[nss:] - alpha * np.concatenate(present)) * np.concatenate(dt_list) + np.concatenate(present)
                y_true = np.concatenate((SS[:, out_idx], np.concatenate(future)))
                ps, _ = pearsonr(np.concatenate((oob_SS, oob_TS)), y_true)
            else:
                oob_TS = (preds - alpha * np.concatenate(present)) * np.concatenate(dt_list) + np.concatenate(present)
                ps, _ = pearsonr(oob_TS, np.concatenate(future))

        # stability
        imps = np.array([t.tree_.compute_feature_importances(normalize=False)
                         for t in model.estimators_])
        if out_idx in regs:
            k = regs.index(out_idx)
            imps = np.delete(imps, k, axis=1)
        imps += np.random.uniform(1e-12, 1e-11, imps.shape)
        if imps.sum() > 0:
            ranks = np.argsort(-imps, axis=1)[:, :ntop]
            ss = (sum(len(set(ranks[i]).intersection(ranks[j]))
                      for i in range(ntrees) for j in range(i))
                  / (ntrees*(ntrees-1)/2) / ntop)
        else:
            ss = 0.0

    return vi, ps, ss, (model if save_models else None)
