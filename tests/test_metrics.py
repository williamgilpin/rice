import unittest
import numpy as np
from rice.metrics import (
    auprc_score, rocauc_score,
    early_precision, top_k_threshold, top_k_precision,
    bootstrap_graph, empirical_percentile,
    reachable_nodes, downstream_adjacency_graph, compute_metrics
)

class TestMetrics(unittest.TestCase):

    def test_auprc_and_rocauc_scores(self):
        atrue = np.array([[1, 0], [0, 1]])
        apred = atrue.astype(float)
        self.assertAlmostEqual(auprc_score(atrue, apred), 1.0)
        self.assertAlmostEqual(rocauc_score(atrue, apred), 1.0)

    def test_early_precision_basic(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.2, 0.8, 0.1])
        ep = early_precision(y_true, y_pred, normalize=False)
        self.assertEqual(ep, 1)
        ep_norm = early_precision(y_true, y_pred, normalize=True)
        self.assertAlmostEqual(ep_norm, 0.5)

    def test_top_k_threshold(self):
        arr = np.array([1, 2, 3, 4])
        threshed = top_k_threshold(arr, k=2)
        self.assertTrue(np.array_equal(threshed, np.array([0, 0, 1, 1])))

    def test_top_k_precision_perfect(self):
        atrue = np.eye(3)
        apred = atrue.copy()
        prec = top_k_precision(atrue, apred, k=3, check_transpose=False)
        self.assertAlmostEqual(prec, 1.0)

    def test_bootstrap_graph_methods(self):
        amat = np.array([[1, 2], [3, 4]])
        # resample
        bs_res = bootstrap_graph(amat, n_samples=10, method="resample", seed=0)
        self.assertEqual(bs_res.shape, (10, 2, 2))
        self.assertTrue(set(bs_res.ravel()).issubset(set(amat.ravel())))
        # permute
        bs_perm = bootstrap_graph(amat, n_samples=5, method="permute", seed=0)
        self.assertEqual(bs_perm.shape, (5, 2, 2))
        for sample in bs_perm:
            self.assertEqual(set(sample.ravel()), set(amat.ravel()))
        # uniform
        bs_uni = bootstrap_graph(amat, n_samples=3, method="uniform", seed=0)
        self.assertEqual(bs_uni.shape, (3, 2, 2))
        self.assertTrue(np.all(bs_uni >= amat.min()))
        self.assertTrue(np.all(bs_uni <= amat.max()))

    def test_empirical_percentile(self):
        arr = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(empirical_percentile(arr, 3), 0.5)

    def test_reachable_and_downstream(self):
        adj = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])
        reach0 = reachable_nodes(adj, 0)
        self.assertEqual(set(reach0), {0, 1, 2})
        downstream = downstream_adjacency_graph(adj)
        expected = np.array([[0, 1, 1],
                             [0, 0, 1],
                             [0, 0, 0]])
        self.assertTrue(np.array_equal(downstream, expected))

    def test_compute_metrics_perfect(self):
        atrue = np.array([[1, 0], [0, 1]])
        apred = atrue.astype(float)
        metrics = compute_metrics(
            atrue.copy(), apred.copy(),
            verbose=False, significance=False
        )
        self.assertAlmostEqual(metrics["AUPRC Multiplier"], 2.0)
        self.assertAlmostEqual(metrics["ROC-AUC Multiplier"], 2.0)
        self.assertAlmostEqual(metrics["AUPRC"], 1.0)
        self.assertAlmostEqual(metrics["ROC-AUC"], 1.0)
        self.assertEqual(metrics["Early Precision"], 0)
        self.assertAlmostEqual(metrics["Early Precision Rate"], 0.0)
        self.assertAlmostEqual(metrics["Early Precision Ratio"], 0.0)

if __name__ == '__main__':
    unittest.main()
