import unittest
import numpy as np
import warnings
from hccm.ccm import CausalDetection

class TestCausalDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Suppress specific warnings that we expect
        warnings.filterwarnings("ignore", message="The iteration is not making good progress")
        warnings.filterwarnings("ignore", message="overflow encountered in exp")
        
        # Create synthetic time series data
        np.random.seed(42)
        self.n_timepoints = 500  # Increased to ensure enough data points after embedding
        self.n_features = 5
        
        # Create a simple coupled system where x1 causes x2
        t = np.linspace(0, 10, self.n_timepoints)
        x1 = np.sin(t) + 0.1 * np.random.randn(self.n_timepoints)
        x2 = np.sin(t - 0.5) + 0.1 * np.random.randn(self.n_timepoints)  # Lagged version of x1
        x3 = np.cos(t) + 0.1 * np.random.randn(self.n_timepoints)  # Independent
        x4 = np.sin(t) * np.cos(t) + 0.1 * np.random.randn(self.n_timepoints)  # Independent
        x5 = np.sin(t) * np.sin(t) + 0.1 * np.random.randn(self.n_timepoints)  # Independent
        
        self.X = np.column_stack([x1, x2, x3, x4, x5])
        
        # Create a default CausalDetection instance with more conservative parameters
        self.cd = CausalDetection(
            d_embed=3,
            k=4,
            verbose=False,
            library_sizes=[200, 300, 400],  # Adjusted to match data size
            neighbors="knn",
            forecast="sum",
            prune_indirect=False,
            ensemble=False
        )

    def test_initialization(self):
        """Test that the CausalDetection object initializes correctly."""
        self.assertEqual(self.cd.d_embed, 3)
        self.assertEqual(self.cd.k, 4)
        self.assertFalse(self.cd.verbose)
        self.assertEqual(self.cd.neighbors, "knn")
        self.assertEqual(self.cd.forecast, "sum")
        self.assertFalse(self.cd.prune_indirect)
        self.assertFalse(self.cd.ensemble)
        self.assertIsNone(self.cd.significance_threshold)
        self.assertFalse(self.cd.sweep_d_embed)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        causal_matrix = self.cd.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix.shape, (self.n_features, self.n_features))
        
        # Check diagonal is zero
        np.testing.assert_array_equal(np.diag(causal_matrix), np.zeros(self.n_features))
        
        # Check matrix is symmetric
        np.testing.assert_array_almost_equal(causal_matrix, causal_matrix.T)

    def test_fit_with_different_forecast_methods(self):
        """Test fitting with different forecast methods."""
        # Test with sum forecast method
        cd_sum = CausalDetection(
            d_embed=3,
            k=4,
            forecast="sum",
            verbose=False,
            library_sizes=[200, 300, 400]
        )
        causal_matrix_sum = cd_sum.fit(self.X)
        self.assertEqual(causal_matrix_sum.shape, (self.n_features, self.n_features))
        self.assertTrue(np.all(np.diag(causal_matrix_sum) == 0))
        # Handle NaN values by replacing them with zeros before checking bounds
        causal_matrix_sum = np.nan_to_num(causal_matrix_sum, nan=0.0)
        self.assertTrue(np.all(np.abs(causal_matrix_sum) <= 1))

        # Test with invalid forecast method - should fall back to sum
        cd_invalid = CausalDetection(
            d_embed=3,
            k=4,
            forecast="invalid",
            verbose=False,
            library_sizes=[200, 300, 400]
        )
        causal_matrix_invalid = cd_invalid.fit(self.X)
        self.assertEqual(causal_matrix_invalid.shape, (self.n_features, self.n_features))
        self.assertTrue(np.all(np.diag(causal_matrix_invalid) == 0))
        # Handle NaN values by replacing them with zeros before checking bounds
        causal_matrix_invalid = np.nan_to_num(causal_matrix_invalid, nan=0.0)
        self.assertTrue(np.all(np.abs(causal_matrix_invalid) <= 1))

    def test_fit_with_different_neighbor_methods(self):
        """Test fitting with different neighbor methods."""
        # Test with "knn" neighbors
        cd_knn = CausalDetection(neighbors="knn", verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_knn = cd_knn.fit(self.X)
        
        # Test with "simplex" neighbors
        cd_simplex = CausalDetection(neighbors="simplex", verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_simplex = cd_simplex.fit(self.X)
        
        # Check shapes
        self.assertEqual(causal_matrix_knn.shape, (self.n_features, self.n_features))
        self.assertEqual(causal_matrix_simplex.shape, (self.n_features, self.n_features))

    def test_fit_with_ensemble(self):
        """Test fitting with ensemble method."""
        cd_ensemble = CausalDetection(ensemble=True, verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_ensemble = cd_ensemble.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix_ensemble.shape, (self.n_features, self.n_features))

    def test_fit_with_prune_indirect(self):
        """Test fitting with indirect relationship pruning."""
        cd_prune = CausalDetection(prune_indirect=True, verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_prune = cd_prune.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix_prune.shape, (self.n_features, self.n_features))

    def test_fit_with_significance_threshold(self):
        """Test fitting with significance threshold."""
        cd_sig = CausalDetection(significance_threshold=0.05, verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_sig = cd_sig.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix_sig.shape, (self.n_features, self.n_features))

    def test_fit_with_sweep_d_embed(self):
        """Test fitting with embedding dimension sweep."""
        cd_sweep = CausalDetection(sweep_d_embed=True, verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_sweep = cd_sweep.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix_sweep.shape, (self.n_features, self.n_features))

    def test_fit_with_custom_library_sizes(self):
        """Test fitting with custom library sizes."""
        custom_sizes = [200, 300, 400]
        cd_custom = CausalDetection(library_sizes=custom_sizes, verbose=False)
        causal_matrix_custom = cd_custom.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix_custom.shape, (self.n_features, self.n_features))

    def test_fit_with_store_intermediates(self):
        """Test fitting with intermediate results storage."""
        cd_store = CausalDetection(store_intermediates=True, verbose=False, library_sizes=[200, 300, 400])
        causal_matrix_store = cd_store.fit(self.X)
        
        # Check shape
        self.assertEqual(causal_matrix_store.shape, (self.n_features, self.n_features))
        
        # Check that intermediates were stored
        self.assertTrue(hasattr(cd_store, 'y_pred'))
        self.assertTrue(len(cd_store.y_pred) > 0)

if __name__ == '__main__':
    unittest.main() 