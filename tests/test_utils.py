import unittest
import numpy as np
from hccm.utils import (
    mask_topk, hankel_matrix, batch_diag, embed_ts, multivariate_embed_ts,
    project_pca, batch_pca, embed_ts_pca, batch_sfa, embed_ts_sfa,
    batch_pearson, batch_spearman, progress_bar, flatten_along_axis,
    max_linear_correlation_ridge, banded_matrix, hollow_matrix
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        self.test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.test_time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.test_multivariate = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    def test_mask_topk(self):
        """Test mask_topk function."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        masked = mask_topk(arr, k=3)
        self.assertEqual(np.sum(masked), 3)
        self.assertTrue(np.all(masked[2, 2] == 1))  # 9 should be masked
        self.assertTrue(np.all(masked[2, 1] == 1))  # 8 should be masked
        self.assertTrue(np.all(masked[2, 0] == 1))  # 7 should be masked

    def test_hankel_matrix(self):
        """Test hankel_matrix function."""
        data = np.array([1, 2, 3, 4, 5])
        hmat = hankel_matrix(data, q=3, p=2)
        # The function returns a 3D array with shape (2, 3, 1)
        self.assertEqual(hmat.shape, (2, 3, 1))
        # Check the values in the first dimension
        np.testing.assert_array_equal(hmat[0, :, 0], [1, 2, 3])
        np.testing.assert_array_equal(hmat[1, :, 0], [2, 3, 4])

    # def test_batch_diag(self):
    #     """Test batch_diag function."""
    #     arr = np.array([[1, 2], [3, 4]])
    #     result = batch_diag(arr, axis=0)
    #     # The function returns a 3D array with shape (2, 2, 2)
    #     self.assertEqual(result.shape, (2, 2, 2))
    #     # Check the diagonal elements
    #     np.testing.assert_array_equal(result[0, 0, 0], 1)
    #     np.testing.assert_array_equal(result[0, 1, 0], 0)
    #     np.testing.assert_array_equal(result[1, 0, 0], 0)
    #     np.testing.assert_array_equal(result[1, 1, 0], 3)
    #     np.testing.assert_array_equal(result[0, 0, 1], 0)
    #     np.testing.assert_array_equal(result[0, 1, 1], 2)
    #     np.testing.assert_array_equal(result[1, 0, 1], 0)
    #     np.testing.assert_array_equal(result[1, 1, 1], 4)

    def test_embed_ts(self):
        """Test embed_ts function."""
        X = np.array([1, 2, 3, 4, 5])
        embedded = embed_ts(X, m=3)
        # The function returns a 3D array with shape (1, 3, 3)
        self.assertEqual(embedded.shape, (1, 3, 3))
        # Check the values in the first batch
        np.testing.assert_array_equal(embedded[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(embedded[0, 1], [2, 3, 4])
        np.testing.assert_array_equal(embedded[0, 2], [3, 4, 5])

    def test_multivariate_embed_ts(self):
        """Test multivariate_embed_ts function."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        embedded = multivariate_embed_ts(X, m=2)
        self.assertEqual(embedded.shape, (3, 4))  # 3 timepoints, 2 dimensions * 2 embedding

    def test_project_pca(self):
        """Test project_pca function."""
        X = np.random.randn(100, 10)
        projected = project_pca(X, k=2)
        self.assertEqual(projected.shape, (100, 2))

    def test_batch_pca(self):
        """Test batch_pca function."""
        X = np.random.randn(5, 100, 10)  # 5 batches, 100 timepoints, 10 features
        eivals, eivecs = batch_pca(X)
        self.assertEqual(eivals.shape, (5, 10))
        self.assertEqual(eivecs.shape, (5, 10, 10))

    def test_embed_ts_pca(self):
        """Test embed_ts_pca function."""
        X = np.random.randn(100, 5)  # 100 timepoints, 5 features
        embedded = embed_ts_pca(X, m=3)
        # The function returns a 3D array with shape (5, 98, 3)
        self.assertEqual(embedded.shape, (5, 98, 3))

    # def test_batch_sfa(self):
    #     """Test batch_sfa function."""
    #     # Create a simple test case with known structure
    #     X = np.zeros((2, 10, 4))  # 2 batches, 10 timepoints, 4 features
    #     # Create a simple oscillating pattern
    #     t = np.linspace(0, 2*np.pi, 10)
    #     X[0, :, 0] = np.sin(t)  # Slow feature
    #     X[0, :, 1] = np.sin(2*t)  # Faster feature
    #     X[0, :, 2] = np.sin(3*t)  # Even faster
    #     X[0, :, 3] = np.sin(4*t)  # Fastest
    #     # Copy to second batch with some noise
    #     X[1] = X[0] + 0.1 * np.random.randn(*X[0].shape)
    #     # Add small regularization to each feature
    #     X = X + 1e-3 * np.ones_like(X)
    #     # Center the data
    #     X = X - np.mean(X, axis=1, keepdims=True)
    #     # Add a small amount to the diagonal of the covariance matrix
    #     X = X + 1e-3 * np.eye(X.shape[-1])[None, None, :]
    #     # Compute SFA
    #     S, W, E = batch_sfa(X, num_features=2, return_transform=True)
    #     self.assertEqual(S.shape, (2, 10, 2))
    #     self.assertEqual(W.shape, (2, 4, 2))
    #     self.assertEqual(E.shape, (2, 2))
    #     # Check that the output is real-valued
    #     self.assertTrue(np.all(np.isreal(S)))
    #     self.assertTrue(np.all(np.isreal(W)))
    #     self.assertTrue(np.all(np.isreal(E)))

    def test_embed_ts_sfa(self):
        """Test embed_ts_sfa function."""
        X = np.random.randn(100, 5)  # 100 timepoints, 5 features
        embedded = embed_ts_sfa(X, m=3)
        # The function returns a 3D array with shape (5, 98, 3)
        self.assertEqual(embedded.shape, (5, 98, 3))

    def test_batch_pearson(self):
        """Test batch_pearson function."""
        x = np.random.randn(5, 100, 10)  # 5 batches, 100 timepoints, 10 features
        y = np.random.randn(5, 100, 10)
        corr = batch_pearson(x, y)
        self.assertEqual(corr.shape, (5, 100))
        self.assertTrue(np.all(np.abs(corr) <= 1))

    def test_batch_spearman(self):
        """Test batch_spearman function."""
        x = np.random.randn(5, 100, 10)  # 5 batches, 100 timepoints, 10 features
        y = np.random.randn(5, 100, 10)
        corr = batch_spearman(x, y)
        self.assertEqual(corr.shape, (5, 100))
        self.assertTrue(np.all(np.abs(corr) <= 1))

    def test_flatten_along_axis(self):
        """Test flatten_along_axis function."""
        arr = np.random.randn(3, 4, 5)
        flattened = flatten_along_axis(arr, axes=(1, 2))
        self.assertEqual(flattened.shape, (3, 20))  # 4*5 = 20

    def test_max_linear_correlation_ridge(self):
        """Test max_linear_correlation_ridge function."""
        A = np.random.randn(100, 5)  # 100 timepoints, 5 features
        B = np.random.randn(100, 3)  # 100 timepoints, 3 features
        corr = max_linear_correlation_ridge(A, B)
        self.assertEqual(corr.shape, (3,))
        self.assertTrue(np.all(np.abs(corr) <= 1))

    def test_banded_matrix(self):
        """Test banded_matrix function."""
        n, r = 5, 2
        matrix = banded_matrix(n, r)
        self.assertEqual(matrix.shape, (n, n))
        self.assertTrue(np.all(matrix[0, :3] == 1))  # First row should have 3 ones
        self.assertTrue(np.all(matrix[0, 3:] == 0))  # Rest should be zeros

    def test_hollow_matrix(self):
        """Test hollow_matrix function."""
        n, r = 5, 2
        matrix = hollow_matrix(n, r)
        self.assertEqual(matrix.shape, (n, n))
        self.assertTrue(np.all(matrix[0, :3] == 0))  # First row should have 3 zeros
        self.assertTrue(np.all(matrix[0, 3:] == 1))  # Rest should be ones

if __name__ == '__main__':
    unittest.main() 