import unittest
import numpy as np
from rice.examples import ecoli100, yeast100
from rice.models import CausalDetection
from rice.metrics import compute_metrics

class TestData(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(0)

    def test_ecoli100(self):
        """Test ecoli100 function."""
        X, amat = ecoli100()
        model = CausalDetection(
            d_embed=3, 
            neighbors="simplex", 
            forecast="smap", 
            ensemble=True, 
        )
        model.ensemble = True
        cmat = model.fit(X)
        scores_dense = compute_metrics(amat, cmat, verbose=True, check_transpose=False)
        self.assertGreater(scores_dense["AUPRC Multiplier"], 30)
        self.assertGreater(scores_dense["ROC-AUC Multiplier"], 1)
            
    def test_yeast100(self):
        """Test yeast100 function."""
        X, amat = yeast100()
        model = CausalDetection(
            d_embed=3, 
            neighbors="simplex", 
            forecast="smap", 
            ensemble=True, 
        )
        model.ensemble = True
        cmat = model.fit(X)
        scores_dense = compute_metrics(amat, cmat, verbose=True, check_transpose=False)
        self.assertGreater(scores_dense["AUPRC Multiplier"], 20)
        self.assertGreater(scores_dense["ROC-AUC Multiplier"], 1)


if __name__ == '__main__':
    unittest.main() 