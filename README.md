# RiCE

The Riemannian Causal Embedding (RiCE) algorithm discovers causal relationships in high-dimensional time series data.

### Example usage

```python
    from rice import CausalDetection
    from rice.examples import kinetic10

    # Load example dataset and ground truth connectivity matrix
    X, y_true = kinetics10()

    # Run causal analysis
    model = CausalDetection()
    y_pred = model.fit_transform(X)

    # Compare true and predicted connectivities
```


### Installation

Install directly from GitHub using PyPI

```bash
    pip install git+https://github.com/williamgilpin/rice
```

Check that everything is installed correctly

```bash
    python -m unittest
```


### Requirements

+ Python 3.7+
+ NumPy
+ Scikit-learn
+ SciPy

The examples and demonstrations require additional dependencies:

+ Scipy
+ Pandas
+ Anndata




