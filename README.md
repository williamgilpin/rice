# RiCE

The Riemannian Causal Embedding (RiCE) algorithm discovers causal relationships in high-dimensional time series data.

![Overview of method](rice/data/fig_schematic.jpg)

### Example usage

```python
    from rice import CausalDetection
    from rice.examples import ecoli100
    from rice.metrics import compute_metrics

    # Load example time series dataset and ground truth connectivity matrix
    X, adj_true = ecoli100() # shapes (300 timepoints x 100 genes) and (100 genes x 100 genes)

    # Run causal analysis and get predicted causal graph
    model = CausalDetection()
    adj_pred = model.fit_transform(X) # shape (100 genes x 100 genes)

    # Score the predicted graph
    scores = compute_metrics(adj_true, adj_pred)
    print(scores["AUPRC Multiplier"]) # AUPRC Multiplier > 32.0
    print(scores["ROC-AUC Multiplier"]) # ROC-AUC Multiplier > 1.0
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
+ [hnswlib](https://github.com/nmslib/hnswlib)
<!-- + [umap-learn](https://umap-learn.readthedocs.io/en/latest/) -->

The examples and tests require additional dependencies:

+ Scipy
+ Pandas
+ Anndata

### Benchmarks

To install and run the benchmarks, see the full instructions in the [BENCHMARKS.md](./benchmarks/BENCHMARKS.md) file. The benchmarks are run by executing the following command in the `benchmarks` directory

```bash
    python run_benchmarks.py --dataset <dataset_name> --model <method_name>
```

Where `<dataset_name>` is one of the six benchmark datasets: `dream4`, `twist`, `smoketest`, `beeline`, `mccalla`, `kuramoto` while `<method_name>` refers to any combination of the 30 benchmark methods currently supported. These include classical methods like `aracne `, `clr`, `genie3`, modern statistical-learning methods like `deepsem`, `regdiffusion`, and dynamics-based methods like `ccm` or `swing`.

### What do we mean by "Causality"?

Our approach aims to discover weak (observational) causality, in the sense of Granger causality, but generalized for nonlinear dynamical systems. This form of causality is equivalent a discovering a forcing term in a system of coupled differential equations.

We do not discover strong (interventional) causality, in the sense of Pearl's do-calculus, which is impossible without the ability to intervene on the data generator (the experimental system).

### References

If you find this code useful, please consider citing our paper:

```bibtex
@article{krieger2025rice,
  title={Interpretable gene network inference with nonlinear causalitys},
  author={Krieger, Madison Ski and Gilpin, William},
  year={2025}
}
```







