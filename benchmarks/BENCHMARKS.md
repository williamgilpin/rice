# Benchmarks

A set of benchmarks for inferring interaction networks from time series data. This library contains data loaders for the following datasets:

+ DREAM4, a set of stochastic kinetic simulations of gene regulatory networks
+ TWIST, a reimplementation of DREAM4 with intrinsic nonlinear dynamics
+ Smoketest, a new trivial dataset with simple correlation structure of varying amplitude
+ BEELINE, a set of developmental single-cell RNA-seq datasets with pseudotime
+ McCalla, a set of developmental single-cell RNA-seq datasets with pseudotime

### Installation

Dependencies are managed using conda or mamba. All external packages have been downloaded and are called locally, and so the only internal dependencies are the standard dependencies `scikit-learn`, `scipy`, `numpy`, `yellowbrick`, `POT`

If you are using mamba, then the following command will create an environment `gene` with the necessary dependencies:

    ./mamba_install.sh

This script assumes that you have working [mamba installation](https://mamba.readthedocs.io/en/latest/) on your system. It will create an environment `gene` and install the necessary dependencies. If `gene` already exists, then the script exits.

### Running benchmarks

The benchmarks can be run by executing the following command in the `dygene/benchmarks` directory

    python run_benchmarks.py --dataset <dataset_name>

Where `<dataset_name>` is one of the following:
+ `dream4` is the DREAM4 dataset
+ `twist` is the TWIST dataset, a modified version of DREAM4 with intrinsic nonlinear dynamics
+ `smoketest` is the Smoketest dataset, a trivial dataset with simple correlation structure of varying amplitude
+ `beeline` is the BEELINE dataset, a set of developmental single-cell RNA-seq datasets with pseudotime
+ `mccalla` is the McCalla dataset, a set of developmental single-cell RNA-seq datasets with pseudotime
+ `kuramoto` is the Kuramoto dataset, a physical nonlinear dynamical system with a known interaction network

The output will be saved in the `benchmark_output` directory. Each benchmark will produce a separate output file. The output files will be named according to the benchmark name, the dataset name, and the condition. Replicates of the same experiment will be saved in the same file.


### Attribution

To minimize dependencies and installer conflicts, this directory contains local versions of several external libraries. Some modifications have been made to these libraries, in order to ensure compatibility within a single environment. The original versions of these libraries are linked here; please cite the relevant papers if using these parts of the benchmarking code

    + [GReNaDIne](https://gitlab.com/bf2i/grenadine)
    + Genie3
    + dynGENIE3
    + NetworkInference.jl
    + DeepSEM
    + SWING
