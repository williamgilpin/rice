# Benchmarks

A set of benchmarks for inferring interaction networks from time series data. This library contains data loaders for the following datasets:

+ DREAM4, a set of stochastic kinetic simulations of gene regulatory networks
+ TWIST, a reimplementation of DREAM4 with intrinsic nonlinear dynamics
+ Smoketest, a new trivial dataset with simple correlation structure of varying amplitude
+ BEELINE, a set of developmental single-cell RNA-seq datasets with pseudotime
+ McCalla, a set of developmental single-cell RNA-seq datasets with pseudotime

### Installation

Dependencies are managed using conda or mamba. All external packages have been downloaded and are called locally, and so the only internal dependencies are the standard dependencies `scikit-learn`, `scipy`, `numpy`, `yellowbrick`, `POT`

If you are using mamba, then the following command will create an environment `gene` with the necessary Pythondependencies:

    ./mamba_install.sh

This script assumes that you have working [mamba installation](https://mamba.readthedocs.io/en/latest/) on your system. It will create an environment `gene` and install the necessary dependencies. If `gene` already exists, then the script exits.

Several benchmark models use [`NetworkInference.jl`](https://github.com/Tchanders/NetworkInference.jl), and so you may need to install Julia. The following command will install Julia:

```bash
    curl -fsSL https://install.julialang.org | sh
```

### Retrieving the datasets

The benchmark datasets are hosted externally. Download each one from the website listed below and place the files in the `benchmarks/benchmark_datasets/` directory (create it if it does not exist), using the file names and folders given here. The total size is approximately 3GB.

**TWIST and Smoketest.** These synthetic datasets are hosted on [Hugging Face](https://huggingface.co/datasets/williamgilpin/grn). To download them, install the Hugging Face CLI and run the fetch script from within the `benchmarks` directory:

```bash
pip install -U "huggingface_hub[cli]"
bash fetch_datasets.sh
```

This places the `gene_expression` and `smoketest` folders in `benchmark_datasets/`, along with the files described below.

**Kuramoto.** This dataset is included with the `rice` package. Copy `rice/data/kuramoto100_X.csv.gz` and `rice/data/kuramoto100_adjmat.csv.gz` into `benchmark_datasets/`.

**BEELINE** (Pratapa et al., *Nature Methods* 2020; licensed CC BY-NC 4.0).

+ We used the [packaged BEELINE dataset](https://bcb.cs.tufts.edu/software.html) distributed and hosted by the Slonim group. Unzip the data and place the `BEELINE` folder at `benchmark_datasets/BEELINE/`. 
+ For pseudotime, download the `PseudoTime.csv` files from [the original BEELINE paper](https://www.nature.com/articles/s41592-019-0690-6). Place each pseudotime file in `benchmark_datasets/BEELINE/` and rename it appropriately. For example, `scRNA-Seq/hESC/PseudoTime.csv` becomes `BEELINE/hESC_PseudoTime.csv`.

**McCalla**. Download `expression_data.zip` and `gold_standard_datasets.zip` from [McCalla et al., *G3* 2023](https://academic.oup.com/g3journal/article-abstract/13/3/jkad004/6982776).

+ From `expression_data.zip` place the `imputed` folder at `benchmark_datasets/mccalla/imputed/`.
+ Rename `normalized/zhao_GSE114952.csv.gz` to `mESC_zhao_GSE114952.csv.gz` in `benchmark_datasets/mccalla/normalized/`.
+ From `gold_standard_datasets.zip`, place the `gold_standards` folder at `benchmark_datasets/mccalla/gold_standards/`.

**DREAM4**. Download the original datasets from [Marbach et al., *PNAS* 2010](https://www.pnas.org/doi/10.1073/pnas.0913357107). Download the two InSilico challenge datasets (10 gene and 100 gene) with their accompanying gold standard datasets. Place the files in `benchmark_datasets/` as follows:

```
benchmark_datasets/
    DREAM4_InSilico_Size10/insilico_size10_<i>/insilico_size10_<i>_timeseries.tsv
    DREAM4_InSilico_Size100/insilico_size100_<i>/insilico_size100_<i>_timeseries.tsv
    DREAM4_Challenge2_GoldStandards/Size 10/DREAM4_GoldStandard_InSilico_Size10_<i>.tsv
    DREAM4_Challenge2_GoldStandards/Size 100/DREAM4_GoldStandard_InSilico_Size100_<i>.tsv
```

### Running benchmarks

The benchmarks can be run by executing the following command in the `dygene/benchmarks` directory

    python run_benchmarks.py --dataset <dataset_name> --model <method_name>

Where `<dataset_name>` is one of the following:
+ `dream4` is the DREAM4 dataset
+ `twist` is the TWIST dataset, a modified version of DREAM4 with intrinsic nonlinear dynamics
+ `smoketest` is the Smoketest dataset, a trivial dataset with simple correlation structure of varying amplitude
+ `beeline` is the BEELINE dataset, a set of developmental single-cell RNA-seq datasets with pseudotime
+ `mccalla` is the McCalla dataset, a set of developmental single-cell RNA-seq datasets with pseudotime
+ `kuramoto` is the Kuramoto dataset, a physical nonlinear dynamical system with a known interaction network

For the BEELINE and McCalla datasets, the `--goldtype` flag selects which gold standard network(s) to score against. One or more may be given. For BEELINE, the options are `STRING` (default), `ChIP-seq`, and `Non-ChIP`. For McCalla, the options are `chipunion`, `KDUnion`, and `chipunion_KDUnion_intersect` (default). For example,

    python run_benchmarks.py --dataset beeline --model ensemble_noprune --goldtype ChIP-seq Non-ChIP
    python run_benchmarks.py --dataset mccalla --model ensemble_noprune --goldtype chipunion

The output will be saved in the `benchmark_output` directory. Each benchmark will produce a separate output file. The output files will be named according to the benchmark name, the dataset name, and the condition. Replicates of the same experiment will be saved in the same file. For the BEELINE and Kuramoto datasets, the reconstructed network matrices are also saved, in the `benchmark_output/matrices` directory (excluded from version control), so that new metrics can be computed later without re-running the benchmarks.

The currently supported options for `<method_name>` are:

+ `ensemble_prune` is the full ensemble RiCEmethod with pruning
+ `ensemble_noprune` is the full ensemble RiCE method without pruning
+ `isolated_noprune` is the isolated RiCE method without pruning
+ `ccm` is the cross-mapping method
+ `smap` is the S-map method
+ `regdiffusion` is the diffusion regression method
+ `deepsem` 
+ `swing` is the SWING method with random forest models
+ `wcorr` is simple partial correlations
+ `scode` is the SCODE method
+ `mi` uses mutual information
+ `clr` 
+ `puc`
+ `pidc`
+ `dyngenie3` 
+ `grenadine_clr` is an alternative implementation of CLR in the GReNaDIne library
+ `grenadine_genie3` is an alternative implementation of GENIE3 in the GReNaDIne library
+ `bayesian_ridge` uses Bayesian ridge regression
+ `svr` uses support vector regression
+ `tigress` 
+ `elastica` uses ElasticNet regression
+ `adaboost`
+ `grnboost2`
+ `xgenie3`
+ `wasserstein_gren` is the Wasserstein distance calculation
+ `energy` is the energy distance
+ `wilcoxon` is the Wilcoxon distance
+ `glasso` is the Graphical Lasso method
+ `genie3` 
+ `sincerities`
+ `aracne`





### Attribution

To minimize dependencies and installer conflicts, this directory contains local versions of several external libraries. Some modifications have been made to these libraries, in order to ensure compatibility within a single environment. The original versions of these libraries are linked here; please cite the relevant papers if using these parts of the benchmarking code

    + [GReNaDIne](https://gitlab.com/bf2i/grenadine)
    + Genie3
    + dynGENIE3
    + NetworkInference.jl
    + DeepSEM
    + SWING
