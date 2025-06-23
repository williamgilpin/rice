import itertools
import subprocess
import pandas as pd
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from models import LaggedCorrelations

def run_deepsem(X, num_replicates=10):
    """
    Convert a data matrix into a format that DeepSEM can process, and run the 
    method on the data

    Args:
        X (np.ndarray): A data matrix with shape (n_genes, n_samples)
        num_replicates (int): The number of replicates to run

    Returns:
        np.ndarray: A matrix of scores for each gene-gene pair
    """ 
    Xpd = pd.DataFrame(X.T)
    gene_names = ["G" + str(i).zfill(4) for i in range(X.shape[1])]
    ## make a list of all combinations of gene pairs
    cell_names = ["Cell" + str(i).zfill(4) for i in range(X.shape[0])]
    # set row index to gene names
    Xpd.index = gene_names
    Xpd.columns = cell_names
    Xpd.transpose().to_csv("dump/data2.csv")

    ## Prior is top 20% of links by Spearman correlation
    local_models_path = os.path.join(os.getcwd(), 'dygene', 'benchmarks', 'models.py')
    if local_models_path not in sys.path:
        sys.path.insert(0, local_models_path)

    model = LaggedCorrelations(method="spearman", max_lag=0.3)
    cmat = model.fit(X)
    cutoff = int(0.2 * cmat.size)
    top_inds_flat = np.argsort(np.ravel(cmat))[::-1][:cutoff]
    top_inds = np.unravel_index(top_inds_flat, cmat.shape)
    gene_pairs = list(zip(top_inds[0], top_inds[1]))
    gene_pairs = [(gene_names[i], gene_names[j]) for i, j in gene_pairs]


    ## Prior is all genes
    # gene_pairs = list(itertools.product(gene_names, gene_names))


    # Add header "Gene1" and "Gene2" and save to CSV
    gene_pairs = [item for item in gene_pairs if item[0] != item[1]] # No self-loops
    title = ["Gene1", "Gene2"]
    gene_pairs.insert(0, title)
    gene_pairs = np.array(gene_pairs)
    gene_pairs = gene_pairs
    np.savetxt("dump/label2.csv", gene_pairs, fmt="%s", delimiter=",")

    command_str = "python DeepSEM/main.py --task non_celltype_GRN --data_file dump/data2.csv"
    command_str += " --net_file dump/label2.csv --setting default --save_name dump"
    all_cmat = list()
    for _ in range(num_replicates):
        # print("Running DeepSEM")
        subprocess.run(command_str.split(" "), capture_output=True, text=True)
        
        file_path = "dump/GRN_inference_result.tsv"
        df = pd.read_csv(file_path, sep='\t', header=None, names=['TF', 'Target', 'EdgeWeight'], skiprows=1)
        # nodes = sorted(set(df['TF']).union(set(df['Target'])))
        # node_to_index = {node: i for i, node in enumerate(nodes)}
        # D2 = len(nodes)
        # if D2 > D:
        #     warnings.warn(f"Number of nodes in the output ({N2}) is greater than the number of nodes in the input ({N}).")

        adj_matrix = np.zeros((X.shape[1], X.shape[1]))
        for _, row in df.iterrows():
            # i, j = node_to_index[row['TF']], node_to_index[row['Target']]
            i, j = int(row['TF'][1:]), int(row['Target'][1:])
            adj_matrix[i, j] = row['EdgeWeight']

        all_cmat.append(adj_matrix.copy())
    all_cmat = np.array(all_cmat)
    cmat = np.mean(all_cmat, axis=0)
    return cmat