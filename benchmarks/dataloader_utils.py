import os
import sys
import glob
# import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

# import itertools
# from itertools import product


import requests
import pandas as pd


def map_genes_to_string_ids(genes, species=9606):
    """
    Given a list of gene names, maps them to STRING IDs using the STRING API.

    Args:
        genes (list): A list of gene names.
        species (int): The NCBI Taxonomy ID for the species. Default is 9606 (human).

    Returns:
        dict: A dictionary mapping gene names to STRING IDs.
    """

    mapping_url = "https://string-db.org/api/json/get_string_ids"
    string_ids = {}
    for gene in genes:
        params = {
            "identifiers": gene,
            "species": species,
            "limit": 1  # get the best match
        }
        response = requests.get(mapping_url, params=params)
        data = response.json()
        if data:
            # Use the first (best) mapping result
            string_ids[gene] = data[0]["stringId"]
        else:
            # If mapping fails, record None
            string_ids[gene] = None
    return string_ids

def fetch_interaction_matrix(gene_names, species=9606,  min_confidence=0, max_requests=300):
    """
    Given a list of genes, fetches a protein-protein interaction matrix from the STRING 
    database.

    Args:
        gene_names (list): A list of gene names.
        species (int): The NCBI Taxonomy ID for the species. Default is 9606 (human).
        min_confidence (int): The minimum confidence score for interactions. 0 means all
            interactions are included.
        max_requests (int): The maximum number of simultaneous requests to make to the
            STRING API.

    Returns:
        DataFrame: A square DataFrame with gene names as both rows and columns, and
            interaction scores as values.
    """

    string_ids = map_genes_to_string_ids(gene_names, species=species)
    mapped_genes = {gene: sid for gene, sid in string_ids.items() if sid is not None}

    if not mapped_genes:
        raise ValueError("None of the gene names could be mapped to STRING IDs.")

    network_url = "https://string-db.org/api/json/network"
    identifiers = "%0d".join(mapped_genes.values())
    params = {
        "identifiers": identifiers,
        "species": species,
        "required_score": min_confidence
    }

    # response = requests.get(network_url, params=params)
    response = requests.post(network_url, data=params)
    response.raise_for_status()
    interactions = response.json()

    # Create an empty square DataFrame with gene names as both rows and columns,
    # and initialize with NaN for unknown interactions.
    matrix = pd.DataFrame(index=gene_names, columns=gene_names, dtype=float)
    matrix[:] = float('nan')
    inverse_mapping = {sid: gene for gene, sid in mapped_genes.items()}

    ## Populate the matrix with interaction scores. The STRING API returns each edge 
    ## with stringId_A, stringId_B, and a score (among other details).
    for interaction in interactions:
        sid_a = interaction.get("stringId_A")
        sid_b = interaction.get("stringId_B")
        score = interaction.get("score")
        gene_a = inverse_mapping.get(sid_a)
        gene_b = inverse_mapping.get(sid_b)
        if gene_a and gene_b:
            matrix.loc[gene_a, gene_b] = score
            matrix.loc[gene_b, gene_a] = score  # ensure symmetry

    ## Add self-interactions
    for gene in gene_names:
        matrix.loc[gene, gene] = 1.0

    return matrix