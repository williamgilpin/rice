import glob
import pandas as pd
import os
import gseapy

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="gseapy")
# import logging
# logging.getLogger("gseapy").setLevel(logging.ERROR)

from pathlib import Path

ontology_path_root = Path(__file__).resolve().parent

def run_enrichr(all_gene_groups, gmt_path=None):
    """
    
    curl -L 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2026' -o GO_Biological_Process_2026.gmt
    
    """

    all_df = list()
    all_community_indices = list()

    for fpath in glob.glob("test/*reports.txt"):
        os.remove(fpath)

    if gmt_path is None:
        gmt_path = ontology_path_root / "gene_ontologies" / "GO_Biological_Process_2026.gmt"
    
    # print(f"Using GMT file: {gmt_path}")
    for i, gl in enumerate(all_gene_groups):
        try:
            gseapy.enrichr(gene_list=list(gl), 
                        gene_sets=[str(gmt_path)],
                        # gene_sets=[
                        #     gmt_path,
                        #     #    "./data/gene_ontologies/Azimuth_2023.gmt",
                        #     # os.path.join(ontology_path_root, "Azimuth_Cell_Types_2021.gmt"),
                        #     # "./data/gene_ontologies/GO_Biological_Process_2025.gmt",
                        #         # "./data/gene_ontologies/Reactome_Pathways_2024.gmt",
                        #     ], 
                        outdir='test', 
                        no_plot=True,
                        cutoff=0.05)
            fpaths = glob.glob("test/*reports.txt")
            if len(fpaths) > 0:
                df = pd.read_csv(fpaths[0], sep="\t", header=0)
                ## sort by adjusted p-value
                df = df.sort_values(by="Adjusted P-value", ascending=True)
            else:
                df = pd.DataFrame()
            all_df.append(df)
            all_community_indices.append(i)
        except ValueError as e:
            print(f"Error processing gene group {gl}: {e}")

        for fpath in glob.glob("test/*reports.txt"):
            os.remove(fpath)

    return all_df, all_community_indices