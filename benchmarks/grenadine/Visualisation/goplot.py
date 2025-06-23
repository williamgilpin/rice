from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import seaborn as sns

class GOplot:
    """
    Class to plot gene ontology enrichment results

    Args:
        go_enrichment (pandas.DataFrame): gene ontology enrichment results

    Examples:
        >>> go_enrichment = pd.read_csv("my_go_enrichment.csv")
        >>> go_plot = GOplot(go_enrichment)
        >>> go_plot.cluster_go_enrichment(n_clusters=5, n_top_genes=5)
        >>> go_plot.plot_go_enrichment()

    """
    def __init__(self, go_enrichment):
        self.go_enrichment = go_enrichment
        self.summ_go_enrichment = None

    def _compute_log_score(self):
        """
        Compute -log10(p_value) for each GO term

        Returns:
            pandas.DataFrame: The input dataframe with an additional column containing -log10(p_value)
        """
        
        self.go_enrichment["-log10(p_value)"] = -np.log10(self.go_enrichment["p_value"])

    def cluster_go_enrichment(self, n_clusters, column = "Intersections"):
        """
        Cluster GO terms based on the similarity of their gene intersections

        Args:
            n_clusters (int): Number of clusters to create
            column (str, optional): The column from the input dataframe which contains the gene intersections. If using the scanpy.queries.enrich method to generate
            gene ontologies, parameter gprofiler_kwargs={'no_evidences':False}) allows for generation of the gene interactions column. Defaults to "Intersections". 

        Returns:
            None: The input dataframe is modified in place with an additional column containing the cluster number
        """

        try:
            genes_in_go = list(self.go_enrichment[column])
        except KeyError:
            print(f"Column {column} not found in go_enrichment dataframe")
            print("Please run sc.queries.enrich with parameter gprofiler_kwargs={'no_evidences':False}) to compute intersections")
            return None
        similarity_go_intra = []
        for i,go1 in enumerate(genes_in_go):
            similarity_go_intra.append([])
            for j,go2 in enumerate(genes_in_go):
                inter = len(set(go1).intersection(go2))
                union = len(set(go1).union(go2))
                similarity_go_intra[i].append( inter / union)
        similarity_go_intra = pd.DataFrame(similarity_go_intra,
                                        index = self.go_enrichment.index,
                                        columns = self.go_enrichment.index)
        if (similarity_go_intra < 1.0).any().any(): 
            ac = AgglomerativeClustering(n_clusters=n_clusters,
                                        metric='correlation',
                                        linkage='average')
            clusters = ac.fit_predict(similarity_go_intra)
            self.go_enrichment["cluster"] = clusters
        else:
            self.go_enrichment["cluster"] = self.go_enrichment.index


    def summarize_go_enrichment(self, n_clusters, n_top_genes, column = "intersections"):
        """
        Summarize the top genes enriched gene ontologies for each cluster

        Args:
            n_clusters (int): Number of clusters to create
            n_top_genes (int): Number of top enriched GO terms to summarize
            column (str, optional): The column from the input dataframe which contains the gene intersections. Defaults to "intersections".

        Returns:
            pandas.DataFrame: A dataframe containing the description of the top n_top_genes enriched gene ontologies for each cluster
        """
        self.cluster_go_enrichment(n_clusters, column)
        enriched = self.go_enrichment
        clusters = enriched["cluster"]
        gb_go = enriched.groupby(clusters)
        summary = []
        for i,group in enumerate(gb_go.groups):
            group_enrichment = gb_go.get_group(group).sort_values("p_value")
            group_enrichment["cluster"] = i
            summary.append(group_enrichment.iloc[:n_top_genes,:])
        self.summ_go_enrichment = pd.concat(summary)
        return self.summ_go_enrichment


    def plot_go_enrichment(self,save=None,cut_comma_name=True,legend=True):
        """
        Plot the gene ontology enrichment results

        Args:
            save (str, optional): Path to save the plot. Defaults to None.
            cut_comma_name (bool, optional): Whether to reformat names depending on their input format. Defaults to True.
            legend (bool, optional): Whether to create a legend for the plot. Defaults to True.
        """
        if self.summ_go_enrichment is not None:
            self.go_enrichment = self.summ_go_enrichment.copy()
        else:
            self.go_enrichment = self.go_enrichment.copy()
        self._compute_log_score()
        clusters = self.go_enrichment["cluster"]
        palette = sns.color_palette("Paired",len(set(clusters)))
        clrs = [palette[i] for i in clusters]
        plt.figure(figsize=(2,len(clusters)//4))
        if cut_comma_name:
            self.go_enrichment["name"] = [n.split(",")[0] for n in self.go_enrichment["name"]]
        sns.barplot(data=self.go_enrichment,
                    y="name",
                    x="-log10(p_value)",
                    palette=clrs,
                )
        if legend:
            handles = [mlines.Line2D([], [], color=p, marker='s', ls='', label=i) for i,p in enumerate(palette)]
            plt.legend(handles=handles,
                    title="Clusters",
                    frameon=False,
                    bbox_to_anchor=(1, 1.03)
                    )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("")
        plt.xlabel("$-log_{10}(P_{value})$",fontsize=15,)
        sns.despine(left=True)
        if save is not None:
            plt.savefig(save, bbox_inches='tight')
        return palette