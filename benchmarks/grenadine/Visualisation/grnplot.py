import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
import networkx as nx
import os

class GRNplot:
    """
    Class to plot gene regulatory networks

    Args:
        G (networkx.Graph): The gene regulatory network
        X_sc (anndata.AnnData): The single cell data object

    Examples:
        >>> G = nx.read_gpickle("my_network.gpickle")
        >>> X_sc = sc.read("my_data.h5ad")
        >>> grn_plot = GRNplot(G, X_sc)
        >>> grn_plot.pyvis_network_plot()

    """
    def __init__(self, G, X_sc):
        self.G = G
        self.X_sc = X_sc
    
    def _select_genes(self,column=None):
        """
        Select genes from the single cell data object based on a column

        Args:
            column (str): The column to select genes from

        Returns:
            list: A list of selected genes
        """
        if column in list(self.X_sc.var.columns):
            self.X_sc.var[column] = self.X_sc.var[column].astype(bool)
            mask = self.X_sc.var[column]
            return list(mask[mask].index)
        _ = os.write(1,bytes(f"No column '{column}' detected, assumed all genes were TFs \n",'utf-8'))
        return list(self.X_sc.var.index)
    
    def _make_titles(self, description_label,G=None): 
        """
        Make titles for the nodes in the network plot

        Args:
            description_label (str): The column in the single cell data object to use as the node title
            G (networkx.Graph, optional): The gene regulatory network. Defaults to None.

        Returns:
            list: A list of titles for the nodes
        """
        if G is None:
            G = self.G 
        titles = []
        for n in G.nodes():
            if description_label in list(self.X_sc.var.columns):
                r = self.X_sc.var.loc[n,description_label]
                if type(r) != str:
                    r = list(r)[0]
            else: 
                r = n
            titles.append(f"{r}")
        return titles
    
    def _color_mapper(self, x):
        """
        Map a score to a color

        Args:
            x (float): The score to map

        Returns:
            dict: A dictionary mapping scores to colors
        """
        x_r = np.round(x,1)
        if x_r < -0.5:
            return "#0648a7"
        if x_r > 0.5:
            return '#780505'
        colors = {0.:"#e4f7f4f4e8ee",
                  -0.1:"#c9d6e8",
                  -0.2:"#abc2e3",
                  -0.3:"#82a6d9",
                  -0.4:"#608dce",
                  -0.5:'#2764bc',
                  0.1:"#e4bebe",
                  0.2:"#d08888",
                  0.3:"#b75353",
                  0.4:"#aa3333",
                  0.5:"#9c1e1e",}
        return colors[x_r]
    
    def networkx_network_plot(self):
        """
        Plot the gene regulatory network using networkx

        Returns:
            None: Displays a plot of the gene regulatory network
        """
        pos = nx.drawing.layout.kamada_kawai_layout(self.G,scale=3)
        plt.figure(figsize=(30,30))
        nx.draw_networkx(self.G,pos=pos, with_labels=True)

        
    def pyvis_network_plot(self,
                   threshold=0.,
                   save_path='my_figure.html',
                   tf_columns="is TF",
                   score_label="mean_test_score",
                   description_label=None,
                   in_notebook=True):
        
        """
        Plot a gene regulatory network using pyvis

        Args:
            threshold (float, optional): Threshold to select edges. Defaults to 0.5.
            save_path (str, optional): HTML file save path. Defaults to 'my_figure.html'.
            tf_columns (list, optional): The name of the columns where the info for a gene to be a TF is kept. Defaults to ["is TF"].
            score_label (str, optional): The name of the weights column in the graph edges list. Defaults to "mean_test_score".
            description_label (_type_, optional): The labels for the titles. Defaults to None.
            in_notebook (bool, optional): Option to display the network directly if the user is using a notebook format. Defaults to True.

        Returns:
            None: Saves the network plot as an HTML file
        """
        G = self.G.copy()
        try:
            G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data=score_label) if np.abs(w) < threshold])
            G.remove_nodes_from(list(nx.isolates(G)))
        except KeyError:
            print(f"Edge attribute {score_label} not found in G")
            return None
        
        nodes = list(G.nodes)
        if tf_columns is not None:
            tfs = self._select_genes(tf_columns)
            colors = ['#5479c1' if g not in tfs else '#9c2d15' for g in nodes]
            shape = ['polygon' if g not in tfs else 'box' for g in nodes]
        else:
            colors = ['#5479c1' for g in nodes]
            shape = ['polygon' for g in nodes]
        labels = nodes
        values = [G.degree(n) for n in nodes]
        titles = self._make_titles(description_label,G)
        
        self.net = Network(notebook=in_notebook,
                          font_color="white",
                          select_menu=True,
                          filter_menu=True,
                          neighborhood_highlight= True,
                          cdn_resources='in_line'
                          )
        
        self.net.add_nodes(nodes,
                           title=titles,
                           label=labels,
                           color=colors,  
                           value=values,
                           shape=shape)
        
        self.net.options.set('''options ={
                                          "physics": {
                                                      "barnesHut": {
                                                      "theta": 0.35,
                                                      "gravitationalConstant": -5550
                                                     },
                                          "minVelocity": 0.75
                                          }
                            }''')
        
        for e1,e2,data in G.edges(data=True):
            color = self._color_mapper(data[score_label])
            self.net.add_edge(e1, 
                              e2, 
                              value=np.round(data[score_label],3)/10,
                              score=data[score_label],
                              color=color,
                              arrows="to",
                              title=str(np.round(data[score_label],2)))

        self.net.force_atlas_2based(gravity=-20, central_gravity=0.0, spring_length=100, spring_strength=0.08, damping=0.9, overlap=0)

        self.net.toggle_physics(True)
        self.net.show_buttons(filter_=['physics'])
        self.net.show(save_path)
