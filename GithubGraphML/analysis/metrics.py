from graph_tool.all import *
import matplotlib.pyplot as plt
import numpy as np


def analyze_node_languages(graph: Graph) -> dict:
    """
    Analyzes the given graph and identifies the languages nodes are associated with
    (defined by the value of an edge's 'language' property). Assumes the graph is
    undirected.

    Args:
        graph (Graph): The graph-tool graph object to analyze.

    Returns:
        dict: A dictionary mapping node IDs to lists of languages for nodes
              associated with two or more languages.
    """
    lang_prop = graph.ep["language"]
    id_prop = graph.vp["id"]
    lang_dict = dict()

    for edge in graph.edges():
        lang = lang_prop[edge]
        v1 = id_prop[edge.source()]
        v2 = id_prop[edge.target()]
        if v1 not in lang_dict:
            lang_dict[v1] = set()
        if v2 not in lang_dict:
            lang_dict[v2] = set()

        lang_dict[v1].add(lang)
        lang_dict[v2].add(lang)

    return {k: tuple(v) for k, v in lang_dict.items()}


def classic_metrics(graph: Graph, ddplot='degree_distribution.png') -> dict:
    component = extract_largest_component(graph)
    n = component.num_vertices()
    m = component.num_edges()
    k = np.mean(component.degree_property_map("total").a)
    cc = np.mean(local_clustering(component).a)
    d = np.sum(shortest_distance(component).get_2d_array()) / (n * (n - 1))
    bc = np.max(betweenness(component)[0].a)
    metrics = {
        'num_vertices': n,
        'num_edges': m,
        'avg_degree': k.item(),
        'clustering_coefficent': cc.item(),
        'avg_shortest_path': d,
        'betweenness_centrality': bc.item(),
    }

    if ddplot is not None:
        degrees, counts = np.unique(component.degree_property_map("total").a, return_counts=True)
        plt.scatter(degrees, counts, color='blue', label='Degree Distribution')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Degree (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Degree Distribution (log-log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(ddplot)
        plt.clf()
    
    return metrics