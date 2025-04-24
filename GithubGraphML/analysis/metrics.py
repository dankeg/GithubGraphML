from graph_tool.all import *
import matplotlib.pyplot as plt
import numpy as np


lang_map = {
    -1: '',
    0: 'AS',
    1: 'JS',
    2: 'PA',
    3: 'PL',
    4: 'PY',
    5: 'RB',
    6: 'VB',
}

def language_usage(graph: Graph) -> dict:
    """
    identifies the languages that each user has used (defined by the value of
    an edge's 'programming_langauge_id' property). Assumes the graph is
    undirected.

    Args:
        graph (Graph): The graph-tool graph object to analyze.

    Returns:
        dict: A dictionary mapping developer IDs to lists of languages that
              a user has used.
    """
    lang_prop = graph.ep['programming_language_id']
    id_prop = graph.vp['developer_id']
    lang_dict = dict()

    for edge in graph.edges():
        lang = lang_prop[edge]
        v1 = id_prop[edge.source()]
        v2 = id_prop[edge.target()]
        if v1 not in lang_dict:
            lang_dict[v1] = set()
        if v2 not in lang_dict:
            lang_dict[v2] = set()

        lang_dict[v1].add(lang_map[lang])
        lang_dict[v2].add(lang_map[lang])
 
    return {k: tuple(sorted(v)) for k, v in lang_dict.items()}


def classic_metrics(graph: Graph, ddplot_prefix=None) -> dict:
    n = graph.num_vertices()
    m = graph.num_edges()
    k, std = vertex_average(graph, graph.degree_property_map("total"))
    gcc = global_clustering(graph)[0]
    lcc = np.mean(local_clustering(graph).a)
    graph_metrics = {
        'num_vertices': n,
        'num_edges': m,
        'avg_degree': k,
        'std_degree': std,
        'global_clustering_coefficent': gcc,
        'avg_local_clustering_coefficent': lcc.item(),
    }
    if n < 100000:
        d = np.sum(shortest_distance(graph).get_2d_array()) / (n * (n - 1))
        bc = np.max(betweenness(graph)[0].a)
        graph_metrics.update({
            'avg_shortest_path': d,
            'betweenness_centrality': bc.item(),
        })

    if ddplot_prefix is not None:
        degrees, counts = np.unique(graph.degree_property_map("total").a, return_counts=True)
        plt.scatter(degrees, counts, color='blue', label='Degree Distribution')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Degree (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Degree Distribution (log-log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{ddplot_prefix}_graph_degree_distribution.png')
        plt.clf()

    component = extract_largest_component(graph)
    n = component.num_vertices()
    m = component.num_edges()
    k, std = vertex_average(component, component.degree_property_map("total"))
    gcc = global_clustering(component)[0]
    lcc = np.mean(local_clustering(component).a)
    component_metrics = {
        'num_vertices': n,
        'num_edges': m,
        'avg_degree': k,
        'std_degree': std,
        'global_clustering_coefficent': gcc,
        'avg_local_clustering_coefficent': lcc.item(),

    }
    if n < 100000:
        d = np.sum(shortest_distance(component).get_2d_array()) / (n * (n - 1))
        bc = np.max(betweenness(component)[0].a)
        component_metrics.update({
            'avg_shortest_path': d,
            'betweenness_centrality': bc.item(),
        })

    if ddplot_prefix is not None:
        degrees, counts = np.unique(component.degree_property_map("total").a, return_counts=True)
        plt.scatter(degrees, counts, color='blue', label='Degree Distribution')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Degree (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Degree Distribution (log-log scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{ddplot_prefix}_component_degree_distribution.png')
        plt.clf()
    
    return {
        'graph': graph_metrics,
        'component': component_metrics
    }


from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from sklearn.metrics import f1_score
from tqdm import tqdm
import heapq
import os


def _community_metrics(b, bv_set, r2v, num_vertices):
    y_pred = np.zeros(num_vertices, dtype=bool)
    y_true = np.zeros(num_vertices, dtype=bool)
    np.put(y_pred, list(bv_set), 1)
    block_results = {}
    for r, rv_set in r2v.items():
        if not bv_set.isdisjoint(rv_set):
            np.put(y_true, list(rv_set), 1)
            score = f1_score(y_true, y_pred)
            block_results[r] = score
            np.put(y_true, list(rv_set), 0)

    top_scores = heapq.nlargest(10, block_results.items(), key=lambda x: x[1])
    return b, {'size': len(bv_set), 'f1': dict(top_scores)}

def community_metrics(state, n_workers=None):
    blocks = state.get_blocks()
    g = state.g

    b2v = defaultdict(set)
    for v in g.vertices():
        b = blocks[v]
        b2v[b].add(int(v))

    repo_prop = g.ep['repository_id']
    r2v = defaultdict(set)
    for edge in g.edges():
        r = repo_prop[edge]
        v1, v2 = edge
        r2v[r].add(int(v1))
        r2v[r].add(int(v2))

    n_workers = min(len(b2v.keys()), os.cpu_count() if n_workers is None else n_workers)
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_community_metrics, b, bv_set, r2v, g.num_vertices()) for b, bv_set in b2v.items()]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Community Metrics Analysis'):
            block, block_results = f.result()
            results[block] = block_results

    return results