import graph_tool.all as graph_tool
from typing import *

from analysis.metrics import analyze_node_languages, classic_metrics
from parsing.loading import combine_graphs, load_csv_graph, load_csv_vertices, transform_bipartite

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

langauge_list = ['Assembly', 'Javascript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
combined_pickle = 'combined.pkl'
use_cached_combined = True
cache_combined = True

use_cached_community = True
cache_community = True

data_dir = './data'

def load_networks(languages=['Assembly', 'Javascript', 'Pascal', 'Perl', 'Python', 'Ruby', 'VisualBasic'], vprop_name='name'):
    developers_social_networks = []
    for language in languages:
        print(f"Loading {language}...")
        csv_template = f"{data_dir}/developers_social_network/%s_developers_social_network.csv"
        graph, _ = load_csv_graph(csv_template % language.upper(), (1, 2), vprop_name=vprop_name)
        graph.ep["language"] = graph.new_ep("string", val=language)
        developers_social_networks.append(graph)
    return developers_social_networks

def analyze_langauge_distribution(combined):
    # Count instances of user language usage combinations
    results = analyze_node_languages(combined)
    combined_list = []
    for _, item in results.items():
        combined_list.append(item)
    data = Counter(combined_list)
    print(data)

    # Prepare labels and sizes for plotting
    labels = ['+'.join(k) for k in data.keys()]
    sizes = list(data.values())

    # Pie chart with matplotlib
    _, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(sizes, startangle=140, wedgeprops=dict(width=0.5))

    # Adding a legend with clear labels
    ax.legend(wedges, labels, title="Language Combinations", loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title("Distribution of Programming Language Combinations")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(combined_pickle) or not use_cached_combined:
        graph_list = load_networks(langauge_list, vprop_name='id')
        print("Starting combination...")
        combined, _ = combine_graphs(graph_list, vprop_name='id')
        combined.set_directed(False)
        if cache_combined:
            print("Caching combination...")
            with open(combined_pickle, 'wb') as f:
                pickle.dump(combined, f)
    else: 
        print("Loading cached combination...")
        with open(combined_pickle, 'rb') as f:
            combined = pickle.load(f)
            combined.set_directed(False)

    # print('loading developers...')
    # with open('developers.pkl', 'rb') as f:
    #     developers = pickle.load(f)

    # print('recombining graph...')
    # with open('developer_combined', 'wb') as f:
    #     developer_combined = combine_graphs(combined, developers)
    #     pickle.dump(developer_combined, f)
    
    # classic_metrics(combined, 'combined_dd.png')    
    
    # vertcies, vp = load_csv_vertices(f"{data_dir}/developer.csv", vprop_name='id')
    # with open('developers.pkl', 'wb') as f:
    #     pickle.dump(vertcies, f)
    #     print(vertcies)

    # print("Loading cached repositories...")
    # with open('repositories.pkl', 'rb') as f:
    #     repositories = pickle.load(f)
        
    # print("transforming_graph...")
    # transform_bipartite(combined, repositories, 'repository_id')
    # del repositories
    # print("dumping bipartite...")
    # with open('bipartite.pkl', 'wb') as f:
    #     pickle.dump(combined, f)
    
    # with open('bipartite.pkl', 'rb') as f:
    #     bipartite = pickle.load(f)

    # print(bipartite)

    # print("Performing Analysis...")
    # analyze_langauge_distribution(combined)

    # if not os.path.exists('simple_state.pkl') or not use_cached_community:
    #     with graph_tool.openmp_context(nthreads=4, schedule="guided"):
    #         print("Starting simple community analysis...")
    #         state = graph_tool.minimize_blockmodel_dl(combined)
    #     if cache_community:
    #         with open('simple_state.pkl', 'wb') as f:
    #                     print("Caching analysis...")
    #                     pickle.dump(state, f)
    # else:
    # with open('simple_state.pkl', 'rb') as f:
    #     print("Loading state...")
    #     state = pickle.load(f)

    # state.draw(output='community.png', max_iter=1000)

    # with graph_tool.openmp_context(nthreads=4, schedule="guided"):
    #     state = graph_tool.minimize_nested_blockmodel_dl(combined)
    #     with open('nested_state.pkl', 'wb') as f:
    #                 print("Caching Analysis...")
    #                 pickle.dump(state, f)
    #                 del state

    # TODO: import repositories.csv and create bipartite with all language graphs + combined graph
    # TODO: merge parallel edges together in output of bipartite graph generation
    # TODO: import developer.csv as node vectors
    # TODO: analyze all language graphs + combined graph basic network stats
    #  - degree distribution
    #  - clustering coeffcient
    #  - average shortest path length
    #  - number of connected components
    #  - betweeness centrality
    #  - ??modularity??
    
    # attempt coarse-grained modeling to simplify combined graph structure?
