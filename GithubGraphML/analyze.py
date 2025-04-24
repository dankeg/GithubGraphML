import graph_tool.all as graph_tool
from datetime import datetime
from typing import *

from analysis.metrics import analyze_node_languages, classic_metrics
from parsing.loading import *

import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import os


langauge_list = ['Assembly', 'JavaScript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
data_dir = './data'

def draw_bipartite(bipartite, **kwargs):
    coloring = bipartite.vp['programming_language_id'].t(lang_color.__getitem__, value_type='string')
    graph_tool.graph_draw(
        bipartite,
        output_size=(10192, 10192), 
        output='bipartite.png',
        vertex_fill_color=coloring,
        max_iter=1000,
        **kwargs
    )

def draw_combined(combined, **kwargs):
    coloring = combined.ep['programming_language_id'].t(lang_color.__getitem__, value_type='string')
    graph_tool.graph_draw(
        combined,
        output_size=(10192, 10192),
        output='combined.png',
        edge_color=coloring,
        max_iter=1000,
        **kwargs,
    )


def analyze_langauge_distribution(combined):
    # Count instances of user language usage combinations
    results = analyze_node_languages(combined)
    combined_list = []
    for _, item in results.items():
        combined_list.append(item)
    data = Counter(combined_list)
    print(data)

    # Prepare labels and sizes for plotting
    labels = ["+".join(k) for k in data.keys()]
    sizes = list(data.values())

    # Pie chart with matplotlib
    _, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(sizes, startangle=140, wedgeprops=dict(width=0.5))

    # Adding a legend with clear labels
    ax.legend(wedges, labels, title='Language Combinations', loc='center left',
            bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title('Distribution of Programming Language Combinations')
    plt.tight_layout()
    plt.show()


# TODO: analyze all language graphs + combined graph basic network stats
#  - degree distribution    
#  - clustering coeffcient
#  - average shortest path length
#  - number of connected components
#  - betweeness centrality
if __name__ == '__main__':
    networks = load_networks(data_dir, langauge_list)
    for i, network in enumerate(networks):
        print(f'Analyzing {langauge_list[i]} network...')
        metrics = classic_metrics(network, ddplot=f'results/{langauge_list[i]}_degree_distribution.png')
        with open(f'results/{langauge_list[i]}_metrics.txt', 'w') as f:
            json.dump(metrics, f, indent=2)
    del networks

    # for i, language in enumerate(langauge_list[1:]):
    #     print(f'Loading {language} network...')
    #     with open(f'{language}.pkl', 'rb') as f:
    #         network = pickle.load(f)
    #     for key in list(network.vp):
    #         del network.vp[key]
    #     for key in list(network.ep):
    #         del network.ep[key]
    #     print(f'Analyzing {language} network...')
    #     metrics = classic_metrics(network, ddplot=f'results/{language}_degree_distribution.png')
    #     with open(f'results/{language}_metrics.txt', 'w') as f:
    #         json.dump(metrics, f, indent=2)

    combined = load_combined(data_dir, langauge_list)
    print('Analyzing combined network...')
    classic_metrics(combined, ddplot='results/combined_degree_distribution.png')
    draw_combined(combined, output='combined.svg', fmt='svg')
    del combined

    bipartite = load_bipartite(data_dir, langauge_list)
    draw_bipartite(bipartite, output='bipartite.svg', fmt='svg')
    del bipartite

    combined = load_combined(data_dir, langauge_list)

    with graph_tool.openmp_context(nthreads=4, schedule='guided'):
        print('Starting simple community analysis...')
        state = graph_tool.minimize_blockmodel_dl(combined)
        with open('simple_state.pkl', 'wb') as f:
            print('Caching simple analysis...')
            pickle.dump(state, f)
            del state

    with graph_tool.openmp_context(nthreads=4, schedule='guided'):
        print('Starting nested community analysis...')
        state = graph_tool.minimize_nested_blockmodel_dl(combined)
        with open('nested_state.pkl', 'wb') as f:
            print('Caching nested analysis...')
            pickle.dump(state, f)
            del state
    
    with graph_tool.openmp_context(nthreads=4, schedule='guided'):
        with open('simple_state.pkl', 'rb') as f:
            print('Loading simple community analysis...')
            state = pickle.load(f)
            print('Drawing simple community analysis...')
            state.draw(output='simple_community.svg', fmt='svg')
            del state
    
    with graph_tool.openmp_context(nthreads=4, schedule='guided'):
        with open('nested_state.pkl', 'rb') as f:
            print('Loading nested community analysis...')
            state = pickle.load(f)
            print('Drawing nested community analysis...')
            state.draw(output='nested_community.svg', fmt='svg')
            del state

    # del combined
    # bipartite = load_bipartite(langauge_list)
    # draw_combined(bipartite, output='bipartite.svg', fmt='svg')

    # print('Performing Analysis...')
    # analyze_langauge_distribution(combined)

    # state = graph_tool.BlockState(combined)
    # dS, nattempts, nmoves = state.multiflip_mcmc_sweep(niter=1000)
    # print("Change in description length:", dS)
    # print("Number of accepted vertex moves:", nmoves)
    # graph_tool.mcmc_equilibrate(state, wait=100, nbreaks=2, mcmc_args=dict(niter=10))
    # bs = [] # collect some partitions
    # def collect_partitions(s):
    #     global bs
    #     bs.append(s.b.a.copy())
    # # Now we collect partitions for exactly 100,000 sweeps, at intervals of 10 sweeps:
    # graph_tool.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10), callback=collect_partitions)
    # # Disambiguate partitions and obtain marginals
    # pmode = graph_tool.PartitionModeState(bs, converge=True)
    # pv = pmode.get_marginal(combined)
    # # Now the node marginals are stored in property map pv. We can visualize them as pie charts on the nodes:
    # state.draw(pos=combined.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv, output="lesmis-sbm-marginals.svg")
