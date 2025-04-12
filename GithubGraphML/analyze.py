import graph_tool.all as graph_tool
from datetime import datetime
from typing import *

from analysis.metrics import analyze_node_languages, classic_metrics
from parsing.loading import combine_graphs, load_csv_graph, load_csv_vertices, transform_bipartite, merge_parallel, prune

import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import os


langauge_list = ['Assembly', 'JavaScript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
data_dir = './data'


date2float = lambda date: datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp()
lang_color = {
    -1: '#A9A9A9',       # User - Grey
    0:  '#FF5733',       # Assembly - strong orange-red
    1:  '#F1C40F',       # JavaScript - vibrant yellow
    2:  '#8E44AD',       # Pascal - deep purple
    3:  '#1ABC9C',       # Perl - teal
    4:  '#3498DB',       # Python - blue
    5:  '#E74C3C',       # Ruby - ruby red
    6:  '#2ECC71',       # VisualBasic - green
}
lang_map = {
    '': -1,
    'Assembly': 0,
    'JavaScript': 1,
    'Pascal': 2,
    'Perl': 3,
    'Python': 4,
    'Ruby': 5,
    'VisualBasic': 6,
}

def load_repositories(cache='repositories.pkl', use_cached = True, cache_result = True, verbose = True):
    if not os.path.exists(cache) or not use_cached:
        if verbose: print('Loading repositories...')
        repositories, _ = load_csv_vertices(f'{data_dir}/repositories.csv', vprop_name='repository_id')
        repositories.vp['programming_language_id'] = repositories.vp['programming_language_id'].t(int, value_type='int')
        repositories.vp['duration_days'] = repositories.vp['duration_days'].t(int, value_type='int')
        repositories.vp['number_commits'] = repositories.vp['number_commits'].t(int, value_type='int')
        repositories.vp['number_commiters'] = repositories.vp['number_commiters'].t(int, value_type='int')
        repositories.vp['create_date'] = repositories.vp['create_date'].t(date2float, value_type='float')
        repositories.vp['end_date'] = repositories.vp['create_date'].t(date2float, value_type='float')
        repositories.vp['repository_id'] = repositories.vp['repository_id'].t(int, value_type='int')
        repositories.shrink_to_fit()
        if cache_result:
            if verbose: print('Caching repositories...')
            with open(cache, 'wb') as f:
                pickle.dump(repositories, f)
    else: 
        if verbose: print('Loading cached repositories...')
        with open(cache, 'rb') as f:
            repositories = pickle.load(f)
    return repositories

def load_developers(cache='developers.pkl', use_cached = True, cache_result = True, verbose = True):
    if not os.path.exists(cache) or not use_cached:
        if verbose: print('Loading developers...')
        developers, _ = load_csv_vertices(f'{data_dir}/developer.csv', vprop_name='developer_id')
        def transform(date):  # format different & defect in dataset
            if date == ',2012-05-0': date = '2012-05-01'
            date = datetime.strptime(date, "%Y-%m-%d")
            return date.timestamp()
        developers.vp['created_at'] = developers.vp['created_at'].t(transform, value_type='float')
        developers.vp['fake'] = developers.vp['fake'].t(bool, value_type='bool')
        developers.vp['deleted'] = developers.vp['deleted'].t(bool, value_type='bool')
        developers.vp['developer_id'] = developers.vp['developer_id'].t(int, value_type='int')
        developers.vp['is_org'] = developers.vp['usr_type'].t(lambda type: type == 'ORG', value_type='bool')
        del developers.vp['company']
        del developers.vp['usr_type']
        del developers.vp['long']
        del developers.vp['lat']
        del developers.vp['country_code']
        del developers.vp['location']
        del developers.vp['state']
        del developers.vp['city']
        developers.shrink_to_fit()
        if cache_result:
            if verbose: print('Caching developers...')
            with open(cache, 'wb') as f:
                pickle.dump(developers, f)
    else: 
        if verbose: print('Loading cached developers...')
        with open(cache, 'rb') as f:
            developers = pickle.load(f)
    return developers

def load_networks(languages=['Assembly', 'Javascript', 'Pascal', 'Perl', 'Python', 'Ruby', 'VisualBasic'], enhance=True, developers_cache='developers.pkl', verbose=True):
    if enhance: 
        if developers_cache is not None: developers = load_developers(cache=developers_cache, verbose=verbose)
        else: developers = load_developers(use_cached=False, cache_result=False, verbose=verbose)
    developers_social_networks = []
    for language in languages:
        if verbose: print(f'Loading {language} network...')
        csv_template = f'{data_dir}/developers_social_network/%s_developers_social_network.csv'
        graph, _ = load_csv_graph(csv_template % language.upper(), (1, 2), vprop_name='developer_id')
        graph.ep['programming_language_id'] = graph.new_ep('int', val=lang_map[language])
        graph.ep['repository_id'] = graph.ep['repository_id'].t(int, value_type='int')
        graph.ep['contribution_days'] = graph.ep['contribution_days'].t(int, value_type='int')
        graph.ep['number_commits'] = graph.ep['number_commits'].t(int, value_type='int')
        graph.ep['end_contribution_date'] = graph.ep['end_contribution_date'].t(date2float, value_type='float')
        graph.ep['begin_contribution_date'] = graph.ep['begin_contribution_date'].t(date2float, value_type='float')
        graph.vp['developer_id'] = graph.vp['developer_id'].t(int, value_type='int')
        if enhance:
            if verbose: print(f'Enhancing {language} network...')
            for key in developers.vp: 
                if key not in graph.vp:
                    dev_vp = developers.vp[key]
                    val_type = dev_vp.value_type()
                    g_vp = graph.new_vp(val_type)
                    graph.vp[key] = g_vp
            dev_id_prop = developers.vp['developer_id']
            graph_id_prop = graph.vp['developer_id']
            dev_map = {}
            for v in developers.vertices():
                dev_id = dev_id_prop[v]
                dev_map[dev_id] = v
            for v in graph.vertices():
                dev_id = graph_id_prop[v]
                dev_v = dev_map[dev_id]
                for key in developers.vp:
                    dev_vp = developers.vp[key]
                    graph_vp = graph.vp[key]
                    graph_vp[v] = dev_vp[dev_v]
        graph.set_directed(False)
        graph.shrink_to_fit()
        developers_social_networks.append(graph)
    return developers_social_networks

def load_combined(languages: list[str], cache = 'combined.pkl', use_cached=True, cache_result=True, verbose=True):
    if not os.path.exists(cache) or not use_cached:
        network_list = load_networks(languages, verbose=verbose)

        if verbose: print('Combining networks...')
        combined, _ = combine_graphs(network_list, vprop_name='developer_id')
        combined.set_directed(False)
        combined.shrink_to_fit()
        if cache_result:
            if verbose: print('Caching combination...')
            with open(cache, 'wb') as f:
                pickle.dump(combined, f)
    else: 
        if verbose: print('Loading cached combination...')
        with open(cache, 'rb') as f:
            combined = pickle.load(f)
    return combined

def load_bipartite(languages: list[str], purge=True, merge=True, cache='bipartite.pkl', use_cached=True, cache_result=True, combined_cache='combined.pkl', repository_cache='repositories.pkl', verbose=True):
    if not os.path.exists(cache) or not use_cached:
        if combined_cache is not None: bipartite = load_combined(languages, cache=combined_cache, verbose=verbose) 
        else: bipartite = load_combined(languages, use_cached=False, cache_result=False, vebose=verbose)
        if repository_cache is not None: repositories = load_repositories(cache=repository_cache, verbose=verbose)
        else: repositories = load_repositories(use_cached=False, cache_result=False, verbose=verbose)
        if verbose: print('Transforming combined...')
        transform_bipartite(bipartite, repositories, 'repository_id')
        bipartite.vp['is_repository'] = bipartite.vp['bipartite']
        del bipartite.vp['bipartite']
        del repositories
        if purge:
            if verbose: print('Pruning bipartite...')
            prune(bipartite)
            bipartite.purge_vertices()
            del bipartite.vp['has_neighbors']
        if merge:
            if verbose: print('Merging parallel edges...')
            merge_parallel(bipartite, [bipartite.ep['number_commits'], bipartite.ep['contribution_days']])
            bipartite.purge_edges()
            del bipartite.ep['merged']
        if verbose: print('dumping bipartite...')
        if cache_result:
            with open(cache, 'wb') as f:
                pickle.dump(bipartite, f)
    else:
        if verbose: print('Loading cached bipartite...')
        with open(cache, 'rb') as f:
            bipartite = pickle.load(f)
    return bipartite


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
    labels = ['+'.join(k) for k in data.keys()]
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
#
# print('Performing Analysis...')
# analyze_langauge_distribution(combined)
if __name__ == '__main__':
    networks = load_networks(langauge_list)
    for i, network in enumerate(networks):
        print(f'Analyzing {langauge_list[i]} network...')
        metrics = classic_metrics(network, ddplot=f'results/{langauge_list[i]}_degree_distribution.png')
        with open(f'results/{langauge_list[i]}_metrics.txt', 'w') as f:
            json.dump(metrics, f, indent=2)
    del networks

    combined = load_combined(langauge_list, ddplot='results/combined_degree_distribution.png')
    print('Analyzing combined network...')
    classic_metrics(combined)
    draw_combined(combined, output='combined.svg', fmt='svg')

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

    del combined
    bipartite = load_bipartite(langauge_list)
    draw_combined(bipartite, output='bipartite.svg', fmt='svg')

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
