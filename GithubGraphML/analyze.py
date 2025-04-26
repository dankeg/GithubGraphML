from graph_tool.all import *
from datetime import datetime
from typing import *

from GithubGraphML.analysis.metrics import language_usage, classic_metrics, community_metrics
from GithubGraphML.visualization.language import language_usage_distribution
from GithubGraphML.parsing.loading import *
from GithubGraphML.analysis.metrics import classic_metrics
from GithubGraphML.parsing.loading import *

import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import json
import tqdm


language_list = ['Assembly', 'JavaScript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
data_dir = './data'

def process_language(lang):
    with open(f'{lang}_simple_state.pkl', 'rb') as f:
        state = pickle.load(f)
    data = community_metrics(state, desc=f'community metrics: {lang}')
    with open(f'{lang}_community_metrics.json', 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    pass

    # with ProcessPoolExecutor() as executor:
    #     executor.map(process_language, language_list)
    # with open(f'combined_simple_state.pkl', 'rb') as f:
    #     state = pickle.load(f)
    # data = community_metrics(state, 8)
    # with open(f'combined_community_metrics.json', 'w') as f:
    #     json.dump(data, f, indent=2)
    # for lang in ['JavaScript']:
    #     with open(f'{lang}_simple_state.pkl', 'rb') as f:
    #         state = pickle.load(f)
    #     with open(f'results/{lang}_community_metrics.json', 'r') as f:
    #         stats = json.load(f)
    #         densities = []
    #         max_f1s = []
    #         sizes = []
    #         for b, stat in stats.items():
    #             max_f1s.append(max(stat['f1'].values()))
    #             sizes.append(int(stat['size']))
    #         stats = {
    #             'avg_max_f1': np.mean(np.array(max_f1s)),
    #             'avg_size': np.mean(np.array(sizes)),
    #             'entropy': state.entropy(),
    #         }
    #         print(lang, stats)
            

    # networks = load_networks(data_dir, language_list)
    # for lang, graph in zip(language_list, networks):
    #     with openmp_context(nthreads=4, schedule='guided'):
    #         print('Starting simple community analysis...')
    #         state = minimize_blockmodel_dl(graph)
    #         with open(f'{lang}_simple_state.pkl', 'wb') as f:
    #             print('Caching simple analysis...')
    #             pickle.dump(state, f)
    #             del state
    #     with openmp_context(nthreads=4, schedule='guided'):
    #         print('Starting nested community analysis...')
    #         state = minimize_nested_blockmodel_dl(graph)
    #         with open(f'{lang}_nested_state.pkl', 'wb') as f:
    #             print('Caching nested analysis...')
    #             pickle.dump(state, f)
    #             del state
    #     with openmp_context(nthreads=16, schedule='guided'):
    #         with open(f'{lang}_simple_state.pkl', 'rb') as f:
    #             print('Loading simple community analysis...')
    #             state = pickle.load(f)
    #             print('Drawing simple community analysis...')
    #             state.draw(output=f'results/{lang}_simple_community.png', output_size=(10192, 10192))
    #             del state
    #     with openmp_context(nthreads=16, schedule='guided'):
    #         with open(f'{lang}_nested_state.pkl', 'rb') as f:
    #             print('Loading nested community analysis...')
    #             state = pickle.load(f)
    #             if lang == 'JavaScript' or lang == 'Python':
    #                 continue
    #             print('Drawing nested community analysis...')
    #             state.draw(output=f'results/{lang}_nested_community.png', output_size=(10192, 10192))
    #             del state


    # networks = load_networks(data_dir, ["Assembly"])
    # for language, network in zip(langauge_list, networks):
    #     with open(f"results2/{language}_metrics.txt", 'w') as f:
    #         json.dump(classic_metrics(network, 'results2/{language}'), f, indent=2)
    # combined = load_combined(data_dir, langauge_list)
    # with open(f"results2/combined_metrics.txt", 'w') as f:
    #     json.dump(classic_metrics(combined, 'results2/combined'), f, indent=2)
    # combined = load_combined(data_dir, langauge_list)
    # language_usage_distribution(language_usage(combined), output='language_distribution.png')








    # bipartite = load_bipartite(data_dir, langauge_list)
    # print('Drawing bipartite network...')
    # draw_bipartite(bipartite)

    # combined = load_combined(data_dir, langauge_list)

    # combined = load_combined(data_dir, langauge_list)
    # with openmp_context(nthreads=4, schedule='guided'):
    #     print('Starting simple community analysis...')
    #     state = minimize_blockmodel_dl(combined)
    #     with open('simple_state.pkl', 'wb') as f:
    #         print('Caching simple analysis...')
    #         pickle.dump(state, f)
    #         del state

    # with openmp_context(nthreads=4, schedule='guided'):
    #     print('Starting nested community analysis...')
    #     state = minimize_nested_blockmodel_dl(combined)
    #     with open('nested_state.pkl', 'wb') as f:
    #         print('Caching nested analysis...')
    #         pickle.dump(state, f)
    #         del state
    
    # with openmp_context(nthreads=16, schedule='guided'):
    #     with open('simple_state.pkl', 'rb') as f:
    #         print('Loading simple community analysis...')
    #         state = pickle.load(f)
    #         print('Drawing simple community analysis...')
    #         state.draw(output='simple_community.png', output_size=(10192, 10192))
    #         del state
    
    # with openmp_context(nthreads=16, schedule='guided'):
    #     with open('nested_state.pkl', 'rb') as f:
    #         print('Loading nested community analysis...')
    #         state = pickle.load(f)
    #         print('Drawing nested community analysis...')
    #         state.draw(output='nested_community.png', output_size=(10192, 10192))
    #         del state

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
