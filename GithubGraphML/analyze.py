import graph_tool.all as graph_tool
from typing import *

from parsing.loading import load_csv_graph, combine_graphs 
from analysis.metrics import analyze_node_languages
import matplotlib.pyplot as plt
import pickle
import os

langauge_list = ['Assembly', 'Javascript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
combined_pickle = 'combined.pkl'
use_cached_combined = True
cache_combined = True

data_dir = './'

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
    if os.path.exists(combined_pickle) and use_cached_combined:
        print("Loading Cached Combination...")
        with open(combined_pickle, 'rb') as f:
            combined = pickle.load(f)
            combined.set_directed(False)
            print(combined)
    else: 
        graph_list = load_networks(langauge_list, vprop_name='id')
        print("Starting Combination...")
        combined, _ = combine_graphs(graph_list, vprop_name='id')
        combined.set_directed(False)
        print(combined)
        if cache_combined:
            print("Caching Combination...")
            with open(combined_pickle, 'wb') as f:
                pickle.dump(combined, f)

    print("Performing Analysis...")
    # analyze_langauge_distribution(combined)

    print("Starting Community Analysis...")
    with graph_tool.openmp_context(nthreads=16, schedule="guided"):
        state = graph_tool.minimize_blockmodel_dl(combined)
        with open('simple_state.pkl', 'wb') as f:
                    print("Caching Analysis...")
                    pickle.dump(state, f)

    with graph_tool.openmp_context(nthreads=16, schedule="guided"):
        state = graph_tool.minimize_blockmodel_dl(combined, state_args=dict(
                recs=[combined.ep['number_commits'].t(lambda n: int(n), value_type='int')],
                rec_types=["discrete-poisson"]    
        ))
        with open('weighted_state.pkl', 'wb') as f:
                    print("Caching Analysis...")
                    pickle.dump(state, f)

    # print("Starting drawing process...")
    # positions = graph_tool.sfdp_layout(combined, p=3, r=0.5, cooling_step=0.995, max_iter=200)
    # graph_tool.graph_draw(
    #     combined,
    #     vertex_text=combined.vp['id'],
    #     pos = positions,
    #     output_size=(10000, 10000),
    #     output='complete_graph.svg',
    #     fmt='svg',
    #     edge_pen_width=1.2,
    # )