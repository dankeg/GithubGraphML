import graph_tool.all as graph_tool
from typing import *

from GithubGraphML.analysis.metrics import analyze_node_languages
from GithubGraphML.parsing.loading import combine_graphs, load_csv_graph
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

    # print("Performing Analysis...")
    # analyze_langauge_distribution(combined)

    with open('simple_state.pkl', 'rb') as f:
        print("loading state...")
        state = pickle.load(f)
        
        g = combined
        del combined

        # Get block assignments (which vertices belong to which block)
        blocks = state.b
        
        # Create a new graph where blocks are aggregated
        coarse_g = graph_tool.Graph()

        # Create a new vertex property to represent the block assignments
        block_property = coarse_g.new_vertex_property("int")
        block_edge_counts = {}
        block_vertex_count = {}
        block_to_vertex = {}

        # Iterate over the original graph to aggregate nodes and count edges between blocks
        print("aggregating graph...")
        # Iterate over the original graph to aggregate nodes and count edges between blocks
        for v in g.vertices():
            block = blocks[v]  # Get block for each vertex
            if block not in block_to_vertex:
                # Create a super-node for each block
                new_vertex = coarse_g.add_vertex()
                block_to_vertex[block] = new_vertex
                block_property[new_vertex] = block
                block_vertex_count[block] = 1  # Initialize the count
            else:
                block_vertex_count[block] += 1
            
            # Now count edges between blocks
            for e in v.out_edges():
                neighbor_block = blocks[e.target()]
                if block != neighbor_block:  # Only count edges between different blocks
                    block_pair = tuple(sorted([block, neighbor_block]))
                    if block_pair not in block_edge_counts:
                        block_edge_counts[block_pair] = 1
                    else:
                        block_edge_counts[block_pair] += 1

        # Create edges in the coarse graph based on block adjacency
        for (block1, block2), edge_count in block_edge_counts.items():
            v1 = block_to_vertex[block1]
            v2 = block_to_vertex[block2]
            coarse_g.add_edge(v1, v2)

        # Now you have a coarse graph, where the vertices represent blocks
        # and the edges represent the connections between those blocks.

        # Increase vertex size according to block size (number of vertices in each block)
        vertex_size = coarse_g.new_vertex_property("int")
        for v in coarse_g.vertices():
            block = block_property[v]
            vertex_size[v] = block_vertex_count[block]  # Scale size by block size

        # Set edge thickness (pen width) proportional to the number of edges between blocks
        edge_thickness = coarse_g.new_edge_property("float")
        for e in coarse_g.edges():
            block1 = block_property[e.source()]
            block2 = block_property[e.target()]
            block_pair = tuple(sorted([block1, block2]))
            edge_thickness[e] = block_edge_counts[block_pair] * 0.01  # Scale edge thickness

        # Plot the coarse graph with edge thickness
        graph_tool.graph_draw(coarse_g, 
                      vertex_color=block_property, 
                      output_size=(1024, 1024), 
                      vertex_size=vertex_size, 
                      edge_pen_width=edge_thickness,
                      output="coarse_community.png")

    # with graph_tool.openmp_context(nthreads=16, schedule="guided"):
    #     state = graph_tool.minimize_blockmodel_dl(combined)
    #     with open('nested_state.pkl', 'wb') as f:
    #                 print("Caching Analysis...")
    #                 pickle.dump(state, f)
    #                 del state
    
    # with graph_tool.openmp_context(nthreads=16, schedule="guided"):
    #     state = graph_tool.minimize_nested_blockmodel_dl(combined)
    #     with open('nested_state.pkl', 'wb') as f:
    #                 print("Caching Analysis...")
    #                 pickle.dump(state, f)
    #                 del state
    
    # print("Starting drawing process...")
    # positions = graph_tool.sfdp_layout(combined, p=3, r=0.5, cooling_step=0.995, max_iter=200)
    # graph_tool.graph_draw(
    #     combined,
    #     pos = graph_tool.sfdp_layout(combined, max_itter=20),
    #     vertex_text=combined.vp['id'],
    #     pos = positions,
    #     output_size=(1000, 1000),
    #     output='complete_graph.png',
    #     fmt='svg',
    #     edge_pen_width=1.2,
    # )