import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from graph_tool.all import *


# Assumes graph is undirected
def coarse_blockstate(g: Graph, state: BlockState, **kwargs) -> Graph:
    # Initialize coarse graph
    coarse_g = Graph(directed=False)
    blocks = state.b

    # Create a new vertex property to represent the block assignments
    block_property = coarse_g.new_vertex_property("int")
    block_edge_counts = {}
    block_vertex_count = {}
    block_to_vertex = {}

    # Iterate over the original graph to coarse nodes and count edges between blocks
    print("Aggregating graph...") 
    for v in g.vertices():
        block = blocks[v]

        # Increment block vertex count
        if block not in block_to_vertex:
            new_vertex = coarse_g.add_vertex()
            block_to_vertex[block] = new_vertex
            block_property[new_vertex] = block
            block_vertex_count[block] = 1
        else:
            block_vertex_count[block] += 1
        
        # Count edges between blocks
        for e in v.out_edges():
            neighbor_block = blocks[e.target()]
            if block != neighbor_block:
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

    # Increase vertex size according to block size (number of vertices in each block)
    vertex_size = coarse_g.new_vertex_property("float")
    for v in coarse_g.vertices():
        vertex_size[v] = max(10, 0.002 * block_vertex_count[block_property[v]])

    # Increase edge color intensity according to edge count (number of edges connecting each block)
    norm = mcolors.Normalize(vmin=0, vmax=max(block_edge_counts.values()))
    cmap = plt.cm.viridis
    edge_colors = coarse_g.new_edge_property("vector<float>")
    for e in coarse_g.edges():
        block_pair = tuple(sorted([block_property[e.source()], block_property[e.target()]]))
        val = norm(block_edge_counts[block_pair])
        color_val = cmap(val) 
        edge_colors[e] = color_val[:3]

    # Increase edge width according to edge count (number of edges connecting each block)
    edge_width = coarse_g.new_edge_property("float")
    for e in coarse_g.edges():
        block_pair = tuple(sorted([block_property[e.source()], block_property[e.target()]]))
        val = max(0.5, 0.0001 * block_edge_counts[block_pair]) 
        edge_width[e] = val

    # Draw edges in assending order of edge count (number of edges connecting each block)
    sort_func = lambda e: block_edge_counts[tuple(sorted([block_property[e.source()], block_property[e.target()]]))]
    sorted_edges = coarse_g.new_edge_property("int")
    for i, e in enumerate(sorted(coarse_g.edges(), key=sort_func)):
        sorted_edges[e] = i

    # Save properties to graph
    coarse_g.vp['vertex_size'] = vertex_size
    coarse_g.vp['color'] = block_property
    coarse_g.ep['pen_width'] = edge_width
    coarse_g.ep['color'] = edge_colors
    coarse_g.ep['order'] = sorted_edges

    # draw graph as sparce
    print('Getting positions...')
    positions = sfdp_layout(coarse_g, p=25, cooling_step=0.995, max_iter=1000, multilevel=False)
    coarse_g.vp['pos'] = positions
    print('Drawing graph...')
    graph_draw(
        coarse_g,
        pos=positions,
        vertex_size=coarse_g.vp['vertex_size'], 
        vertex_fill_color=coarse_g.vp['color'], 
        vertex_color=coarse_g.vp['color'],
        edge_pen_width=coarse_g.ep['pen_width'],
        edge_color=coarse_g.ep['color'],
        eorder=coarse_g.ep['order'],
        output_size=(5096, 5096), 
        output='plotting/coarse_output.png',
        **kwargs
    )

    # return coarse graph
    return coarse_g
