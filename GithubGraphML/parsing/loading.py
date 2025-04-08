import graph_tool.all as graph_tool
import csv
from typing import *

from GithubGraphML.analysis.metrics import analyze_multi_language_nodes
import matplotlib as plt

def load_csv_graph(path: str, edge_indices: tuple[int, int] = (0, 1), titles: list[str] = None) -> graph_tool.Graph:
    """Loads a graph from a provided CSV file.

    This function reads a CSV file representing a graph, extracts edges based on 
    the specified column indices, and loads them into a `graph_tool.Graph` object.

    Args:
        path (str): CSV file representing a graph.
        edge_indices (tuple[int, int]): Column indices corresponding to the source 
            and target vertices of each edge. Edge properties for these columns are
            not generated. Defaults to (0, 1).
        titles (list[str], optional): List of column names to be used as edge 
            property names. If None, the first row of the CSV is assumed to contain
            column titles.

    Returns:
        graph_tool.Graph: Loaded Graph.
        graph_tool.VertexPropertyMap: Loaded vertex property map.
    """
    new_graph = graph_tool.Graph()
    edge_list = []
    with open(path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        if titles is None: titles = next(csv_reader)
        for line in csv_reader:
                props = [prop for idx, prop in enumerate(line) if idx not in edge_indices]
                v1 = line[edge_indices[0]]
                v2 = line[edge_indices[1]]
                edge = (v1, v2, *props)
                edge_list.append(edge)
    
    titles = [title for idx, title in enumerate(titles) if idx not in edge_indices]
    eprops = [new_graph.new_ep("string") for _ in titles]
    for eprop, title in zip(eprops, titles):
        new_graph.ep[title] = eprop

    vp = new_graph.add_edge_list(edge_list, eprops=eprops, hashed=True)
    return new_graph, vp


def combine_graphs(graphs: list[graph_tool.Graph], vprops: list[graph_tool.VertexPropertyMap]) -> graph_tool.Graph:
    """Combines several graphs into a single one.

    Args:
        graphs (list[graph_tool.Graph]): List of graphs to combine.
        vprops (list[graph_tool.VertexPropertyMap]): A list of vertex property maps to combine on.
    Returns:
        graph_tool.Graph: Combined graph
    """
    if len(graphs) != len(vprops):
        raise ValueError('Numbers of graphs and vertex property maps must match.')
    if not graphs:
        return graph_tool.Graph()
    
    vertices = {v for vprop in vprops for v in vprop}
    combined = graph_tool.Graph(len(vertices))
    
    nvprop = combined.new_vp('string', vals=vertices)
    nvprop_map = {}
    for i, prop in enumerate(nvprop):
        nvprop_map[prop] = i

    for g, vprop in zip(graphs, vprops):
        for v, *vprops in zip(g.vertices(), *[[(key, prop) for prop in g.vp[key]] for key in g.vp]):
            for key, prop in vprops:
                if key not in combined.vp:
                    val_type = g.vp[key].value_type()
                    vp = combined.new_vp(val_type)
                    combined.vp[key] = vp
                idx = nvprop_map[vprop[v]]
                combined.vp[key][idx] = prop

        for (v1, v2), *eprops in zip(g.edges(), *[[(key, prop) for prop in g.ep[key]] for key in g.ep]):
            edge = nvprop_map[vprop[v1]], nvprop_map[vprop[v2]]
            e = combined.add_edge(*edge)
            for key, prop in eprops:
                if key not in combined.ep:
                    val_type = g.ep[key].value_type()
                    ep = combined.new_ep(val_type)
                    combined.ep[key] = ep
                combined.ep[key][e] = prop

    return combined, nvprop


if __name__ == "__main__":
    print("Starting Assembly loading...")
    assembly_graph, assembly_vp = load_csv_graph(
        "developers_social_network/ASSEMBLY_developers_social_network.csv", (1, 2)
    )
    assembly_graph.ep["language"] = assembly_graph.new_ep("string", val="Assembly")

    print("Starting Javascript loading...")
    js_graph, js_vp = load_csv_graph(
        "developers_social_network/JAVASCRIPT_developers_social_network.csv", (1, 2)
    )
    js_graph.ep["language"] = js_graph.new_ep("string", val="JS")

    print("Starting Pascal loading...")
    pascal_graph, pascal_vp = load_csv_graph(
        "developers_social_network/PASCAL_developers_social_network.csv", (1, 2)
    )
    pascal_graph.ep["language"] = pascal_graph.new_ep("string", val="Pascal")

    print("Starting Perl loading...")
    perl_graph, perl_vp = load_csv_graph(
        "developers_social_network/PERL_developers_social_network.csv", (1, 2)
    )
    perl_graph.ep["language"] = perl_graph.new_ep("string", val="Perl")

    print("Starting Python loading...")
    python_graph, python_vp = load_csv_graph(
        "developers_social_network/PYTHON_developers_social_network.csv", (1, 2)
    )
    python_graph.ep["language"] = python_graph.new_ep("string", val="Python")

    # print("Starting Ruby loading...")
    # ruby_graph, ruby_vp = load_csv_graph(
    #     "developers_social_network/RUBY_developers_social_network.csv", (1, 2)
    # )
    # ruby_graph.ep["language"] = ruby_graph.new_ep("string", val="Ruby")

    print("Starting Visual Basic loading...")
    visual_basic_graph, visual_basic_vp = load_csv_graph(
        "developers_social_network/VISUALBASIC_developers_social_network.csv", (1, 2)
    )
    visual_basic_graph.ep["language"] = visual_basic_graph.new_ep("string", val="Visual Basic")

    print("Starting Combination...")
    graph_list = [assembly_graph, js_graph, pascal_graph, perl_graph, python_graph, visual_basic_graph]
    vp_list = [assembly_vp, js_vp, pascal_vp, perl_vp, python_vp, visual_basic_vp]
    combined = combine_graphs(graph_list, vp_list)

    print("Performing Analysis")
    results = analyze_multi_language_nodes(combined)

    combined_list = []
    for key, item in results.items():
        combined_list.append(item)

    data = Counter(combined_list)
    print(data)

    # Prepare labels and sizes for plotting
    labels = ['+'.join(k) for k in data.keys()]
    sizes = list(data.values())

    # Pie chart with matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts = ax.pie(sizes, startangle=140, wedgeprops=dict(width=0.5))

    # Adding a legend with clear labels
    ax.legend(wedges, labels, title="Language Combinations", loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title("Distribution of Programming Language Combinations")

    plt.tight_layout()
    plt.show()

    print("Starting drawing process...")
    # graph_tool.graph_draw(
    #     combined,
    #     # vertex_text=combined.vertex_properties["language"],
    #     output_size=(10000, 10000),
    #     edge_pen_width=1.2,
    #     output="graph_output.png",
    # )

    positions = graph_tool.sfdp_layout(G, p=3, r=0.5, cooling_step=0.995, max_iter=1000)
    graph_tool.graph_draw(
        combined,
        pos = positions,
        output_size=(10000, 10000),
        edge_pen_width=1.2,
    )

    
