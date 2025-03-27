import graph_tool.all as graph_tool
import csv
from typing import *

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
        graph_toool.VertexPropertyMap: Loaded vertex property map.
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


def combine_graphs(graphs: list[graph_tool.Graph]) -> graph_tool.Graph:
    """Combines several graphs into a single one.

    Args:
        graphs (list[graph_tool.Graph]): List of graphs

    Returns:
        graph_tool.Graph: Combined graph
    """
    if not graphs:
        return graph_tool.Graph()

    combined = graph_tool.Graph(directed=graphs[0].is_directed())

    # Collect all unique vertex property names and their types
    prop_types = {}
    for g in graphs:
        for name, prop in g.vertex_properties.items():
            if name not in prop_types:
                prop_types[name] = prop.value_type()
            else:
                # Sanity check: raise error if types conflict
                if prop_types[name] != prop.value_type():
                    raise ValueError(f"Conflicting types for vertex property '{name}'")

    # Create corresponding property maps in the combined graph
    combined_vprops = {
        name: combined.new_vertex_property(prop_type)
        for name, prop_type in prop_types.items()
    }

    # Add properties to the graph's property map so they're saved
    for name, prop in combined_vprops.items():
        combined.vertex_properties[name] = prop

    # Map each vertex in each graph to a new vertex in the combined graph
    for g in graphs:
        vmap = {}  # original vertex -> new vertex in combined graph

        # Copy vertices and assign properties
        for v in g.vertices():
            new_v = combined.add_vertex()
            vmap[int(v)] = new_v

            for name, prop in g.vertex_properties.items():
                combined_vprops[name][new_v] = prop[v]

        # Copy edges using the vertex mapping
        for e in g.edges():
            combined.add_edge(vmap[int(e.source())], vmap[int(e.target())])

    return combined


if __name__ == "__main__":
    print("Starting initial loading...")
    assembly_graph, assembly_vp = load_csv_graph(
        "developers_social_network/ASSEMBLY_developers_social_network.csv", (1, 2)
    )
    assembly_graph.ep["language"] = assembly_graph.new_ep("string", val="Assembly")
    print("Starting initial loading...")
    js_graph, js_vp = load_csv_graph(
        "developers_social_network/JAVASCRIPT_developers_social_network.csv", (1, 2)
    )
    js_graph.ep["language"] = js_graph.new_ep("string", val="JS")
    print("Starting initial loading...")
    pascal_graph, pascal_vp = load_csv_graph(
        "developers_social_network/PASCAL_developers_social_network.csv", (1, 2)
    )
    pascal_graph.ep["language"] = pascal_graph.new_ep("string", val="Pascal")
    print("Starting initial loading...")
    perl_graph, perl_vp = load_csv_graph(
        "developers_social_network/PERL_developers_social_network.csv", (1, 2)
    )
    perl_graph.ep["language"] = perl_graph.new_ep("string", val="Perl")
    print("Starting initial loading...")
    python_graph, python_vp = load_csv_graph(
        "developers_social_network/PYTHON_developers_social_network.csv", (1, 2)
    )
    python_graph.ep["language"] = python_graph.new_ep("string", val="Python")
    print("Starting initial loading...")
    ruby_graph, ruby_vp = load_csv_graph(
        "developers_social_network/RUBY_developers_social_network.csv", (1, 2)
    )
    ruby_graph.ep["language"] = assembly_graph.new_ep("string", val="Ruby")
    print("Starting initial loading...")
    visual_basic_graph, visual_basic_vp = load_csv_graph(
        "developers_social_network/VISUALBASIC_developers_social_network.csv", "Basic"
    )
    visual_basic_graph.ep["language"] = visual_basic_graph.new_ep("string", "Basic")
    print("Starting initial loading...")

    print("Starting Combination...")
    graph_list = [assembly_graph, js_graph, pascal_graph, perl_graph, python_graph, ruby_graph, visual_basic_graph]
    combined = combine_graphs(graph_list)

    print("Starting drawing process...")
    graph_tool.graph_draw(
        combined,
        # vertex_text=combined.vertex_properties["language"],
        output_size=(10000, 10000),
        edge_pen_width=1.2,
        output="graph_output.png",
    )
