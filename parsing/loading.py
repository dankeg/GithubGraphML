import csv
import graph_tool
from graph_tool.all import *


def load_csv_graph(path: str, label: str) -> graph_tool.Graph:
    """Loads a graph from a provided CSV file, using the provided label.

    Args:
        path (str): CSV file representing a graph
        label (str): Primary property, such as coding language, to add as a property

    Returns:
        graph_tool.Graph: Loaded Graph
    """
    edge_list = []
    with open(path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        titles = None
        for row in csv_reader:
            if titles is None:
                titles = row
            else:
                edge_list.append((row[1], row[2]))

    new_graph = graph_tool.Graph()
    new_graph.add_edge_list(edge_list, hashed=True)

    vlabel = new_graph.new_vertex_property("string")
    for v in new_graph.vertices():
        vlabel[v] = label

    new_graph.vertex_properties["language"] = vlabel
    return new_graph


def combine_graphs(graphs: list) -> Graph:
    """Combines several graphs into a single one.

    Args:
        graphs (list): List of graphs

    Returns:
        Graph: Combined graph
    """
    if not graphs:
        return Graph()

    combined = Graph(directed=graphs[0].is_directed())

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


print("Starting initial loading...")
assembly_graph = load_csv_graph(
    "developers_social_network/ASSEMBLY_developers_social_network.csv", "Assembly"
)
print("Starting initial loading...")
js_graph = load_csv_graph(
    "developers_social_network/JAVASCRIPT_developers_social_network.csv", "JS"
)
print("Starting initial loading...")
pascal_graph = load_csv_graph(
    "developers_social_network/PASCAL_developers_social_network.csv", "Pascal"
)
print("Starting initial loading...")
perl_graph = load_csv_graph(
    "developers_social_network/PERL_developers_social_network.csv", "Perl"
)
print("Starting initial loading...")
python_graph = load_csv_graph(
    "developers_social_network/PYTHON_developers_social_network.csv", "Python"
)
print("Starting initial loading...")
ruby_graph = load_csv_graph(
    "developers_social_network/RUBY_developers_social_network.csv", "Ruby"
)
print("Starting initial loading...")
visual_basic_graph = load_csv_graph(
    "developers_social_network/VISUALBASIC_developers_social_network.csv", "Basic"
)
print("Starting initial loading...")

print("Starting Combination...")
graph_list = [assembly_graph, js_graph, pascal_graph, perl_graph, python_graph, ruby_graph, visual_basic_graph]
combined = combine_graphs(graph_list)

print("Starting drawing process...")
graph_draw(
    combined,
    vertex_text=combined.vertex_properties["language"],
    output_size=(10000, 10000),
    edge_pen_width=1.2,
    output="graph_output.png",
)
