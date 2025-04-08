from graph_tool import Graph


def analyze_multi_language_nodes(graph: Graph) -> dict:
    """
    Analyzes the given graph and identifies nodes associated with multiple languages.

    Args:
        graph (Graph): The graph-tool graph object to analyze.

    Returns:
        dict: A dictionary mapping node IDs to lists of languages for nodes
              associated with two or more languages.
    """
    multi_lang_nodes = {}
    id_prop = graph.vertex_properties["id"]
    lang_prop = graph.vertex_properties["language"]

    for v in graph.vertices():
        languages = lang_prop[v]
        if len(languages) >= 2:
            node_id = id_prop[v]
            multi_lang_nodes[node_id] = languages

    return multi_lang_nodes
