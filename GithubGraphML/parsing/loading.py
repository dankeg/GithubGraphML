import graph_tool.all as graph_tool
import csv
from typing import *

# NOTE: kind of silly since graph_tool.load_graph_from_csv exists and does exactly the same thing
def load_csv_graph(path: str, edge_indices: tuple[int, int] = (0, 1), titles: list[str] = None, vprop_name: str = 'name') -> graph_tool.Graph:
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
        vprop_name str: Name to give loaded vertex property map. Defaults to 'name'.

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

    vprop = new_graph.add_edge_list(edge_list, eprops=eprops, hashed=True)
    new_graph.vp[vprop_name] = vprop
    return new_graph, vprop


# def load_vertices(path: str, graph: graph_tool.Graph = None, vertex_index: int = 0, titles: list[str] = None, vprop_name: str = 'name') -> tuple[graph_tool.Graph, graph_tool.VerteXPropertyMap]:
#     # if not graph:
#     #     graph = graph_tool.Graph()
#     # if vprop_name not in graph.vp:
#     #     vp = graph.new_vp('string')
#     #     graph[vprop_name] = vp
#     # vprop = graph[vprop_name]

#     new_graph = graph_tool.Graph()
#     vertex_list = []
#     with open(path, newline="") as csvfile:
#         csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
#         if titles is None: titles = next(csv_reader)
#         for line in csv_reader:
#             props = [prop for idx, prop in enumerate(line) if idx != vertex_index]
#             v = line[vertex_index]
#             vertex = (v, *props)
#             vertex_list.append(vertex)
    
#     titles = [title for idx, title in enumerate(titles) if idx != vertex_index]
#     vprops = [new_graph.new_ep("string") for _ in titles]
#     for vp, title in zip(vprops, titles):
#         new_graph.ep[title] = vp
    
#     return new_graph
#     # return new_graph, vprop
#     # vprop = new_graph.add_vertex(vertex_list, eprops=eprops, hashed=True)
#     # new_graph.vp[vprop_name] = vprop


def combine_graphs(graphs: list[graph_tool.Graph], vprop_name: str = 'name') -> graph_tool.Graph:
    """Combines several graphs into a single one.

    Args:
        graphs (list[graph_tool.Graph]): List of graphs to combine.
        vprop_name (str): Name of the vertex property combined on. Defaults to 'name'.
    Returns:
        graph_tool.Graph: Combined graph
    """
    if not graphs:
        return graph_tool.Graph()

    vprops = [g.vp[vprop_name] for g in graphs]
    vertices = {v for vp in vprops for v in vp}
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

    combined.vp[vprop_name] = nvprop
    return combined, nvprop
