from graph_tool.all import *
from typing import *
import csv

def load_csv_graph(path: str, edge_indices: tuple[int, int] = (0, 1), titles: list[str] = None, vprop_name: str = 'name') -> tuple[Graph, VertexPropertyMap]:
    """Loads a graph from a provided CSV file.

    This function reads a CSV file representing a graph, extracts edges based on 
    the specified column indices, and loads them into a `Graph` object.

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
        Graph: Loaded Graph.
        VertexPropertyMap: Loaded vertex property map.
    """
    new_graph = Graph()
    edge_list = []
    with open(path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        if titles is None: 
            titles = [title for idx, title in enumerate(next(csv_reader)) if idx not in edge_indices]

        eprops = [new_graph.new_ep("string") for _ in titles]
        for ep, title in zip(eprops, titles):
            new_graph.ep[title] = ep

        for line in csv_reader:
            props = [prop for idx, prop in enumerate(line) if idx not in edge_indices]
            v1 = line[edge_indices[0]]
            v2 = line[edge_indices[1]]
            edge = (v1, v2, *props)
            edge_list.append(edge)

    vprop = new_graph.add_edge_list(edge_list, eprops=eprops, hashed=True)
    new_graph.vp[vprop_name] = vprop
    return new_graph, vprop


def load_csv_vertices(path: str, vertex_index: int = 0, titles: list[str] = None, vprop_name: str = 'name') -> tuple[Graph, VertexPropertyMap]:
    new_graph = Graph()
    vprop = new_graph.new_vp('string')
    with open(path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        if titles is None: 
            titles = [title for idx, title in enumerate(next(csv_reader)) if idx != vertex_index]

        vprops = [new_graph.new_vp("string") for _ in titles]
        for vp, title in zip(vprops, titles):
            new_graph.vp[title] = vp

        for line in csv_reader:
            v = new_graph.add_vertex()
            vprop[v] = line[vertex_index]
            ij = 0
            for ii, prop in enumerate(line):
                if ii != vertex_index:
                    vp = vprops[ij]
                    vp[v] = prop
                    ij += 1
    
    new_graph.vp[vprop_name] = vprop
    return new_graph, vprop


def combine_graphs(graphs: list[Graph], vprop_name: str = 'name') -> tuple[Graph, VertexPropertyMap]:
    """Combines several graphs into a single one.

    Args:
        graphs (list[Graph]): List of graphs to combine.
        vprop_name (str): Name of the vertex property combined on. Defaults to 'name'.
    Returns:
        Graph: Combined graph
    """
    if not graphs:
        return Graph()

    vprops = [g.vp[vprop_name] for g in graphs]
    vertices = {v for vp in vprops for v in vp}
    combined = Graph(len(vertices))
    combined.set_directed(graphs[0].is_directed())

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

def transform_bipartite(graph: Graph, vertices: Graph, prop_name: str) -> None:
    """
    Transform a graph into a bipartite graph in place by inserting 
    a vertex between all edges based on on a property. The edges 
    affected essentially are duplicated.

    All uneffected are edges are removed.
    """
    vprop_map = {vertices.vp[prop_name][v]: v for v in vertices.vertices()}
    bipartite = graph.new_vp('int')
    graph.vp['bipartite'] = bipartite
    vmap = {}

    # copy vertices to graph
    for vv, vg in zip(vertices.vertices(), graph.add_vertex(vertices.num_vertices())):
        for key in vertices.vp:
            if key not in graph.vp:
                new_vp = graph.new_vp('string')
                graph.vp[key] = new_vp
            vv_prop = vertices.vp[key][vv]
            graph.vp[key][vg] = vv_prop
            vmap[vv] = vg
        bipartite[vg] = 1

    # modify edges
    for e in graph.get_edges():
        prop = graph.ep[prop_name][e]
        vv = vprop_map[prop]
        vg = vmap[vv]
        for v in e:
            ne = graph.add_edge(v, vg)
            for key in graph.ep:
                eprop = graph.ep[key][e]
                graph.ep[key][ne] = eprop
        graph.remove_edge(tuple(e))
