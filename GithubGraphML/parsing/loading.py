from collections import defaultdict
from datetime import datetime
from graph_tool.all import *
from typing import *

import numpy as np
import csv
import pickle
import os


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

    nvprop = combined.new_vp(vprops[0].value_type(), vals=vertices)
    nvprop_map = {}
    for i, prop in enumerate(nvprop):
        nvprop_map[prop] = i

    for g, vprop in zip(graphs, vprops):
        for v, *vprops in zip(
            g.vertices(), *[[(key, prop) for prop in g.vp[key]] for key in g.vp]
        ):
            for key, prop in vprops:
                if key not in combined.vp:
                    val_type = g.vp[key].value_type()
                    vp = combined.new_vp(val_type)
                    combined.vp[key] = vp
                idx = nvprop_map[vprop[v]]
                combined.vp[key][idx] = prop

        for (v1, v2), *eprops in zip(
            g.edges(), *[[(key, prop) for prop in g.ep[key]] for key in g.ep]
        ):
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
    bipartite = graph.new_vp('bool', val=False)
    graph.vp['bipartite'] = bipartite
    vmap = {}

    # copy vertices to graph
    for vv, vg in zip(vertices.vertices(), graph.add_vertex(vertices.num_vertices())):
        for key in vertices.vp:
            if key not in graph.vp:
                value_type = vertices.vp[key].value_type()
                new_vp = graph.new_vp(value_type)
                graph.vp[key] = new_vp
            vv_prop = vertices.vp[key][vv]
            graph.vp[key][vg] = vv_prop
            vmap[vv] = vg
        bipartite[vg] = True

    # modify edges
    graph.set_fast_edge_removal(True)
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
    graph.set_fast_edge_removal(False)

def merge_parallel(graph: Graph, eprops: list[VertexPropertyMap]) -> None:
    # Assumes eprops are numeric & graph is undirected
    def get_parallel_edges(g):
        edge_dict = defaultdict(list)
        for e in g.edges():
            key = tuple(sorted((int(e.source()), int(e.target()))))
            edge_dict[key].append(e)

        parallel_edges = {k: v for k, v in edge_dict.items() if len(v) > 1}
        return parallel_edges

    parallel_edges = get_parallel_edges(graph)
    merged = graph.new_ep('bool', val=True)
    graph.ep['merged'] = merged
    for edge, edges in parallel_edges.items():
        edge = graph.add_edge(*edge)
        for eprop in eprops:
            merged[edges[0]] = False
            prop = eprop[edges[0]]
            for e in edges[1:]:
                prop += eprop[e]
                merged[e] = False
            eprop[edge] = prop

        merged[edge] = True

    graph.set_edge_filter(merged)

def prune(graph: Graph) -> VertexPropertyMap:
    has_neighbors = graph.new_vp('bool')
    for v in graph.vertices():
        has_neighbors[v] = v.in_degree() + v.out_degree() > 0

    graph.vp['has_neighbors'] = has_neighbors
    graph.set_vertex_filter(has_neighbors)


date2float = lambda date: datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp()
lang_map = {
    '': -1,
    'Assembly': 0,
    'JavaScript': 1,
    'Pascal': 2,
    'Perl': 3,
    'Python': 4,
    'Ruby': 5,
    'VisualBasic': 6,
}


def load_repositories(data_dir, cache='repositories.pkl', use_cached = True, cache_result = True, verbose = True):
    if not os.path.exists(cache) or not use_cached:
        if verbose: print('Loading repositories...')
        repositories, _ = load_csv_vertices(f'{data_dir}/repositories.csv', vprop_name='repository_id')
        repositories.vp['programming_language_id'] = repositories.vp['programming_language_id'].t(int, value_type='int')
        repositories.vp['duration_days'] = repositories.vp['duration_days'].t(int, value_type='int')
        repositories.vp['number_commits'] = repositories.vp['number_commits'].t(int, value_type='int')
        repositories.vp['number_commiters'] = repositories.vp['number_commiters'].t(int, value_type='int')
        repositories.vp['create_date'] = repositories.vp['create_date'].t(date2float, value_type='float')
        repositories.vp['end_date'] = repositories.vp['end_date'].t(date2float, value_type='float')
        repositories.vp['repository_id'] = repositories.vp['repository_id'].t(int, value_type='int')
        repositories.shrink_to_fit()
        if cache_result:
            if verbose: print('Caching repositories...')
            with open(cache, 'wb') as f:
                pickle.dump(repositories, f)
    else: 
        if verbose: print('Loading cached repositories...')
        with open(cache, 'rb') as f:
            repositories = pickle.load(f)
    return repositories

def load_developers(data_dir, cache='developers.pkl', use_cached=True, cache_result=True, verbose = True):
    if not os.path.exists(cache) or not use_cached:
        if verbose: print('Loading developers...')
        developers, _ = load_csv_vertices(f'{data_dir}/developer.csv', vprop_name='developer_id')
        def timestamp(date):  # format different & defect in dataset
            if date == ',2012-05-0': date = '2012-05-01'
            date = datetime.strptime(date, "%Y-%m-%d")
            return date.timestamp()
        def loc(loc):
            if loc: return float(loc)
            else: return 0.0
        developers.vp['created_at'] = developers.vp['created_at'].t(timestamp, value_type='float')
        developers.vp['fake'] = developers.vp['fake'].t(int, value_type='bool')
        developers.vp['deleted'] = developers.vp['deleted'].t(int, value_type='bool')
        developers.vp['developer_id'] = developers.vp['developer_id'].t(int, value_type='int')
        developers.vp['is_org'] = developers.vp['usr_type'].t(lambda type: type == 'ORG', value_type='bool')
        developers.vp['long'] = developers.vp['long'].t(loc, value_type='float')
        developers.vp['lat'] = developers.vp['lat'].t(loc, value_type='float')
        del developers.vp['company']
        del developers.vp['usr_type']
        del developers.vp['country_code']
        del developers.vp['location']
        del developers.vp['state']
        del developers.vp['city']
        developers.shrink_to_fit()
        if cache_result:
            if verbose: print('Caching developers...')
            with open(cache, 'wb') as f:
                pickle.dump(developers, f)
    else: 
        if verbose: print('Loading cached developers...')
        with open(cache, 'rb') as f:
            developers = pickle.load(f)
    return developers

def load_networks(data_dir, languages=['Assembly', 'JavaScript', 'Pascal', 'Perl', 'Python', 'Ruby', 'VisualBasic'], enhance=True, cache='%s.pkl', use_cache=True, cache_result=True, developers_cache='developers.pkl', verbose=True):
    if enhance: 
        if developers_cache is not None: developers = load_developers(data_dir, cache=developers_cache, verbose=verbose)
        else: developers = load_developers(data_dir, use_cached=False, cache_result=False, verbose=verbose)
    developers_social_networks = []
    for language in languages:
        if not os.path.exists(cache % language) or not use_cache:
            if verbose: print(f'Loading {language} network...')
            csv_template = f'{data_dir}/developers_social_network/%s_developers_social_network.csv'
            graph, _ = load_csv_graph(csv_template % language.upper(), (1, 2), vprop_name='developer_id')
            graph.ep['programming_language_id'] = graph.new_ep('int', val=lang_map[language])
            graph.ep['repository_id'] = graph.ep['repository_id'].t(int, value_type='int')
            graph.ep['contribution_days'] = graph.ep['contribution_days'].t(int, value_type='int')
            graph.ep['number_commits'] = graph.ep['number_commits'].t(int, value_type='int')
            graph.ep['end_contribution_date'] = graph.ep['end_contribution_date'].t(date2float, value_type='float')
            graph.ep['begin_contribution_date'] = graph.ep['begin_contribution_date'].t(date2float, value_type='float')
            graph.vp['developer_id'] = graph.vp['developer_id'].t(int, value_type='int')
            if enhance:
                if verbose: print(f'Enhancing {language} network...')
                for key in developers.vp: 
                    if key not in graph.vp:
                        dev_vp = developers.vp[key]
                        val_type = dev_vp.value_type()
                        g_vp = graph.new_vp(val_type)
                        graph.vp[key] = g_vp
                dev_id_prop = developers.vp['developer_id']
                graph_id_prop = graph.vp['developer_id']
                dev_map = {}
                for v in developers.vertices():
                    dev_id = dev_id_prop[v]
                    dev_map[dev_id] = v
                for v in graph.vertices():
                    dev_id = graph_id_prop[v]
                    try:
                        dev_v = dev_map[dev_id]
                        for key in developers.vp:
                            dev_vp = developers.vp[key]
                            graph_vp = graph.vp[key]
                            graph_vp[v] = dev_vp[dev_v]
                    except KeyError:
                        print(f'Missing developer: {dev_id}')
                        continue
            graph.set_directed(False)
            graph.shrink_to_fit()
            if cache_result:
                if verbose: print(f'Cached {language} network...')
                with open(cache % language, 'wb') as f:
                    pickle.dump(graph, f)
        else:
            if verbose: print(f'Loading cached {language} network...')
            with open(cache % language, 'rb') as f:
                graph = pickle.load(f)
        developers_social_networks.append(graph)
    return developers_social_networks

def load_combined(data_dir, languages: list[str], cache = 'combined.pkl', use_cached=True, cache_result=True, verbose=True):
    if not os.path.exists(cache) or not use_cached:
        network_list = load_networks(data_dir, languages, verbose=verbose)

        if verbose: print('Combining networks...')
        combined, _ = combine_graphs(network_list, vprop_name='developer_id')
        combined.set_directed(False)
        combined.shrink_to_fit()
        if cache_result:
            if verbose: print('Caching combination...')
            with open(cache, 'wb') as f:
                pickle.dump(combined, f)
    else: 
        if verbose: print('Loading cached combination...')
        with open(cache, 'rb') as f:
            combined = pickle.load(f)
    return combined

def load_bipartite(data_dir, languages: list[str], purge=True, merge=True, cache='bipartite.pkl', use_cached=True, cache_result=True, combined_cache='combined.pkl', repository_cache='repositories.pkl', verbose=True):
    if not os.path.exists(cache) or not use_cached:
        if combined_cache is not None: bipartite = load_combined(data_dir, languages, cache=combined_cache, verbose=verbose) 
        else: bipartite = load_combined(data_dir, languages, use_cached=False, cache_result=False, vebose=verbose)
        if repository_cache is not None: repositories = load_repositories(data_dir, cache=repository_cache, verbose=verbose)
        else: repositories = load_repositories(data_dir, use_cached=False, cache_result=False, verbose=verbose)
        if verbose: print('Transforming combined...')
        transform_bipartite(bipartite, repositories, 'repository_id')
        bipartite.vp['is_repository'] = bipartite.vp['bipartite']
        del bipartite.vp['bipartite']
        del repositories
        if purge:
            if verbose: print('Pruning bipartite...')
            prune(bipartite)
            bipartite.purge_vertices()
            del bipartite.vp['has_neighbors']
        if merge:
            if verbose: print('Merging parallel edges...')
            merge_parallel(bipartite, [bipartite.ep['number_commits'], bipartite.ep['contribution_days']])
            bipartite.purge_edges()
            del bipartite.ep['merged']
        if verbose: print('dumping bipartite...')
        if cache_result:
            with open(cache, 'wb') as f:
                pickle.dump(bipartite, f)
    else:
        if verbose: print('Loading cached bipartite...')
        with open(cache, 'rb') as f:
            bipartite = pickle.load(f)
    return bipartite
