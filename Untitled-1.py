# %%
import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tqdm

os.getcwd()

# %%
from GithubGraphML.analyze import load_networks
pa_network = load_networks('./data', ['Pascal'])[0]
pa_network.list_properties()
pa_network

# %%
from GithubGraphML.parsing.loading import load_bipartite
bipartite_network = load_bipartite('./data', ['Pascal'], cache='test_bipartite.pkl')
bipartite_network.clear_filters()
def thing(x):
    try:
        return int(x)
    except:
        return -1
    
bipartite_network.vp['number_commits'] = bipartite_network.vp['number_commits'].t(thing, value_type='int')
bipartite_network.vp['number_commiters'] = bipartite_network.vp['number_commiters'].t(thing, value_type='int')
bipartite_network.list_properties()
bipartite_network 


# %%
from torch_geometric.utils import to_undirected
def trasfrom_graph_tool(graph, eprops, reduce="mean"):
    edges = graph.get_edges(eprops)
    edge_index = torch.tensor(edges[:, :2], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edges[:, 2:], dtype=torch.float)
    return to_undirected(edge_index, edge_attr=edge_attr, reduce=reduce, )

indices, attrs = trasfrom_graph_tool(pa_network,[pa_network.ep['number_commits']], "sum")
indices, attrs

# %%
from torch_geometric.data import HeteroData
from graph_tool.all import *
from collections import defaultdict
import numpy as np

def transform_data(graph, vclass=[], vprops=[], eclass=[], eprops=[]):
    def filter_nodes(graph, vcls, vprops):
        return graph.get_vertices(vprops)[vcls.a.astype(bool)]
    def filter_edges(graph, ecls, vprops):
        return graph.get_edges(eprops)[ecls.a.astype(bool)]
    
    nodes = [filter_nodes(graph, vcls, vprops) for vcls in vclass] if vclass else [graph.get_vertices(vprops).reshape(-1, len(vprops) + 1)]
    edges = [filter_edges(graph, ecls, eprops) for ecls in eclass] if eclass else [graph.get_edges(eprops)]
    data = HeteroData()
    nmap = {}
    for idx, node_array in enumerate(nodes):
        nmap.update({int(oidx): (int(nidx), f'v{idx}') for nidx, oidx in enumerate(node_array[:, 0])})
        data[f'v{idx}'].x = torch.tensor(node_array[:, 1:], dtype=torch.float)
        data[f'v{idx}'].num_nodes = len(node_array)
        
    for idx, edge_array in enumerate(edges):
        # print(edge_array)
        src_nodes = edge_array[:, 0]
        dst_nodes = edge_array[:, 1]
        emap = defaultdict(list)
        for jdx in range(len(edge_array)):
            try:
                src, dst = edge_array[jdx, :2]
                src, src_cls = nmap[int(src)] 
                dst, dst_cls = nmap[int(dst)] 
                edge_array[jdx][0] = src
                edge_array[jdx][1] = dst
                emap[(src_cls, f'e{idx}', dst_cls)].append((src, dst))
            except:
                pass

        for rel, indices in emap.items():
            data[*rel].edge_index = torch.tensor(np.array(indices)[:, :2].T, dtype=torch.long)
            data[*rel].edge_attrs = torch.tensor(np.array(indices)[:, 2:].T, dtype=torch.long)
            data[*rel].edge_label = torch.ones(len(indices), dtype=torch.long)
            data[*rel].num_edges = len(indices)

    return data

data = transform_data(bipartite_network, vprops=[bipartite_network.vp['number_commits']], vclass=[bipartite_network.vp['is_repository'], bipartite_network.vp['is_repository'].t(np.logical_not)])
data

# %%
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader

# Define the GNN model
class HeteroGraphConvGNN(nn.Module):
    def __init__(self, data):
        super(HeteroGraphConvGNN, self).__init__()
        
        # Get input feature dimensions for node types ('v0', 'v1', etc.)
        in_channels_v0 = data['v0'].x.shape[1]
        in_channels_v1 = data['v1'].x.shape[1] if 'v1' in data else in_channels_v0  # Assuming 'v1' exists, else fall back to 'v0'
        
        out_channels = 64  # You can adjust this as needed

        # GraphConv layers for different node types
        self.conv_v0 = pyg_nn.GraphConv(in_channels_v0, out_channels)
        self.conv_v1 = pyg_nn.GraphConv(in_channels_v1, out_channels)
        
        # Final fully connected layer
        self.fc = nn.Linear(out_channels, 1)  # For output (e.g., regression or classification)
    
    def forward(self, data):
        # Node feature dictionary
        x_dict = {
            'v0': data['v0'].x,  # Node features for 'v0'
            'v1': data['v1'].x,  # Node features for 'v1' (if 'v1' exists in your data)
        }

        # Perform GraphConv for node type 'v0'
        x_dict['v0'] = self.conv_v0(x_dict['v0'], data['v0', 'e0', 'v1'].edge_index)  # 'v0' to 'v1' edge indices
        x_dict['v0'] = torch.relu(x_dict['v0'])  # Apply ReLU

        # Perform GraphConv for node type 'v1'
        x_dict['v1'] = self.conv_v1(x_dict['v1'], data['v1', 'e0', 'v0'].edge_index)  # 'v1' to 'v0' edge indices
        x_dict['v1'] = torch.relu(x_dict['v1'])  # Apply ReLU

        # Combine features from both node types (just concatenate in this example)
        x = torch.cat([x_dict['v0'], x_dict['v1']], dim=1)
        
        # Final output layer
        x = self.fc(x)
        return x

# Assuming that the data is already transformed into a HeteroData object
train_data = [data]  # Wrap the data in a list (or a batch if you have multiple graphs)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

# Initialize the GNN model
model = HeteroGraphConvGNN(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()  # Use an appropriate loss function based on your task

# Example training loop
for epoch in range(100):
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        
        # Assuming batch['v0'].y contains the target labels for node class 'v0'
        target = batch['v0'].y  # You may need to modify this based on your graph structure
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


