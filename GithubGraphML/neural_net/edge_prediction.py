import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tqdm

# Import custom modules from your GitHubGraphML package.
from GithubGraphML.analyze import load_networks
from GithubGraphML.parsing.loading import combine_graphs

###########################################
# Data Loading & Processing Functions
###########################################

def load_combined_graph(language_list, pickle_filename, use_cached=True, cache_combined=True):
    """
    Load combined graph from pickle or create it from individual networks.
    The graph is set to undirected. Also prints some sample edge properties.
    """
    if os.path.exists(pickle_filename) and use_cached:
        print("Loading cached combination...")
        with open(pickle_filename, 'rb') as f:
            combined = pickle.load(f)
            combined.set_directed(False)
    else:
        print("Combining graphs...")
        graph_list = load_networks(language_list, vprop_name='id')
        combined, _ = combine_graphs(graph_list, vprop_name='id')
        combined.set_directed(False)
        if cache_combined:
            print("Caching combined graph...")
            with open(pickle_filename, 'wb') as f:
                pickle.dump(combined, f)
    print("Graph properties:", combined.list_properties())
    edge = next(combined.edges())
    print(f"Inspecting edge from {int(edge.source())} to {int(edge.target())}:")
    for prop_name in combined.ep.keys():
        value = combined.ep[prop_name][edge]
        print(f"  {prop_name}: {value}")
    return combined

def build_edge_index(graph):
    """Create edge_index tensor from the graph-tools graph (undirected)."""
    edge_list = []
    for e in graph.edges():
        src = int(e.source())
        tgt = int(e.target())
        edge_list.append([src, tgt])
        edge_list.append([tgt, src])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def build_node_features(graph, num_nodes):
    """
    Build node feature matrix including:
      - A constant feature (1)
      - Sum of contribution_days (converted from string to float) across incident edges
      - Sum of number_commits (converted from string to float) across incident edges
    """
    # Constant feature: a column of ones.
    constant_feature = torch.ones((num_nodes, 1), dtype=torch.float)
    
    # Initialize accumulators for edge-based attributes.
    total_contribution_days = torch.zeros(num_nodes, dtype=torch.float)
    total_commits = torch.zeros(num_nodes, dtype=torch.float)
    
    # Iterate over every edge in the graph.
    for e in tqdm.tqdm(graph.edges()):
        src = int(e.source())
        tgt = int(e.target())
        # Convert string edge properties to floats.
        cd_val = float(graph.ep['contribution_days'][e])
        commit_val = float(graph.ep['number_commits'][e])
        
        # Since the graph is undirected, add to both endpoints.
        total_contribution_days[src] += cd_val
        total_contribution_days[tgt] += cd_val
        total_commits[src] += commit_val
        total_commits[tgt] += commit_val
        
    # Reshape accumulators into column vectors.
    total_contribution_days = total_contribution_days.view(num_nodes, 1)
    total_commits = total_commits.view(num_nodes, 1)
    
    # Concatenate to form a node feature matrix of shape (num_nodes, 3).
    x = torch.cat([constant_feature, total_contribution_days, total_commits], dim=1)
    return x

###########################################
# Data Splitting Functions (by Node Fraction)
###########################################

# Use the same fractions as in your node classification code.
DATA_FRACTION = 0.2    # Fraction of all nodes to use.
TRAIN_FRACTION = 0.8   # Fraction of the selected nodes to be in training.

def split_data(data, data_fraction, train_fraction):
    """
    Split nodes into training and evaluation sets.
    Adds boolean masks 'train_mask' and 'eval_mask' to the data object.
    """
    N = data.num_nodes
    selected_count = int(data_fraction * N)
    perm = torch.randperm(N)
    selected_indices = perm[:selected_count]
    train_count = int(train_fraction * selected_count)
    train_indices = selected_indices[:train_count]
    eval_indices = selected_indices[train_count:]
    
    train_mask = torch.zeros(N, dtype=torch.bool)
    eval_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_indices] = True
    eval_mask[eval_indices] = True
    
    data.train_mask = train_mask
    data.eval_mask = eval_mask
    return data

def filter_edges_by_mask(edge_index, mask):
    """
    Filters edges from edge_index, keeping only those edges for which
    both endpoints have mask==True.
    
    Args:
        edge_index (torch.Tensor): shape [2, num_edges].
        mask (torch.Tensor): boolean tensor of shape [num_nodes].
        
    Returns:
        torch.Tensor: filtered edge_index.
    """
    valid = mask[edge_index[0]] & mask[edge_index[1]]
    return edge_index[:, valid]

def negative_sampling_subgraph(pos_edge_index, node_subset, num_neg_samples):
    """
    Samples negative edges from within a node subset.
    Args:
         pos_edge_index (torch.Tensor): shape [2, num_edges] of positive edges.
         node_subset (torch.Tensor): 1D tensor with the indices of nodes to sample from.
         num_neg_samples (int): number of negative edges to sample.
    Returns:
         torch.Tensor: shape [2, num_neg_samples] with negative edge indices.
    """
    pos_edges_set = set([(u.item(), v.item()) for u, v in zip(pos_edge_index[0], pos_edge_index[1])])
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        u = node_subset[torch.randint(0, len(node_subset), (1,)).item()].item()
        v = node_subset[torch.randint(0, len(node_subset), (1,)).item()].item()
        if u == v:
            continue
        if (u, v) in pos_edges_set or (v, u) in pos_edges_set:
            continue
        neg_edges.append([u, v])
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
    return neg_edge_index

###########################################
# Data Preparation
###########################################

# Define parameters: languages to consider and pickle file name.
language_list = ['Assembly', 'Javascript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
combined_pickle = 'combined.pkl'

# 1. Load the combined GitHub graph.
combined_graph = load_combined_graph(language_list, combined_pickle)

# 2. Get the number of nodes.
num_nodes = combined_graph.num_vertices()

# 3. Build node features using edge property aggregation.
print("Building features")
x = build_node_features(combined_graph, num_nodes)

# 4. Build the edge_index tensor.
print("Edge index")
edge_index = build_edge_index(combined_graph)

# 5. Create a PyTorch Geometric Data object.
data = Data(x=x, edge_index=edge_index)

# 6. Split nodes into training and evaluation sets.
data = split_data(data, DATA_FRACTION, TRAIN_FRACTION)
print(f"Training nodes: {data.train_mask.sum().item()} / {data.num_nodes}")
print(f"Evaluation nodes: {data.eval_mask.sum().item()} / {data.num_nodes}")

# 7. Build training and evaluation positive edge indices by filtering the full edge index.
train_edge_index = filter_edges_by_mask(data.edge_index, data.train_mask)
eval_edge_index = filter_edges_by_mask(data.edge_index, data.eval_mask)

###########################################
# Model Initialization for Edge Prediction
###########################################

class GCNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Our node features are of dimension 3.
        self.conv1 = GCNConv(3, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

def decode(z, edge_index):
    """
    Compute the edge score via a dot product between source and target embeddings.
    """
    edge_index = edge_index.long()
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

model = GCNEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

###########################################
# Training Loop with Metric Logging
###########################################

num_epochs = 100
losses = []
accuracies = []
precisions = []
recalls = []
f1s = []

# Get indices for training nodes for our custom negative sampling.
train_nodes = torch.where(data.train_mask)[0]

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Use the training subgraph (train_edge_index) for message passing.
    z = model(data.x, train_edge_index)
    
    # Positive training edges.
    pos_edge = train_edge_index
    pos_score = decode(z, pos_edge)
    pos_label = torch.ones(pos_score.size(0))
    
    # Negative training edges sampled from within training nodes.
    neg_edge = negative_sampling_subgraph(pos_edge, train_nodes, pos_edge.size(1))
    neg_score = decode(z, neg_edge)
    neg_label = torch.zeros(neg_score.size(0))
    
    # Concatenate scores and labels.
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([pos_label, neg_label], dim=0)
    
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    ###########################################
    # Compute Training Metrics
    ###########################################
    model.eval()
    with torch.no_grad():
        z_eval = model(data.x, train_edge_index)
        pos_score_eval = decode(z_eval, pos_edge)
        
        neg_edge_eval = negative_sampling_subgraph(pos_edge, train_nodes, pos_edge.size(1))
        neg_score_eval = decode(z_eval, neg_edge_eval)
        
        scores_eval = torch.cat([pos_score_eval, neg_score_eval], dim=0)
        probs = torch.sigmoid(scores_eval)
        preds = (probs > 0.5).float().cpu().numpy()
        true_labels = torch.cat([pos_label, neg_label], dim=0).cpu().numpy()
        
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1_val = f1_score(true_labels, preds, zero_division=0)
    
    losses.append(loss.item())
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1_val)
    
    model.train()
    if epoch % 10 == 0:
        print(f"[Edge Prediction] Epoch {epoch}, Loss: {loss.item():.4f}, "
              f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1_val:.4f}")

###########################################
# Test Evaluation on Held-out Edges
###########################################

# For evaluation, restrict to edges among evaluation nodes.
eval_nodes = torch.where(data.eval_mask)[0]

model.eval()
with torch.no_grad():
    # Compute embeddings using the training subgraph (to mimic no leakage).
    z_test = model(data.x, train_edge_index)
    
    # Evaluation positive edges are those among evaluation nodes.
    pos_eval_edge = eval_edge_index
    test_pos_score = decode(z_test, pos_eval_edge).sigmoid()
    
    # Sample negative evaluation edges from within evaluation nodes.
    neg_eval_edge = negative_sampling_subgraph(pos_eval_edge, eval_nodes, pos_eval_edge.size(1))
    test_neg_score = decode(z_test, neg_eval_edge).sigmoid()
    
    print("\n[Edge Prediction] Test positive edge scores:", test_pos_score.tolist())
    print("[Edge Prediction] Test negative edge scores:", test_neg_score.tolist())

###########################################
# Plotting Training Metrics
###########################################

epochs = range(num_epochs)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (Edge Prediction)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, label='Accuracy')
plt.plot(epochs, precisions, label='Precision')
plt.plot(epochs, recalls, label='Recall')
plt.plot(epochs, f1s, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Metric Score')
plt.title('Training Metrics (Edge Prediction)')
plt.legend()

plt.tight_layout()
plt.show()
