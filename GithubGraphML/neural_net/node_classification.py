import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from GithubGraphML.analyze import load_networks
from GithubGraphML.parsing.loading import combine_graphs

# Split constants
DATA_FRACTION = 0.5    # Fraction of nodes from the entire graph to use
TRAIN_FRACTION = 0.8   # Fraction of selected nodes that go to training

def load_combined_graph(language_list, pickle_filename, use_cached=True, cache_combined=True):
    """Load combined graph from pickle or create it from individual networks."""
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

    # Print a header indicating which edge we are inspecting
    print(f"Inspecting edge from {int(edge.source())} to {int(edge.target())}:\n")

    # Loop over all edge properties and print their values for this edge
    for prop_name in combined.ep.keys():
        value = combined.ep[prop_name][edge]
        print(f"{prop_name}: {value}")

    return combined

def extract_node_labels(graph, language_list):
    """Extract multi-label vectors based on edge 'language' property."""
    num_nodes = graph.num_vertices()
    lang_prop = graph.ep['language']
    node_languages = {int(v): set() for v in graph.vertices()}
    for e in graph.edges():
        lang_val = lang_prop[e]
        src = int(e.source())
        tgt = int(e.target())
        node_languages[src].add(lang_val)
        node_languages[tgt].add(lang_val)
    language_to_idx = {lang: i for i, lang in enumerate(language_list)}
    num_languages = len(language_list)
    labels_list = []
    for i in range(num_nodes):
        vec = [0] * num_languages
        for lang in node_languages[i]:
            if lang in language_to_idx:
                vec[language_to_idx[lang]] = 1
        labels_list.append(vec)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float)
    return labels_tensor, num_nodes

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

def build_data_object(x, edge_index, y):
    """Build PyTorch Geometric Data object."""
    return Data(x=x, edge_index=edge_index, y=y)

def split_data(data, data_fraction, train_fraction):
    """Split nodes into training and evaluation sets based on given fractions."""
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

def build_node_features(graph, num_nodes):
    """
    Build node feature matrix including:
      - a constant feature (1)
      - sum of contribution_days (converted from string to float) across incident edges
      - sum of number_commits (converted from string to float) across incident edges
    """
    # Constant feature: a column of ones.
    constant_feature = torch.ones((num_nodes, 1), dtype=torch.float)
    
    # Initialize tensors for accumulating edge properties per node.
    total_contribution_days = torch.zeros(num_nodes, dtype=torch.float)
    total_commits = torch.zeros(num_nodes, dtype=torch.float)
    
    # Iterate over all edges in the graph.
    for e in graph.edges():
        src = int(e.source())
        tgt = int(e.target())
        # Convert the string edge properties to float values.
        cd_val = float(graph.ep['contribution_days'][e])
        commit_val = float(graph.ep['number_commits'][e])
        
        # Since the graph is undirected, add to both endpoints.
        total_contribution_days[src] += cd_val
        total_contribution_days[tgt] += cd_val
        total_commits[src] += commit_val
        total_commits[tgt] += commit_val
        
    # Reshape the accumulated vectors into column tensors.
    total_contribution_days = total_contribution_days.view(num_nodes, 1)
    total_commits = total_commits.view(num_nodes, 1)
    
    # Concatenate the constant feature with the accumulated edge properties.
    x = torch.cat([constant_feature, total_contribution_days, total_commits], dim=1)
    return x

def create_model(num_features, num_languages, hidden_dim=4):
    """Create a simple 2-layer GCN for multi-label classification."""
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_languages)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x
    return GCN()

def train_model(model, data, num_epochs=100):
    """Train model and log metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1s = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate on training set.
        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            pred = (torch.sigmoid(out_eval) > 0.5).float()
            y_train = data.y[data.train_mask].cpu().numpy().flatten()
            pred_train = pred[data.train_mask].cpu().numpy().flatten()
            acc = accuracy_score(y_train, pred_train)
            prec = precision_score(y_train, pred_train, average='micro', zero_division=0)
            rec = recall_score(y_train, pred_train, average='micro', zero_division=0)
            f1_val = f1_score(y_train, pred_train, average='micro', zero_division=0)
        losses.append(loss.item())
        train_accuracies.append(acc)
        train_precisions.append(prec)
        train_recalls.append(rec)
        train_f1s.append(f1_val)

        model.train()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, "
                  f"Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1_val:.4f}")
    metrics = {
        'losses': losses,
        'accuracies': train_accuracies,
        'precisions': train_precisions,
        'recalls': train_recalls,
        'f1s': train_f1s
    }
    return metrics

def plot_metrics(metrics, num_epochs, filename):
    """Plot training loss and metrics and save the plot."""
    epochs_range = range(num_epochs)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics['losses'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Multi-label)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics['accuracies'], label='Accuracy')
    plt.plot(epochs_range, metrics['precisions'], label='Precision')
    plt.plot(epochs_range, metrics['recalls'], label='Recall')
    plt.plot(epochs_range, metrics['f1s'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Metrics (Multi-label)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    language_list = ['Assembly', 'Javascript', 'Pascal', 'Perl', 'Python', 'VisualBasic']
    combined_pickle = 'combined.pkl'
    num_epochs = 50

    # 1. Load/Combine the graph.
    combined_graph = load_combined_graph(language_list, combined_pickle)

    # 2. Extract multi-label node targets.
    y, num_nodes = extract_node_labels(combined_graph, language_list)
    print(f"Extracted labels shape: {y.shape}")

    # 3. Build node features including constant, contribution_days, and number_commits.
    x = build_node_features(combined_graph, num_nodes)
    
    # 4. Build edge_index tensor.
    edge_index = build_edge_index(combined_graph)

    # 5. Create the PyG data object.
    data = build_data_object(x, edge_index, y)

    # 6. Split nodes into training and evaluation sets.
    data = split_data(data, DATA_FRACTION, TRAIN_FRACTION)
    print(f"Training nodes: {data.train_mask.sum().item()} / {data.num_nodes}")
    print(f"Eval nodes: {data.eval_mask.sum().item()} / {data.num_nodes}")

    # 7. Create the model. Update input features from 1 to 3.
    model = create_model(num_features=3, num_languages=len(language_list))

    # 8. Train the model.
    metrics = train_model(model, data, num_epochs=num_epochs)

    # 9. Final evaluation on the evaluation set.
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = (torch.sigmoid(out) > 0.5).float()
        y_eval = data.y[data.eval_mask].cpu().numpy().flatten()
        pred_eval = pred[data.eval_mask].cpu().numpy().flatten()
        print(y_eval)
        print(pred_eval)
        eval_acc = accuracy_score(y_eval, pred_eval)
        eval_prec = precision_score(y_eval, pred_eval, average='macro', zero_division=0)
        eval_rec = recall_score(y_eval, pred_eval, average='macro', zero_division=0)
        eval_f1 = f1_score(y_eval, pred_eval, average='macro', zero_division=0)
    print(f"\n[Final Evaluation] Eval Acc: {eval_acc:.4f}, Prec: {eval_prec:.4f}, Rec: {eval_rec:.4f}, F1: {eval_f1:.4f}")

    # 10. Plot and save training metrics.
    plot_metrics(metrics, num_epochs, 'multi_label_node_classification_metrics.png')

if __name__ == "__main__":
    main()
