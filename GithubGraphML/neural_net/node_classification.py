import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from GithubGraphML.analyze import load_networks
from GithubGraphML.parsing.loading import combine_graphs

# Configuration constants
DATA_FRACTION: float = 1
TRAIN_FRACTION: float = 0.8
MODEL_PATH: Path = Path("gnn_model.pth")
CACHE_PATH: Path = Path("combined.pkl")


def save_model(model: torch.nn.Module, path: Path) -> None:
    """
    Save the model's state dictionary to disk.

    Args:
        model: The PyTorch model to save.
        path: File system path where the state dict will be written.
    """
    torch.save(model.state_dict(), path)


def load_model(
    model: torch.nn.Module, path: Path, device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a state dictionary from disk into the given model.

    Args:
        model: An uninitialized or existing PyTorch model instance.
        path: File system path to load the state dict from.
        device: Optional torch.device for map_location. Defaults to CPU if None.

    Returns:
        The model with loaded parameters.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def load_graph(
    languages: List[str], cache_path: Path, use_cache: bool = True, cache: bool = True
) -> Any:
    """
    Load or combine multiple language graphs into an undirected graph.

    Args:
        languages: Sequence of language names to include.
        cache_path: Path to a pickle file for caching the combined graph.
        use_cache: If True and cache_path exists, load from cache.
        cache: If True, write a new combined graph to cache_path.

    Returns:
        A graph object with directedness set to False.
    """
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as f:
            graph = pickle.load(f)
    else:
        graphs = load_networks(languages, vprop_name="id")
        graph, _ = combine_graphs(graphs, vprop_name="id")
        if cache:
            cache_path.parent.mkdir(exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(graph, f)
    graph.set_directed(False)
    return graph


def extract_labels(graph: Any, language_list: List[str]) -> torch.Tensor:
    """
    Construct multi-hot labels for each node based on edge 'language'.

    Args:
        graph: Graph object with edge property 'language'.
        language_list: Ordered list of language names mapping to vector indices.

    Returns:
        A tensor of shape (num_nodes, num_languages) with 0/1 entries.
    """
    num_nodes = graph.num_vertices()
    lang_to_idx = {lang: i for i, lang in enumerate(language_list)}
    labels = torch.zeros((num_nodes, len(language_list)), dtype=torch.float)
    for e in graph.edges():
        lang = graph.ep["language"][e]
        idx = lang_to_idx.get(lang)
        if idx is not None:
            s, t = int(e.source()), int(e.target())
            labels[s, idx] = 1.0
            labels[t, idx] = 1.0
    return labels


def build_node_features(graph: Any) -> torch.Tensor:
    """
    Build node-level features: constant plus summed edge properties.

    Args:
        graph: Graph object with edge properties 'contribution_days' and 'number_commits'.

    Returns:
        A tensor of shape (num_nodes, 3) with columns [1, sum_days, sum_commits].
    """
    n = graph.num_vertices()
    feats = torch.ones((n, 3), dtype=torch.float)
    for e in graph.edges():
        s, t = int(e.source()), int(e.target())
        cd: float = float(graph.ep["contribution_days"][e])
        cm: float = float(graph.ep["number_commits"][e])
        feats[s, 1] += cd
        feats[t, 1] += cd
        feats[s, 2] += cm
        feats[t, 2] += cm
    return feats


def build_edge_index(graph: Any) -> torch.Tensor:
    """
    Convert graph edges to PyG-style edge_index.

    Args:
        graph: Graph object with vertices and edges.

    Returns:
        A tensor of shape (2, num_edges*2) for bidirectional edges.
    """
    edges = [(int(e.source()), int(e.target())) for e in graph.edges()]
    src, dst = zip(*edges)
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def split_masks(
    num_nodes: int, data_frac: float, train_frac: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate boolean masks for train/eval splits.

    Args:
        num_nodes: Total number of nodes.
        data_frac: Fraction of nodes to select.
        train_frac: Fraction of selected nodes for training.

    Returns:
        A tuple (train_mask, eval_mask), each a boolean tensor.
    """
    perm = torch.randperm(num_nodes)
    sel = perm[: int(data_frac * num_nodes)]
    split = int(train_frac * len(sel))
    train_idx, eval_idx = sel[:split], sel[split:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    eval_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    eval_mask[eval_idx] = True
    return train_mask, eval_mask


class GCNModel(torch.nn.Module):
    """
    Two-layer GCN for multi-label node classification.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """
        Args:
            in_dim: Number of input features per node.
            hidden_dim: Hidden layer dimensionality.
            out_dim: Number of output classes (multi-label).
        """
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through two GCN layers.

        Args:
            x: Node feature matrix (num_nodes, in_dim).
            edge_index: Edge index tensor (2, num_edges).

        Returns:
            Raw logits tensor of shape (num_nodes, out_dim).
        """
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def train_model(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
) -> Dict[str, List[float]]:
    """
    Train the GCN model and record metric history.

    Args:
        model: Initialized GCNModel.
        data: PyG Data object with x, edge_index, y.
        train_mask: Boolean mask for training nodes.
        epochs: Number of training epochs.
        lr: Learning rate for optimizer.

    Returns:
        Dictionary mapping metric names to lists of values per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history: Dict[str, List[float]] = {
        k: [] for k in ["loss", "acc", "prec", "rec", "f1"]
    }
    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            y_true = data.y[train_mask].cpu().numpy()
            y_pred = pred[train_mask].cpu().numpy()

        history["loss"].append(loss.item())
        history["acc"].append(accuracy_score(y_true, y_pred))
        history["prec"].append(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        history["rec"].append(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        history["f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))

        if ep % 10 == 0:
            print(
                f"Epoch {ep}/{epochs} | Loss: {history['loss'][-1]:.4f} | F1: {history['f1'][-1]:.4f}"
            )
    return history


def plot_metrics(history: Dict[str, List[float]], save_path: Path) -> None:
    """
    Plot training metrics over epochs and save the figure.

    Args:
        history: Metric history from train_model.
        save_path: File path to write the plot image.
    """
    epochs = list(range(1, len(history["loss"]) + 1))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    for metric in ["acc", "prec", "rec", "f1"]:
        plt.plot(epochs, history[metric], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def evaluate_model(
    model: torch.nn.Module, data: Data, mask: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate model performance on a given mask.

    Args:
        model: Trained GCNModel.
        data: PyG Data containing features and labels.
        mask: Boolean mask for test or eval nodes.

    Returns:
        Dictionary of evaluation metrics (acc, prec, rec, f1).
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = (torch.sigmoid(logits) > 0.5).float()
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    """
    Full pipeline: load data, train GCN, persist model, reload and evaluate.
    """
    # Data preparation
    languages: List[str] = [
        "Assembly",
        "Javascript",
        "Pascal",
        "Perl",
        "Python",
        "VisualBasic",
    ]
    graph = load_graph(languages, CACHE_PATH)
    data = Data(
        x=build_node_features(graph),
        edge_index=build_edge_index(graph),
        y=extract_labels(graph, languages),
    )
    train_mask, eval_mask = split_masks(data.num_nodes, DATA_FRACTION, TRAIN_FRACTION)
    data.train_mask, data.eval_mask = train_mask, eval_mask

    # Model training
    start_time = time.time()
    model = GCNModel(in_dim=data.x.size(1), hidden_dim=16, out_dim=len(languages))
    history = train_model(model, data, train_mask)
    print(f"Training complete. Saving model to {MODEL_PATH}")
    save_model(model, MODEL_PATH)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time}")

    # Model evaluation
    loaded_model = GCNModel(
        in_dim=data.x.size(1), hidden_dim=16, out_dim=len(languages)
    )
    load_model(loaded_model, MODEL_PATH, device=torch.device("cpu"))
    print("Model reloaded for evaluation.")
    metrics = evaluate_model(loaded_model, data, eval_mask)
    print(
        f"Eval | Acc: {metrics['acc']:.4f} | Prec: {metrics['prec']:.4f} |"
        f" Rec: {metrics['rec']:.4f} | F1: {metrics['f1']:.4f}"
    )

    # Plot results
    plot_metrics(history, Path("plotting/training_metrics.png"))


if __name__ == "__main__":
    main()
