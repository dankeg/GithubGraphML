import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
import graph_tool.all as gt

CACHE_PATH = Path("bipartite.pkl")
MODEL_PATH = Path("bipartite_edge_model.pth")
GRAPHML_FILE = "pascal_bipartite.graphml"
TEST_FRAC = 0.2
EPOCHS = 100
LR = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bipartite_graph(graphml_path: str, cache_path: Path) -> gt.Graph:
    """
    Load or cache a bipartite graph from GraphML via graph-tool.
    """
    if cache_path.exists():
        with cache_path.open("rb") as f:
            graph = pickle.load(f)
    else:
        graph = gt.load_graph(graphml_path)
        graph.set_directed(False)
        with cache_path.open("wb") as f:
            pickle.dump(graph, f)
    return graph


def ensure_bipartite_property(graph: gt.Graph) -> Tuple[List[int], List[int]]:
    """
    Ensure the 'bipartite' vertex property exists by parity if missing.
    Returns lists of left and right node indices.
    """
    vp = graph.vertex_properties
    if "bipartite" not in vp:
        prop = graph.new_vertex_property("int")
        for v in graph.vertices():
            prop[v] = int(v) % 2
        vp["bipartite"] = prop

    left_nodes = []
    right_nodes = []
    prop = graph.vertex_properties["bipartite"]
    for v in graph.vertices():
        idx = int(v)
        if prop[v] == 0:
            left_nodes.append(idx)
        else:
            right_nodes.append(idx)
    return left_nodes, right_nodes


def graph_to_bipartite_data(graph: gt.Graph) -> Tuple[Data, List[int], List[int]]:
    """
    Convert a raw bipartite graph-tool Graph to PyG Data,
    returning Data and left/right partitions.
    """
    left, right = ensure_bipartite_property(graph)
    prop = graph.vertex_properties["bipartite"]

    num_nodes = graph.num_vertices()
    x = torch.ones((num_nodes, 3), dtype=torch.float)
    for e in graph.edges():
        s = int(e.source())
        t = int(e.target())
        cd = float(graph.ep["contribution_days"][e])
        cm = float(graph.ep["number_commits"][e])
        x[s, 1] += cd
        x[t, 1] += cd
        x[s, 2] += cm
        x[t, 2] += cm

    edges = [(int(e.source()), int(e.target())) for e in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    positives = []
    for e in graph.edges():
        u = int(e.source())
        v = int(e.target())
        if prop[e.source()] != prop[e.target()]:
            if prop[e.source()] == 1:
                u, v = v, u
            positives.append([u, v])

    pos_edge_index = torch.tensor(positives, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    data.pos_edge_label_index = pos_edge_index
    return data, left, right


def sample_bipartite_negatives(
    pos_edge_index: torch.Tensor, left: List[int], right: List[int], num_neg: int
) -> torch.Tensor:
    """
    Sample num_neg negative edges uniformly from left-right pairs,
    excluding positives.
    Returns a [2, num_neg] tensor.
    """
    pos_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    neg = []
    while len(neg) < num_neg:
        u = random.choice(left)
        v = random.choice(right)
        if (u, v) not in pos_set:
            neg.append([u, v])
    neg_index = torch.tensor(neg, dtype=torch.long).t().contiguous()
    return neg_index


class BipartiteEncoder(nn.Module):
    """
    GraphSAGE-based encoder for bipartite graphs.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h


class EdgeDecoder(nn.Module):
    """
    MLP decoder to score bipartite edges from embeddings.
    """

    def __init__(self, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2 * emb_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        h = torch.cat([z[u], z[v]], dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).view(-1)


class BipartiteLinkPredictor(nn.Module):
    """
    Full bipartite link prediction model.
    """

    def __init__(self, in_dim: int, enc_h: int, emb_dim: int, dec_h: int) -> None:
        super().__init__()
        self.encoder = BipartiteEncoder(in_dim, enc_h, emb_dim)
        self.decoder = EdgeDecoder(emb_dim, dec_h)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge: torch.Tensor,
        neg_edge: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        pos_logits = self.decoder(z, pos_edge)
        neg_logits = self.decoder(z, neg_edge)
        return torch.cat([pos_logits, neg_logits], dim=0)


def train_and_save(
    data: Data,
    left: List[int],
    right: List[int],
    epochs: int,
    lr: float,
    model_path: Path,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Train model and save weights. Return history of metrics.
    """
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    pos_train = data.pos_edge_label_index.to(device)
    neg_train = sample_bipartite_negatives(
        pos_train, left, right, pos_train.size(1)
    ).to(device)

    model = BipartiteLinkPredictor(in_dim=x.size(1), enc_h=32, emb_dim=16, dec_h=8).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare test split
    splitter = RandomLinkSplit(
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
        num_val=0.0,
        num_test=TEST_FRAC,
    )
    _, _, test_data = splitter(data)
    pos_test = test_data.pos_edge_label_index.to(device)
    neg_test = sample_bipartite_negatives(pos_test, left, right, pos_test.size(1)).to(
        device
    )

    history: Dict[str, List[float]] = {
        "loss": [],
        "acc": [],
        "prec": [],
        "rec": [],
        "f1": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, ei, pos_train, neg_train)
        labels = torch.cat(
            [torch.ones(pos_train.size(1)), torch.zeros(neg_train.size(1))], dim=0
        ).to(device)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            labs = labels.cpu()

            history["loss"].append(loss.item())
            history["acc"].append(accuracy_score(labs, preds))
            history["prec"].append(precision_score(labs, preds, zero_division=0))
            history["rec"].append(recall_score(labs, preds, zero_division=0))
            history["f1"].append(f1_score(labs, preds, zero_division=0))

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Loss {loss:.4f} | "
                f"F1 {history['f1'][-1]:.4f}"
            )

    torch.save(model.state_dict(), model_path)
    return history


def plot_bipartite_metrics(history: Dict[str, List[float]]) -> None:
    """
    Plot loss and all metrics, saving two figures.
    """
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plotting/bip_train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["acc"], label="Acc")
    plt.plot(epochs, history["prec"], label="Prec")
    plt.plot(epochs, history["rec"], label="Rec")
    plt.plot(epochs, history["f1"], label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("plotting/bip_train_metrics.png")
    plt.close()


def reload_and_evaluate(
    data: Data,
    left: List[int],
    right: List[int],
    model_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    """
    Reload the trained model and evaluate on a fresh test split.
    """
    x = data.x.to(device)
    ei = data.edge_index.to(device)

    splitter = RandomLinkSplit(
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
        num_val=0.0,
        num_test=TEST_FRAC,
    )
    _, _, test_data = splitter(data)
    pos_test = test_data.pos_edge_label_index.to(device)
    neg_test = sample_bipartite_negatives(pos_test, left, right, pos_test.size(1)).to(
        device
    )

    model = BipartiteLinkPredictor(in_dim=x.size(1), enc_h=32, emb_dim=16, dec_h=8).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x, ei, pos_test, neg_test)
        labels = torch.cat(
            [torch.ones(pos_test.size(1)), torch.zeros(neg_test.size(1))], dim=0
        ).cpu()
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()

        return {
            "acc": accuracy_score(labels, preds),
            "prec": precision_score(labels, preds, zero_division=0),
            "rec": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }


def main() -> None:
    graph = load_bipartite_graph(GRAPHML_FILE, CACHE_PATH)
    data, left, right = graph_to_bipartite_data(graph)

    history = train_and_save(
        data, left, right, epochs=EPOCHS, lr=LR, model_path=MODEL_PATH, device=DEVICE
    )

    plot_bipartite_metrics(history)

    metrics = reload_and_evaluate(
        data, left, right, model_path=MODEL_PATH, device=DEVICE
    )
    print(
        f"Final Test Metrics | "
        f"Acc: {metrics['acc']:.4f} | "
        f"Prec: {metrics['prec']:.4f} | "
        f"Rec: {metrics['rec']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
