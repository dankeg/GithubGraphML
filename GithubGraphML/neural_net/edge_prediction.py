import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv

# Configuration constants
CACHE_PATH = Path("combined.pkl")
MODEL_PATH = Path("edge_gnn_model.pth")
TEST_FRAC = 0.2
EPOCHS = 50
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_graph(languages: List[str], cache_path: Path) -> Any:
    """
    Load or combine language graphs via graph-tool, returning an undirected graph.
    """
    if cache_path.exists():
        with cache_path.open("rb") as f:
            g = pickle.load(f)
    else:
        from GithubGraphML.analyze import load_networks
        from GithubGraphML.parsing.loading import combine_graphs

        graphs = load_networks(languages, vprop_name="id")
        g, _ = combine_graphs(graphs, vprop_name="id")
        with cache_path.open("wb") as f:
            pickle.dump(g, f)
    g.set_directed(False)
    return g


def graph_to_data(g: Any) -> Data:
    """
    Convert a graph-tool Graph to a PyG Data object with node features and edge_index.
    """
    num_nodes = g.num_vertices()
    x = torch.ones((num_nodes, 3), dtype=torch.float)
    for e in g.edges():
        s, t = int(e.source()), int(e.target())
        cd = float(g.ep["contribution_days"][e])
        cm = float(g.ep["number_commits"][e])
        x[s, 1] += cd
        x[t, 1] += cd
        x[s, 2] += cm
        x[t, 2] += cm
    edges = [(int(e.source()), int(e.target())) for e in g.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        return self.conv2(h, edge_index)


class EdgeDecoder(nn.Module):
    def __init__(self, emb: int, hidden: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2 * emb, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        h = torch.cat([z[u], z[v]], dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).view(-1)


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, enc_h: int, emb: int, dec_h: int) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_dim, enc_h, emb)
        self.decoder = EdgeDecoder(emb, dec_h)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        p = self.decoder(z, pos_edges)
        n = self.decoder(z, neg_edges)
        return torch.cat([p, n], dim=0)


def train_loop(
    train_data: Data, epochs: int, lr: float, device: torch.device
) -> Tuple[Dict[str, List[float]], nn.Module]:
    """
    Train on train_data only, returns history and trained model.
    """
    # Extract supervision pos/neg from PyG split
    train_pos = train_data.pos_edge_label_index.to(device)
    train_neg = train_data.neg_edge_label_index.to(device)
    x = train_data.x.to(device)
    ei = train_data.edge_index.to(device)

    model = LinkPredictor(in_dim=x.size(1), enc_h=64, emb=32, dec_h=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, ei, train_pos, train_neg)
        labels = torch.cat(
            [torch.ones(train_pos.size(1)), torch.zeros(train_neg.size(1))], dim=0
        ).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            labs = labels.cpu().numpy()
            history["loss"].append(loss.item())
            history["acc"].append(accuracy_score(labs, preds))
            history["prec"].append(precision_score(labs, preds, zero_division=0))
            history["rec"].append(recall_score(labs, preds, zero_division=0))
            history["f1"].append(f1_score(labs, preds, zero_division=0))
        if ep % 10 == 0:
            print(
                f"Epoch {ep}/{epochs} | Loss {loss:.4f} | Train F1 {history['f1'][-1]:.4f}"
            )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return history, model


def evaluate(
    test_data: Data, model: nn.Module, device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the saved model on test_data only.
    """
    model.eval()
    pos_test = test_data.pos_edge_label_index.to(device)
    neg_test = test_data.neg_edge_label_index.to(device)
    x = test_data.x.to(device)
    ei = test_data.edge_index.to(device)
    with torch.no_grad():
        logits = model(x, ei, pos_test, neg_test)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        labs = (
            torch.cat(
                [torch.ones(pos_test.size(1)), torch.zeros(neg_test.size(1))], dim=0
            )
            .cpu()
            .numpy()
        )
    return {
        "acc": accuracy_score(labs, preds),
        "prec": precision_score(labs, preds, zero_division=0),
        "rec": recall_score(labs, preds, zero_division=0),
        "f1": f1_score(labs, preds, zero_division=0),
    }


def plot_train_metrics(history: Dict[str, List[float]]) -> None:
    """
    Plot training loss & metrics across epochs.
    """
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plotting/train_loss_edge.png")
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["acc"], label="Acc")
    plt.plot(epochs, history["prec"], label="Prec")
    plt.plot(epochs, history["rec"], label="Rec")
    plt.plot(epochs, history["f1"], label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("plotting/train_metrics_edge.png")
    plt.close()


def main() -> None:
    langs = ["Assembly", "Javascript", "Pascal", "Perl", "Python", "VisualBasic"]
    g = load_graph(langs, CACHE_PATH)
    data = graph_to_data(g)

    splitter = RandomLinkSplit(
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        num_val=0.0,
        num_test=TEST_FRAC,
    )
    train_data, _, test_data = splitter(data)
    history, model = train_loop(train_data, EPOCHS, LR, DEVICE)
    plot_train_metrics(history)

    # reload and test
    model2 = LinkPredictor(in_dim=data.x.size(1), enc_h=64, emb=32, dec_h=16).to(DEVICE)
    model2.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    metrics = evaluate(test_data, model2, DEVICE)
    print(
        f"Test Metrics | Acc: {metrics['acc']:.4f} | Prec: {metrics['prec']:.4f} | Rec: {metrics['rec']:.4f} | F1: {metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
