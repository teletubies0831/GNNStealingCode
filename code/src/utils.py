"""Shared utilities for dataset loading, evaluation, and graph manipulation (PyG)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch_geometric.data import Data

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"


class Classification(torch.nn.Module):
    """Small MLP used by detached classifiers."""

    def __init__(self, emb_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(emb_size, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


@dataclass
class GraphStats:
    """Summary statistics for a learned adjacency."""

    avg_degree: float
    nnz: int


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """Row-normalize features to unit norm for cosine similarity."""
    return F.normalize(x, p=2.0, dim=1)


def build_knn_graph(
    x: torch.Tensor,
    k: int,
    self_loop: bool = True,
    symmetric: bool = True,
) -> torch.Tensor:
    """Build a kNN adjacency (sparse edge index) using cosine similarity."""
    x_norm = normalize_features(x)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1.0)
    _, idx = torch.topk(sim, k=k, dim=1)
    row = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
    col = idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    if self_loop:
        self_loop_idx = torch.arange(x.size(0), device=x.device)
        self_edges = torch.stack([self_loop_idx, self_loop_idx], dim=0)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
    if symmetric:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def sparse_from_similarity(
    sim: torch.Tensor,
    top_k: int,
    self_loop: bool = True,
    symmetric: bool = True,
) -> torch.Tensor:
    """Sparsify a similarity matrix into an edge index via top-k."""
    sim = sim.clone()
    sim.fill_diagonal_(-1.0)
    _, idx = torch.topk(sim, k=top_k, dim=1)
    row = torch.arange(sim.size(0), device=sim.device).repeat_interleave(top_k)
    col = idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    if self_loop:
        diag = torch.arange(sim.size(0), device=sim.device)
        edge_index = torch.cat([edge_index, torch.stack([diag, diag], dim=0)], dim=1)
    if symmetric:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def adjacency_stats(edge_index: torch.Tensor, num_nodes: int) -> GraphStats:
    """Compute average degree and nnz for logging."""
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
    return GraphStats(avg_degree=deg.mean().item(), nnz=edge_index.size(1))


def graph_smoothness_loss(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Smoothness penalty: sum_{i,j} A_ij ||h_i - h_j||^2."""
    src, dst = edge_index
    diff = h[src] - h[dst]
    return (diff.pow(2).sum(dim=1)).mean()


def sparse_degree_loss(edge_index: torch.Tensor, num_nodes: int, target_degree: float) -> torch.Tensor:
    """Penalty to avoid degenerate graphs by matching degree statistics."""
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
    return (deg.mean() - target_degree).abs()


def relative_change(a_prev: torch.Tensor, a_next: torch.Tensor) -> float:
    """Relative Frobenius change between two dense matrices."""
    delta = (a_next - a_prev).pow(2).sum().sqrt()
    denom = a_prev.pow(2).sum().sqrt().clamp_min(1e-8)
    return (delta / denom).item()


def similarity_matrix(h: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix from node embeddings."""
    h_norm = normalize_features(h)
    return h_norm @ h_norm.t()


def _unwrap_npz_array(array: np.ndarray):
    """Return the actual object stored in an NPZ array if it was pickled."""
    if isinstance(array, np.ndarray) and array.dtype == object:
        return array.item()
    return array


def _load_csr_from_npz(data: np.lib.npyio.NpzFile, name_candidates: Sequence[str]) -> sp.csr_matrix:
    """Load a CSR matrix from an NPZ file using common key conventions."""
    for name in name_candidates:
        if name in data:
            matrix = _unwrap_npz_array(data[name])
            if sp.issparse(matrix):
                return matrix.tocsr()
            return sp.csr_matrix(matrix)

        data_key = f"{name}_data"
        indices_key = f"{name}_indices"
        indptr_key = f"{name}_indptr"
        shape_key = f"{name}_shape"
        if data_key in data and indices_key in data and indptr_key in data and shape_key in data:
            return sp.csr_matrix(
                (data[data_key], data[indices_key], data[indptr_key]),
                shape=tuple(data[shape_key]),
            )

    raise KeyError(f"Could not find CSR matrix keys in NPZ: {name_candidates}")


def _load_labels_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Load labels from an NPZ file using common key conventions."""
    for key in ("labels", "label", "node_label", "y"):
        if key in data:
            labels = _unwrap_npz_array(data[key])
            labels = np.asarray(labels)
            if labels.ndim > 1:
                labels = labels.argmax(axis=1)
            return labels.astype(np.int64)
    raise KeyError("Could not find label keys in NPZ")


def load_npz_graph(dataset: str, root_dir: Path | None = None, add_self_loop: bool = True) -> Tuple[Data, int]:
    """Load a graph stored as NPZ into a PyG Data object."""
    root_dir = root_dir or DATASET_DIR
    path = Path(root_dir) / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        adjacency = _load_csr_from_npz(data, ("adj", "adj_matrix", "adjacency"))
        features = _load_csr_from_npz(data, ("attr", "features", "node_attr", "x"))
        labels = _load_labels_from_npz(data)

    if sp.issparse(features):
        features = features.toarray()
    features = np.asarray(features, dtype=np.float32)

    adjacency = adjacency.tocoo()
    edge_index = torch.from_numpy(np.vstack((adjacency.row, adjacency.col))).long()

    if add_self_loop:
        num_nodes = features.shape[0]
        self_loops = torch.arange(num_nodes, dtype=torch.long)
        self_edge_index = torch.stack([self_loops, self_loops])
        edge_index = torch.cat([edge_index, self_edge_index], dim=1)

    data_obj = Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        y=torch.from_numpy(labels),
    )

    num_classes = int(np.unique(labels).shape[0])
    return data_obj, num_classes


def compute_fidelity(pred_surrogate: torch.Tensor, pred_target: torch.Tensor) -> float:
    """Return the agreement ratio between surrogate and target predictions."""
    surrogate = torch.argmax(pred_surrogate, dim=1)
    target = torch.argmax(pred_target, dim=1)
    return (surrogate == target).float().mean().item()


def compute_acc(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute accuracy of model predictions."""
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().mean()


def projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    transform_name: str = "TSNE",
    show_figure: bool = True,
    gnn: str = "Graphsage",
    dataset: str = "Cora",
) -> np.ndarray:
    """Project embeddings into 2D via TSNE or PCA, optionally plotting them."""
    if transform_name.upper() == "TSNE":
        transformer = TSNE(n_components=2, n_iter=3000, n_jobs=-1)
    elif transform_name.upper() == "PCA":
        transformer = PCA(n_components=2)
    else:
        raise ValueError("transform_name should be TSNE or PCA")

    projected = transformer.fit_transform(embeddings)

    if show_figure:
        import pandas as pd
        import matplotlib.pyplot as plt

        frame = pd.DataFrame(projected, columns=["x", "y"])
        frame["label"] = labels
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(frame["x"], frame["y"], c=frame["label"].astype("category"), cmap="jet", alpha=0.7)
        ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
        plt.axis("off")
        plt.title(f"{transformer.__class__.__name__} visualization of {gnn} embeddings for {dataset}")
        plt.show()

    return projected


def split_graph(
    data_obj: Data,
    frac_list: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split nodes into train/val/test indices."""
    num_nodes = data_obj.num_nodes
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_end = int(num_nodes * frac_list[0])
    val_end = train_end + int(num_nodes * frac_list[1])

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    return train_idx, val_idx, test_idx


def split_graph_different_ratio(
    data_obj: Data,
    frac_list: Sequence[float] = (0.6, 0.2, 0.2),
    ratio: float = 0.5,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split nodes and shrink the training subset by a ratio."""
    train_idx, val_idx, test_idx = split_graph(data_obj, frac_list=frac_list, seed=seed)
    reduced_train = train_idx[: int(len(train_idx) * ratio)]
    return reduced_train, val_idx, test_idx


def train_detached_classifier(train_labels: torch.Tensor, embeddings: torch.Tensor) -> MLPClassifier:
    """Train an sklearn MLP on frozen embeddings."""
    features = embeddings.detach().cpu().numpy()
    labels = train_labels.detach().cpu().numpy()

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        stratify=labels,
        random_state=1,
    )

    classifier = MLPClassifier(random_state=1, max_iter=300).fit(train_x, train_y)
    classifier.predict_proba(test_x[:1])
    classifier.predict(test_x[:5, :])

    print(classifier.score(test_x, test_y))
    return classifier
