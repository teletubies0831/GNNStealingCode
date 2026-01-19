"""Shared utilities for dataset loading, evaluation, and graph manipulation.

The helpers here intentionally keep dataset loading and graph splitting logic
in one place so other modules can stay focused on model training/inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


def _unwrap_npz_array(array: np.ndarray):
    """Return the actual object stored in an NPZ array if it was pickled.

    Some NPZ files store a sparse matrix inside a 0-d object array; this helper
    unwraps it so downstream logic can treat it as a normal matrix.
    """
    if isinstance(array, np.ndarray) and array.dtype == object:
        return array.item()
    return array


def _load_csr_from_npz(data: np.lib.npyio.NpzFile, name_candidates: Sequence[str]) -> sp.csr_matrix:
    """Load a CSR matrix from an NPZ file using common key conventions.

    The loader accepts both:
    1) Direct sparse/dense arrays stored under a key.
    2) Triplet keys (<name>_data, <name>_indices, <name>_indptr, <name>_shape).
    """
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
    """Load labels from an NPZ file using common key conventions.

    If labels are one-hot encoded, we convert them to class indices.
    """
    for key in ("labels", "label", "node_label", "y"):
        if key in data:
            labels = _unwrap_npz_array(data[key])
            labels = np.asarray(labels)
            if labels.ndim > 1:
                labels = labels.argmax(axis=1)
            return labels.astype(np.int64)
    raise KeyError("Could not find label keys in NPZ")


def load_npz_graph(dataset: str, root_dir: Path | None = None, add_self_loop: bool = True) -> Tuple[dgl.DGLGraph, int]:
    """Load a graph stored as NPZ into a DGLGraph.

    The loader supports multiple NPZ layouts that are commonly produced by
    GraphGallery, PyG, or custom scripts. It handles both sparse and dense
    adjacency/feature matrices.
    """
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

    # Build a DGL graph from the adjacency matrix and attach node attributes.
    graph = dgl.from_scipy(adjacency)
    graph.ndata["features"] = torch.from_numpy(features)
    graph.ndata["labels"] = torch.from_numpy(labels)

    if add_self_loop:
        # GNN baselines in this repo expect self-loops to stabilize training.
        graph = dgl.add_self_loop(graph)

    num_classes = int(np.unique(labels).shape[0])
    return graph, num_classes


def compute_fidelity(pred_surrogate: torch.Tensor, pred_target: torch.Tensor) -> float:
    """Return the agreement ratio between surrogate and target predictions.

    Fidelity is defined as the fraction of nodes where the surrogate and target
    predict the same class.
    """
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
    """Project embeddings into 2D via TSNE or PCA, optionally plotting them.

    The function returns the 2D embedding coordinates. Visualization is optional
    so this can be used in headless environments (e.g., batch training).
    """
    if transform_name.upper() == "TSNE":
        transformer = TSNE(n_components=2, n_iter=3000, n_jobs=-1)
    elif transform_name.upper() == "PCA":
        transformer = PCA(n_components=2)
    else:
        raise ValueError("transform_name should be TSNE or PCA")

    projected = transformer.fit_transform(embeddings)

    if show_figure:
        # Only import plotting libraries when a figure is requested to avoid
        # unnecessary dependencies during headless runs.
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
    graph: dgl.DGLGraph,
    frac_list: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """Split a graph into train/val/test subgraphs by node indices.

    We permute nodes deterministically with a seed to keep experiments repeatable.
    """
    num_nodes = graph.number_of_nodes()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_end = int(num_nodes * frac_list[0])
    val_end = train_end + int(num_nodes * frac_list[1])

    # Slice node indices into train/val/test splits.
    train_nodes = perm[:train_end]
    val_nodes = perm[train_end:val_end]
    test_nodes = perm[val_end:]

    train_g = graph.subgraph(train_nodes)
    val_g = graph.subgraph(val_nodes)
    test_g = graph.subgraph(test_nodes)

    _ensure_features_and_labels(train_g)
    _ensure_features_and_labels(val_g)
    _ensure_features_and_labels(test_g)

    return train_g, val_g, test_g


def split_graph_different_ratio(
    graph: dgl.DGLGraph,
    frac_list: Sequence[float] = (0.6, 0.2, 0.2),
    ratio: float = 0.5,
    seed: int = 0,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """Split graph and shrink the training subset by a ratio.

    This supports attack settings where the query graph is only a fraction of
    the full training split.
    """
    train_g, val_g, test_g = split_graph(graph, frac_list=frac_list, seed=seed)
    train_nodes = train_g.nodes()[: int(train_g.number_of_nodes() * ratio)]
    train_g = graph.subgraph(train_nodes)

    _ensure_features_and_labels(train_g)
    _ensure_features_and_labels(val_g)
    _ensure_features_and_labels(test_g)

    return train_g, val_g, test_g


def delete_dgl_graph_edge(train_g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Return a graph with only self-loops while preserving node features/labels."""
    empty_graph = dgl.graph(([], []), num_nodes=train_g.number_of_nodes())
    empty_graph = dgl.add_self_loop(empty_graph)
    empty_graph.ndata["features"] = train_g.ndata["features"]
    empty_graph.ndata["labels"] = train_g.ndata["labels"]
    return empty_graph


def train_detached_classifier(train_g: dgl.DGLGraph, embeddings: torch.Tensor) -> MLPClassifier:
    """Train an sklearn MLP on frozen embeddings.

    The detached classifier emulates the evaluation in the original paper,
    where the surrogate embeddings are fed into a separate MLP head.
    """
    features = embeddings.detach().cpu().numpy()
    labels = train_g.ndata["labels"].cpu().numpy()

    # Use stratified split so all classes appear in train/test sets.
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        stratify=labels,
        random_state=1,
    )

    classifier = MLPClassifier(random_state=1, max_iter=300).fit(train_x, train_y)
    classifier.predict_proba(test_x[:1])
    classifier.predict(test_x[:5, :])

    # Keep the score printed for parity with previous logs.
    print(classifier.score(test_x, test_y))
    return classifier


def _ensure_features_and_labels(graph: dgl.DGLGraph) -> None:
    """Make sure features/labels live under the expected keys."""
    if "features" not in graph.ndata and "feat" in graph.ndata:
        graph.ndata["features"] = graph.ndata["feat"]
    if "labels" not in graph.ndata and "label" in graph.ndata:
        graph.ndata["labels"] = graph.ndata["label"]
