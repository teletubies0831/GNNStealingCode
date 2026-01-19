"""Run model-stealing attacks against inductive GNNs.

High-level flow:
1) Load a target model trained on a dataset.
2) Query the target on a subset of nodes (query graph).
3) Train a surrogate model to imitate target outputs/embeddings.
4) Evaluate fidelity and save results.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy import sparse

from core.model_handler import ModelHandler
from src.constants import config
from src.gat import evaluate_gat_target
from src.gatsurrogate import evaluate_gat_surrogate, run_gat_surrogate
from src.gin import GIN, evaluate_gin_target
from src.ginsurrogate import evaluate_gin_surrogate, run_gin_surrogate
from src.sage import evaluate_sage_target
from src.sagesurrogate import evaluate_sage_surrogate, run_sage_surrogate
from src.utils import (
    compute_fidelity,
    delete_dgl_graph_edge,
    load_npz_graph,
    projection,
    split_graph_different_ratio,
)

torch.set_num_threads(1)

torch.manual_seed(0)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the attack pipeline."""
    parser = argparse.ArgumentParser("Model stealing for inductive GNNs")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device ID, -1 for CPU")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--transform", type=str, default="TSNE")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--fan-out", type=str, default="10,50")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--target-model", type=str, default="sage")
    parser.add_argument("--target-model-dim", type=int, default=256)
    parser.add_argument("--surrogate-model", type=str, default="sage")
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--recovery-from", type=str, default="embedding")
    parser.add_argument("--round_index", type=int, default=1)
    parser.add_argument("--query_ratio", type=float, default=1.0)
    parser.add_argument("--structure", type=str, default="original")
    parser.add_argument("--delete_edges", type=str, default="no")
    args, _ = parser.parse_known_args()
    args.inductive = True
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    """Return a CUDA or CPU device based on the requested GPU id."""
    return torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")


def _load_target_model(
    args: argparse.Namespace,
    model_args: dict,
    graph,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load a target model checkpoint (GIN uses state dict, others saved directly)."""
    model_dir = Path(f"./target_model_{args.target_model}_{args.target_model_dim}")
    model_path = model_dir / f"target_model_{args.target_model}_{args.dataset}"

    if args.target_model == "gin":
        model = GIN(
            graph.ndata["features"].shape[1],
            model_args["num_hidden"],
            n_classes,
            model_args["num_layers"],
            torch.relu,
            model_args["batch_size"],
            model_args["num_workers"],
            model_args["dropout"],
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        model = torch.load(model_path, map_location="cpu")

    return model.to(device)


def _query_target_model(args, target_model, query_graph, model_args, device):
    """Run inference on the target model to get predictions and embeddings."""
    if args.target_model == "sage":
        return evaluate_sage_target(
            target_model,
            query_graph,
            query_graph.ndata["features"],
            query_graph.ndata["labels"],
            query_graph.nodes(),
            args.batch_size,
            device,
        )
    if args.target_model == "gin":
        return evaluate_gin_target(
            target_model,
            query_graph,
            query_graph.ndata["features"],
            query_graph.ndata["labels"],
            query_graph.nodes(),
            args.batch_size,
            device,
        )
    if args.target_model == "gat":
        return evaluate_gat_target(
            target_model,
            query_graph,
            query_graph.ndata["features"],
            query_graph.ndata["labels"],
            query_graph.nodes(),
            model_args["val_batch_size"],
            model_args["head"],
            device,
        )
    raise ValueError("target-model must be gat, gin, or sage")


def _prepare_query_graph(args, train_g) -> torch.Tensor:
    """Return the graph used to query the target model.

    The graph can either be the original subgraph or an IDGL-reconstructed one.
    """
    if args.structure == "original":
        query_g = train_g
        if args.delete_edges == "yes":
            query_g = delete_dgl_graph_edge(train_g)
        return query_g

    if args.structure == "idgl":
        config["dgl_graph"] = train_g
        config["cuda_id"] = args.gpu
        # Run IDGL to reconstruct the adjacency matrix used for querying.
        model = ModelHandler(config)
        model.train()
        _, adjacency = model.test()

        adjacency = adjacency.detach().cpu().numpy()
        if args.dataset in {"acm", "amazon_cs"}:
            adjacency = (adjacency > 0.9).astype(np.int32)
        elif args.dataset in {"coauthor_phy"}:
            adjacency = (adjacency >= 0.999).astype(np.int32)
        else:
            adjacency = (adjacency > 0.999).astype(np.int32)

        sparse_adj = sparse.csr_matrix(adjacency)
        query_g = dgl.from_scipy(sparse_adj)
        query_g.ndata["features"] = train_g.ndata["features"]
        query_g.ndata["labels"] = train_g.ndata["labels"]
        return dgl.add_self_loop(query_g)

    raise ValueError("structure must be original or idgl")


def _prepare_surrogate_data(args, train_g, val_g, test_g, query_preds, query_embs):
    """Prepare surrogate training inputs from the query responses."""
    if args.recovery_from == "prediction":
        response = query_preds
    elif args.recovery_from == "embedding":
        response = query_embs
    elif args.recovery_from == "projection":
        projected = projection(
            query_embs.detach().cpu().numpy(),
            train_g.ndata["labels"].cpu().numpy(),
            transform_name=args.transform,
            show_figure=False,
            gnn=args.target_model,
            dataset=args.dataset,
        )
        response = torch.from_numpy(projected).float().to(query_embs.device)
    else:
        raise ValueError("recovery-from must be prediction, embedding, or projection")

    # Return a tuple layout expected by the surrogate training helpers.
    return (
        train_g.ndata["features"].shape[1],
        query_preds.shape[1],
        train_g,
        val_g,
        test_g,
        response,
    )


def _train_surrogate(args, device, data):
    """Train the surrogate model and return embeddings + detached classifier."""
    if args.surrogate_model == "gin":
        model_s, classifier, detached_classifier = run_gin_surrogate(args, device, data, "./surrogate_model")
        metrics = evaluate_gin_surrogate(
            model_s,
            classifier,
            data[4],
            data[4].ndata["features"],
            data[4].ndata["labels"],
            data[4].nodes(),
            args.batch_size,
            device,
        )
    elif args.surrogate_model == "gat":
        model_s, classifier, detached_classifier = run_gat_surrogate(args, device, data, "./surrogate_model")
        metrics = evaluate_gat_surrogate(
            model_s,
            classifier,
            data[4],
            data[4].ndata["features"],
            data[4].ndata["labels"],
            data[4].nodes(),
            args.batch_size,
            args.head,
            device,
        )
    elif args.surrogate_model == "sage":
        model_s, classifier, detached_classifier = run_sage_surrogate(args, device, data, "./surrogate_model")
        metrics = evaluate_sage_surrogate(
            model_s,
            classifier,
            data[4],
            data[4].ndata["features"],
            data[4].ndata["labels"],
            data[4].nodes(),
            args.batch_size,
            device,
        )
    else:
        raise ValueError("surrogate-model must be gat, gin, or sage")

    _, _, embds_surrogate = metrics
    return embds_surrogate, detached_classifier


def _save_results(args, test_acc, surrogate_acc, fidelity):
    """Append a single experiment line to the results file."""
    output_folder = Path("./results_acc_fidelity") / f"results_{args.target_model}_{args.target_model_dim}_{args.surrogate_model}_{args.num_hidden}"
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.structure == "original":
        filename = output_folder / f"{args.dataset}_original.txt"
    elif args.structure == "idgl":
        filename = output_folder / f"{args.dataset}_idgl.txt"
    else:
        raise ValueError("structure must be original or idgl")

    with filename.open("a") as handle:
        handle.write(
            f"{args.target_model},{args.target_model_dim},{args.surrogate_model},{args.num_hidden},"
            f"{args.recovery_from},{args.round_index},{args.query_ratio},"
            f"{test_acc},{surrogate_acc},{fidelity}\n"
        )


def main() -> None:
    """Entry point for the attack workflow."""
    args = _parse_args()
    device = _resolve_device(args.gpu)

    # Load dataset and target model.
    graph, n_classes = load_npz_graph(args.dataset)
    model_args = pickle.load(
        open(f"./target_model_{args.target_model}_{args.target_model_dim}/model_args", "rb")
    ).__dict__
    model_args["gpu"] = args.gpu

    target_model = _load_target_model(args, model_args, graph, n_classes, device)

    # Build the query graph split based on the requested ratio.
    train_g, val_g, test_g = split_graph_different_ratio(
        graph, frac_list=[0.3, 0.2, 0.5], ratio=args.query_ratio
    )
    query_g = _prepare_query_graph(args, train_g)

    # Query the target model for predictions and embeddings.
    _, query_preds, query_embs = _query_target_model(args, target_model, query_g, model_args, device)
    query_preds = query_preds.to(device)
    query_embs = query_embs.to(device)

    if args.structure != "original":
        train_g = query_g

    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # Train surrogate based on the chosen recovery signal.
    data = _prepare_surrogate_data(args, train_g, val_g, test_g, query_preds, query_embs)
    surrogate_embs, detached_classifier = _train_surrogate(args, device, data)

    # Evaluate surrogate embeddings with the detached classifier.
    detached_acc = detached_classifier.score(
        surrogate_embs.detach().cpu().numpy(), test_g.ndata["labels"].cpu().numpy()
    )
    detached_preds = detached_classifier.predict_proba(surrogate_embs.detach().cpu().numpy())

    if args.target_model == "sage":
        test_acc, target_preds, _ = evaluate_sage_target(
            target_model,
            test_g,
            test_g.ndata["features"],
            test_g.ndata["labels"],
            test_g.nodes(),
            args.batch_size,
            device,
        )
    elif args.target_model == "gat":
        test_acc, target_preds, _ = evaluate_gat_target(
            target_model,
            test_g,
            test_g.ndata["features"],
            test_g.ndata["labels"],
            test_g.nodes(),
            model_args["val_batch_size"],
            model_args["head"],
            device,
        )
    elif args.target_model == "gin":
        test_acc, target_preds, _ = evaluate_gin_target(
            target_model,
            test_g,
            test_g.ndata["features"],
            test_g.ndata["labels"],
            test_g.nodes(),
            args.batch_size,
            device,
        )
    else:
        raise ValueError("target-model must be gat, gin, or sage")

    # Fidelity compares surrogate predictions with target predictions.
    fidelity = compute_fidelity(torch.from_numpy(detached_preds).to(device), target_preds.to(device))
    _save_results(args, test_acc, detached_acc, fidelity)


if __name__ == "__main__":
    main()
