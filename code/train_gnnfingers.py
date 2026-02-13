"""Train GNNFingers fingerprints and univerifier end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from gnnfingers.fingerprints import GraphFingerprintSet
from gnnfingers.joint_train import JointTrainConfig, train_gnnfingers
from gnnfingers.obfuscation import EnsembleConfig, prepare_model_ensemble
from gnnfingers.univerifier import UniverifierMLP
from gnnfingers.utils import (
    compute_thresholds,
    evaluate_metrics,
    flatten_outputs,
    save_config,
    select_best_threshold,
)
from src.gat import GAT
from src.gin import GIN
from src.sage import SAGE
from src.utils import load_npz_graph, split_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train GNNFingers")
    parser.add_argument("--task", type=str, default="node_classification")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--target-model", type=str, default="gat")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-fingerprints", type=int, default=32)
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--feat-dim", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--feature-lr", type=float, default=0.1)
    parser.add_argument("--adj-lr", type=float, default=1.0)
    parser.add_argument("--num-positive", type=int, default=40)
    parser.add_argument("--num-negative", type=int, default=40)
    parser.add_argument("--negative-datasets", type=str, default="")
    parser.add_argument("--negative-architectures", type=str, default="gat,gin,sage")
    parser.add_argument("--negative-per-dataset", type=int, default=10)
    parser.add_argument("--negative-epochs", type=int, default=20)
    parser.add_argument("--target-epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./gnnfingers_out")
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _build_model(model_name: str, in_feats: int, hidden_dim: int, num_classes: int, num_layers: int, heads: int, dropout: float):
    if model_name == "gat":
        return GAT(in_feats, hidden_dim, num_classes, num_layers, heads, dropout)
    if model_name == "gin":
        return GIN(in_feats, hidden_dim, num_classes, num_layers, dropout)
    if model_name == "sage":
        return SAGE(in_feats, hidden_dim, num_classes, num_layers, dropout)
    raise ValueError("model_name must be gat, gin, or sage")


def _train_target(model, data, train_idx, num_epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()


def _parse_csv(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _align_dataset(data, feat_dim: int, num_classes: int):
    features = data.x
    if features.size(1) < feat_dim:
        pad_cols = feat_dim - features.size(1)
        padding = torch.zeros((features.size(0), pad_cols), dtype=features.dtype)
        features = torch.cat([features, padding], dim=1)
    elif features.size(1) > feat_dim:
        features = features[:, :feat_dim]
    labels = data.y % num_classes
    return type(data)(x=features, edge_index=data.edge_index, y=labels)


def _train_negative_models_on_dataset(
    dataset_name: str,
    architectures: List[str],
    base_cfg: dict,
    feat_dim: int,
    num_classes: int,
    num_models: int,
    num_epochs: int,
    device: torch.device,
):
    data, _ = load_npz_graph(dataset_name)
    data = _align_dataset(data, feat_dim, num_classes)
    train_idx, _, _ = split_graph(data, frac_list=[0.6, 0.2, 0.2])
    data = data.to(device)
    models = []
    for idx in range(num_models):
        arch = architectures[idx % len(architectures)]
        model = _build_model(
            arch,
            feat_dim,
            base_cfg["hidden_dim"],
            num_classes,
            base_cfg["num_layers"],
            base_cfg["heads"],
            base_cfg["dropout"],
        ).to(device)
        _train_target(model, data, train_idx, num_epochs=num_epochs)
        models.append(model)
    return models


def _score_models(
    univerifier: UniverifierMLP,
    models: List,
    fingerprints: GraphFingerprintSet,
    task: str,
    device: torch.device,
) -> List[float]:
    graphs = fingerprints.build_graphs()
    meta_list = fingerprints.get_query_meta()
    scores = []
    univerifier.eval()
    for model in models:
        model = model.to(device)
        outputs = []
        for graph in graphs:
            graph = graph.to(device)
            with torch.no_grad():
                logits, embeddings = model(graph.x, graph.edge_index)
            outputs.append({"logits": logits, "embeddings": embeddings})
        vectors = [
            flatten_outputs(outputs[i], task, meta_list[i]) for i in range(len(meta_list))
        ]
        batch = torch.stack(vectors).to(device)
        probs = univerifier(batch)
        scores.append(probs[:, 0].mean().item())
    return scores


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)

    data, num_classes = load_npz_graph(args.dataset)
    train_idx, val_idx, test_idx = split_graph(data, frac_list=[0.6, 0.2, 0.2])

    target_model = _build_model(
        args.target_model,
        data.num_features,
        args.hidden_dim,
        num_classes,
        args.num_layers,
        args.heads,
        args.dropout,
    ).to(device)
    _train_target(target_model, data.to(device), train_idx, num_epochs=args.target_epochs)

    feat_dim = data.num_features if args.feat_dim <= 0 else args.feat_dim
    fingerprint_set = GraphFingerprintSet(
        num_fingerprints=args.num_fingerprints,
        num_nodes=args.num_nodes,
        feat_dim=feat_dim,
        epsilon=args.epsilon,
        num_queries=args.num_queries,
        task_type=args.task,
        device=device,
    )

    negative_architectures = _parse_csv(args.negative_architectures) or [args.target_model]
    ensemble_cfg = EnsembleConfig(
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        architectures=tuple(negative_architectures),
    )
    base_cfg = {
        "hidden_dim": args.hidden_dim,
        "num_classes": num_classes,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    pos_models, neg_models = prepare_model_ensemble(target_model, data.to(device), train_idx, ensemble_cfg, base_cfg)
    negative_datasets = _parse_csv(args.negative_datasets)
    if negative_datasets:
        for dataset_name in negative_datasets:
            neg_models.extend(
                _train_negative_models_on_dataset(
                    dataset_name,
                    negative_architectures,
                    base_cfg,
                    feat_dim,
                    num_classes,
                    args.negative_per_dataset,
                    args.negative_epochs,
                    device,
                )
            )
    pos_train = pos_models[: len(pos_models) // 2]
    pos_test = pos_models[len(pos_models) // 2 :]
    neg_train = neg_models[: len(neg_models) // 2]
    neg_test = neg_models[len(neg_models) // 2 :]

    train_cfg = JointTrainConfig(
        iterations=args.iterations,
        lr=args.lr,
        top_k=args.top_k,
        feature_lr=args.feature_lr,
        adj_lr=args.adj_lr,
        task_type=args.task,
    )
    result = train_gnnfingers(target_model, pos_train, neg_train, fingerprint_set, train_cfg, device)
    univerifier = result["univerifier"]

    scores_pos = _score_models(univerifier, pos_test, fingerprint_set, args.task, device)
    scores_neg = _score_models(univerifier, neg_test, fingerprint_set, args.task, device)
    lambdas, _ = compute_thresholds(scores_pos + scores_neg)
    metrics = evaluate_metrics(scores_pos, scores_neg, lambdas)
    best_lambda = select_best_threshold(scores_pos, scores_neg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path = output_dir / "fingerprints.pt"
    univerifier_path = output_dir / "univerifier.pt"
    fingerprint_set.save(str(fingerprint_path))
    torch.save(univerifier.state_dict(), univerifier_path)

    save_config(
        output_dir / "config.yaml",
        {
            "task": args.task,
            "dataset": args.dataset,
            "target_model": args.target_model,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "num_fingerprints": args.num_fingerprints,
            "num_nodes": args.num_nodes,
            "feat_dim": feat_dim,
            "epsilon": args.epsilon,
            "num_queries": args.num_queries,
            "iterations": args.iterations,
            "lr": args.lr,
            "top_k": args.top_k,
            "feature_lr": args.feature_lr,
            "adj_lr": args.adj_lr,
            "num_positive": args.num_positive,
            "num_negative": args.num_negative,
            "negative_datasets": negative_datasets,
            "negative_architectures": negative_architectures,
            "negative_per_dataset": args.negative_per_dataset,
            "negative_epochs": args.negative_epochs,
            "target_epochs": args.target_epochs,
            "univerifier_input_dim": univerifier.net[0].in_features,
            "metrics": metrics,
            "lambda_threshold": best_lambda,
        },
    )

    print(f"[GNNFingers] Saved fingerprints to {fingerprint_path}")
    print(f"[GNNFingers] Saved univerifier to {univerifier_path}")
    print(f"[GNNFingers] Metrics: {metrics}")
    print(f"[GNNFingers] Suggested lambda_threshold: {best_lambda:.4f}")


if __name__ == "__main__":
    main()
