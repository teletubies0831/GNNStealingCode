"""Run model-stealing attacks against inductive GNNs (PyG)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from src.gat import GAT, TrainConfig as GATConfig, evaluate_gat
from src.gin import GIN, TrainConfig as GINConfig, evaluate_gin
from src.sage import SAGE, TrainConfig as SAGEConfig, evaluate_sage
from src.utils import (
    compute_fidelity,
    load_npz_graph,
    projection,
    split_graph_different_ratio,
    train_detached_classifier,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Model stealing for inductive GNNs (PyG)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID, -1 for CPU")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--head", type=int, default=2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--target-model", type=str, default="sage")
    parser.add_argument("--target-model-dim", type=int, default=128)
    parser.add_argument("--surrogate-model", type=str, default="sage")
    parser.add_argument("--num-hidden", type=int, default=128)
    parser.add_argument("--recovery-from", type=str, default="embedding")
    parser.add_argument("--round_index", type=int, default=1)
    parser.add_argument("--query_ratio", type=float, default=0.5)
    parser.add_argument("--structure", type=str, default="original")
    parser.add_argument("--save-surrogate", action="store_true", help="Save surrogate checkpoint after training")
    parser.add_argument("--surrogate-save-dir", type=str, default="./surrogate_models")
    parser.add_argument("--transform", type=str, default="TSNE")
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
    raise ValueError("model name must be gat, gin, or sage")


def _load_target_model(args, data, n_classes, device):
    model_dir = Path(f"./target_model_{args.target_model}_{args.target_model_dim}")
    model_path = model_dir / f"target_model_{args.target_model}_{args.dataset}"

    model = _build_model(
        args.target_model,
        data.num_features,
        args.target_model_dim,
        n_classes,
        args.num_layers,
        args.head,
        args.dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.to(device)


def _train_surrogate(args, data, train_idx, query_logits, query_embeddings, device):
    model = _build_model(
        args.surrogate_model,
        data.num_features,
        args.num_hidden,
        query_logits.size(1),
        args.num_layers,
        args.head,
        args.dropout,
    ).to(device)

    config = GATConfig(args.num_epochs, args.lr, args.wd, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for _ in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, embeddings = model(data.x, data.edge_index)

        if args.recovery_from == "prediction":
            loss = F.kl_div(
                F.log_softmax(logits[train_idx], dim=1),
                F.softmax(query_logits, dim=1),
                reduction="batchmean",
            )
        elif args.recovery_from == "embedding":
            loss = F.mse_loss(embeddings[train_idx], query_embeddings)
        elif args.recovery_from == "projection":
            target_proj = projection(
                query_embeddings.detach().cpu().numpy(),
                data.y[train_idx].cpu().numpy(),
                transform_name=args.transform,
                show_figure=False,
                gnn=args.target_model,
                dataset=args.dataset,
            )
            target_proj = torch.from_numpy(target_proj).float().to(device)
            surrogate_proj = projection(
                embeddings[train_idx].detach().cpu().numpy(),
                data.y[train_idx].cpu().numpy(),
                transform_name=args.transform,
                show_figure=False,
                gnn=args.surrogate_model,
                dataset=args.dataset,
            )
            surrogate_proj = torch.from_numpy(surrogate_proj).float().to(device)
            loss = F.mse_loss(surrogate_proj, target_proj)
        else:
            raise ValueError("recovery-from must be prediction, embedding, or projection")

        loss.backward()
        optimizer.step()

    return model


def main() -> None:
    args = _parse_args()

    device = _resolve_device(args.gpu)

    data, n_classes = load_npz_graph(args.dataset)
    train_idx, val_idx, test_idx = split_graph_different_ratio(
        data, frac_list=[0.3, 0.2, 0.5], ratio=args.query_ratio
    )
    if args.structure != "original":
        raise ValueError("Only 'original' structure is supported in the PyG refactor.")

    data = data.to(device)

    target_model = _load_target_model(args, data, n_classes, device)

    if args.target_model == "gat":
        _, target_logits, target_embeddings = evaluate_gat(target_model, data, train_idx, device)
    elif args.target_model == "gin":
        _, target_logits, target_embeddings = evaluate_gin(target_model, data, train_idx, device)
    else:
        _, target_logits, target_embeddings = evaluate_sage(target_model, data, train_idx, device)

    target_logits = target_logits[train_idx].detach()
    target_embeddings = target_embeddings[train_idx].detach()

    surrogate_model = _train_surrogate(args, data, train_idx, target_logits, target_embeddings, device)
    if args.save_surrogate:
        output_dir = Path(args.surrogate_save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        surrogate_path = output_dir / f"surrogate_{args.surrogate_model}_{args.dataset}.pt"
        torch.save(surrogate_model.state_dict(), surrogate_path)

    if args.surrogate_model == "gat":
        _, surrogate_logits, surrogate_embeddings = evaluate_gat(surrogate_model, data, test_idx, device)
    elif args.surrogate_model == "gin":
        _, surrogate_logits, surrogate_embeddings = evaluate_gin(surrogate_model, data, test_idx, device)
    else:
        _, surrogate_logits, surrogate_embeddings = evaluate_sage(surrogate_model, data, test_idx, device)

    detached_classifier = train_detached_classifier(data.y[test_idx], surrogate_embeddings[test_idx])
    detached_acc = detached_classifier.score(
        surrogate_embeddings[test_idx].detach().cpu().numpy(),
        data.y[test_idx].cpu().numpy(),
    )
    detached_preds = detached_classifier.predict_proba(surrogate_embeddings[test_idx].detach().cpu().numpy())

    if args.target_model == "gat":
        _, target_test_logits, _ = evaluate_gat(target_model, data, test_idx, device)
    elif args.target_model == "gin":
        _, target_test_logits, _ = evaluate_gin(target_model, data, test_idx, device)
    else:
        _, target_test_logits, _ = evaluate_sage(target_model, data, test_idx, device)

    fidelity = compute_fidelity(
        torch.from_numpy(detached_preds).to(device),
        target_test_logits[test_idx].to(device),
    )

    output_folder = Path("./results_acc_fidelity") / f"results_{args.target_model}_{args.target_model_dim}_{args.surrogate_model}_{args.num_hidden}"
    output_folder.mkdir(parents=True, exist_ok=True)
    filename = output_folder / f"{args.dataset}_original.txt"
    with filename.open("a") as handle:
        handle.write(
            f"{args.target_model},{args.target_model_dim},{args.surrogate_model},{args.num_hidden},"
            f"{args.recovery_from},{args.round_index},{args.query_ratio},"
            f"{detached_acc},{detached_acc},{fidelity}\n"
        )


if __name__ == "__main__":
    main()
