"""Run model-stealing attacks against inductive GNNs (PyG)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

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
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    args, _ = parser.parse_known_args()
    return args


DEFAULT_CONFIG = {
    "gpu": -1,
    "dataset": "citeseer_full",
    "num_epochs": 200,
    "num_layers": 2,
    "lr": 0.001,
    "dropout": 0.5,
    "head": 4,
    "wd": 0.0,
    "target_model": "sage",
    "target_model_dim": 256,
    "surrogate_model": "sage",
    "num_hidden": 256,
    "recovery_from": "embedding",
    "round_index": 1,
    "query_ratio": 1.0,
    "structure": "original",
    "transform": "TSNE",
    "logging": {
        "log_every": 10
    },
}


def _load_config(config_path: Path) -> SimpleNamespace:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    merged = {**DEFAULT_CONFIG, **data}
    if "logging" in data:
        merged["logging"] = {**DEFAULT_CONFIG["logging"], **data["logging"]}
    return SimpleNamespace(**merged)


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
    args_path = model_dir / "model_args"

    target_hidden = args.target_model_dim
    target_layers = args.num_layers
    target_heads = args.head
    target_dropout = args.dropout
    if args_path.exists():
        try:
            saved_args = pickle.load(open(args_path, "rb"))
            target_hidden = getattr(saved_args, "num_hidden", target_hidden)
            target_layers = getattr(saved_args, "num_layers", target_layers)
            target_heads = getattr(saved_args, "head", target_heads)
            target_dropout = getattr(saved_args, "dropout", target_dropout)
            print(
                "[Attack] Loaded target model config from disk: "
                f"hidden={target_hidden}, layers={target_layers}, heads={target_heads}, dropout={target_dropout}"
            )
        except Exception as exc:
            print(f"[Attack] Warning: failed to load target model config ({exc}). Using attack config instead.")

    model = _build_model(
        args.target_model,
        data.num_features,
        target_hidden,
        n_classes,
        target_layers,
        target_heads,
        target_dropout,
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

    log_every = max(1, int(args.logging.get("log_every", 10)))
    for epoch in range(1, config.num_epochs + 1):
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
        if epoch == 1 or epoch % log_every == 0 or epoch == config.num_epochs:
            print(f"[Attack] Surrogate training epoch {epoch}/{config.num_epochs} loss={loss.item():.4f}")

    return model


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).with_name("attack_config.json")
    cfg = _load_config(config_path)
    if cfg.structure != "original":
        print(f"[Attack] Warning: structure='{cfg.structure}' is ignored; using original graph structure.")

    device = _resolve_device(cfg.gpu)

    print(f"[Attack] Loading dataset '{cfg.dataset}'.")
    data, n_classes = load_npz_graph(cfg.dataset)
    train_idx, val_idx, test_idx = split_graph_different_ratio(
        data, frac_list=[0.3, 0.2, 0.5], ratio=cfg.query_ratio
    )
    print(f"[Attack] Split sizes -> query: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    data = data.to(device)

    print(f"[Attack] Loading target model '{cfg.target_model}' from disk.")
    target_model = _load_target_model(cfg, data, n_classes, device)

    if cfg.target_model == "gat":
        target_acc, target_logits, target_embeddings = evaluate_gat(target_model, data, train_idx, device)
    elif cfg.target_model == "gin":
        target_acc, target_logits, target_embeddings = evaluate_gin(target_model, data, train_idx, device)
    else:
        target_acc, target_logits, target_embeddings = evaluate_sage(target_model, data, train_idx, device)
    print(f"[Attack] Target model query accuracy={float(target_acc):.4f}")

    target_logits = target_logits[train_idx].detach()
    target_embeddings = target_embeddings[train_idx].detach()

    print(f"[Attack] Training surrogate model '{cfg.surrogate_model}'.")
    surrogate_model = _train_surrogate(cfg, data, train_idx, target_logits, target_embeddings, device)

    if cfg.surrogate_model == "gat":
        surrogate_acc, surrogate_logits, surrogate_embeddings = evaluate_gat(surrogate_model, data, test_idx, device)
    elif cfg.surrogate_model == "gin":
        surrogate_acc, surrogate_logits, surrogate_embeddings = evaluate_gin(surrogate_model, data, test_idx, device)
    else:
        surrogate_acc, surrogate_logits, surrogate_embeddings = evaluate_sage(surrogate_model, data, test_idx, device)
    print(f"[Attack] Surrogate test accuracy={float(surrogate_acc):.4f}")

    detached_classifier = train_detached_classifier(data.y[test_idx], surrogate_embeddings[test_idx])
    detached_acc = detached_classifier.score(
        surrogate_embeddings[test_idx].detach().cpu().numpy(),
        data.y[test_idx].cpu().numpy(),
    )
    print(f"[Attack] Detached classifier accuracy={detached_acc:.4f}")
    detached_preds = detached_classifier.predict_proba(surrogate_embeddings[test_idx].detach().cpu().numpy())

    if cfg.target_model == "gat":
        _, target_test_logits, _ = evaluate_gat(target_model, data, test_idx, device)
    elif cfg.target_model == "gin":
        _, target_test_logits, _ = evaluate_gin(target_model, data, test_idx, device)
    else:
        _, target_test_logits, _ = evaluate_sage(target_model, data, test_idx, device)

    fidelity = compute_fidelity(
        torch.from_numpy(detached_preds).to(device),
        target_test_logits[test_idx].to(device),
    )
    print(f"[Attack] Fidelity={fidelity:.4f}")

    output_folder = Path("./results_acc_fidelity") / (
        f"results_{cfg.target_model}_{cfg.target_model_dim}_{cfg.surrogate_model}_{cfg.num_hidden}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    filename = output_folder / f"{cfg.dataset}_original.txt"
    with filename.open("a") as handle:
        handle.write(
            f"{cfg.target_model},{cfg.target_model_dim},{cfg.surrogate_model},{cfg.num_hidden},"
            f"{cfg.recovery_from},{cfg.round_index},{cfg.query_ratio},"
            f"{detached_acc},{detached_acc},{fidelity}\n"
        )


if __name__ == "__main__":
    main()
