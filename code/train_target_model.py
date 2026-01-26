"""Train target GNN models using PyTorch Geometric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import torch

from src.gat import GAT, TrainConfig as GATConfig, evaluate_gat, train_gat
from src.gin import GIN, TrainConfig as GINConfig, evaluate_gin, train_gin
from src.sage import SAGE, TrainConfig as SAGEConfig, evaluate_sage, train_sage
from src.utils import load_npz_graph, split_graph


DEFAULT_CONFIG = {
    "gpu": -1,
    "target_model": "gat",
    "dataset": "dblp",
    "num_epochs": 200,
    "num_hidden": 256,
    "num_layers": 3,
    "lr": 0.001,
    "dropout": 0.5,
    "wd": 0.0,
    "head": 4,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Target model training (PyG)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    args, _ = parser.parse_known_args()
    return args


def _load_config(config_path: Path) -> SimpleNamespace:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    merged = {**DEFAULT_CONFIG, **data}
    return SimpleNamespace(**merged)


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config) if args.config else Path(__file__).with_name("train_config.json")
    cfg = _load_config(config_path)
    device = _resolve_device(cfg.gpu)

    print(f"[Train] Loading dataset '{cfg.dataset}'.")
    data, n_classes = load_npz_graph(cfg.dataset)
    train_idx, val_idx, test_idx = split_graph(data, frac_list=[0.6, 0.2, 0.2])
    print(f"[Train] Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    save_dir = Path(f"./target_model_{cfg.target_model}_{cfg.num_hidden}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"target_model_{cfg.target_model}_{cfg.dataset}"

    if cfg.target_model == "gat":
        print("[Train] Training target GAT model.")
        model = GAT(data.num_features, cfg.num_hidden, n_classes, cfg.num_layers, cfg.head, cfg.dropout)
        config = GATConfig(cfg.num_epochs, cfg.lr, cfg.wd, cfg.dropout)
        train_gat(model, data, train_idx, config, device)
        torch.save(model.state_dict(), save_dir / save_name)
        test_acc, _, _ = evaluate_gat(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    elif cfg.target_model == "gin":
        print("[Train] Training target GIN model.")
        model = GIN(data.num_features, cfg.num_hidden, n_classes, cfg.num_layers, cfg.dropout)
        config = GINConfig(cfg.num_epochs, cfg.lr, cfg.wd, cfg.dropout)
        train_gin(model, data, train_idx, config, device)
        torch.save(model.state_dict(), save_dir / save_name)
        test_acc, _, _ = evaluate_gin(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    elif cfg.target_model == "sage":
        print("[Train] Training target GraphSAGE model.")
        model = SAGE(data.num_features, cfg.num_hidden, n_classes, cfg.num_layers, cfg.dropout)
        config = SAGEConfig(cfg.num_epochs, cfg.lr, cfg.wd, cfg.dropout)
        train_sage(model, data, train_idx, config, device)
        torch.save(model.state_dict(), save_dir / save_name)
        test_acc, _, _ = evaluate_sage(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    else:
        raise ValueError("target-model should be gat, gin, or sage")

    print(f"[Train] Finished training. Test accuracy={float(test_acc):.4f}")
    pickle.dump(cfg, open(save_dir / "model_args", "wb"))
    print("[Train] Done.")


if __name__ == "__main__":
    main()
