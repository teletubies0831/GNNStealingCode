"""Train target GNN models using PyTorch Geometric."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import torch

from src.gat import GAT, TrainConfig as GATConfig, evaluate_gat, train_gat
from src.gin import GIN, TrainConfig as GINConfig, evaluate_gin, train_gin
from src.sage import SAGE, TrainConfig as SAGEConfig, evaluate_sage, train_sage
from src.utils import load_npz_graph, split_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Target model training (PyG)")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device ID, -1 for CPU")
    parser.add_argument("--target-model", type=str, default="gat")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dblp",
        help="['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']",
    )
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--head", type=int, default=4)
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)

    data, n_classes = load_npz_graph(args.dataset)
    train_idx, val_idx, test_idx = split_graph(data, frac_list=[0.6, 0.2, 0.2])
    print(len(train_idx), len(val_idx), len(test_idx))

    save_dir = Path(f"./target_model_{args.target_model}_{args.num_hidden}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"target_model_{args.target_model}_{args.dataset}"

    if args.target_model == "gat":
        model = GAT(data.num_features, args.num_hidden, n_classes, args.num_layers, args.head, args.dropout)
        config = GATConfig(args.num_epochs, args.lr, args.wd, args.dropout)
        train_gat(model, data, train_idx, config, device)
        torch.save(model.state_dict(), save_dir / save_name)
        evaluate_gat(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    elif args.target_model == "gin":
        model = GIN(data.num_features, args.num_hidden, n_classes, args.num_layers, args.dropout)
        config = GINConfig(args.num_epochs, args.lr, args.wd, args.dropout)
        train_gin(model, data, train_idx, config, device)
        torch.save(model.state_dict(), save_dir / save_name)
        evaluate_gin(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    elif args.target_model == "sage":
        model = SAGE(data.num_features, args.num_hidden, n_classes, args.num_layers, args.dropout)
        config = SAGEConfig(args.num_epochs, args.lr, args.wd, args.dropout)
        train_sage(model, data, train_idx, config, device)
        torch.save(model.state_dict(), save_dir / save_name)
        evaluate_sage(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    else:
        raise ValueError("target-model should be gat, gin, or sage")

    pickle.dump(args, open(save_dir / "model_args", "wb"))
    print("Finish")


if __name__ == "__main__":
    main()
