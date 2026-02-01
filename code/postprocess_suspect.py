"""Fine-tune or prune suspect checkpoints for GNNFingers evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.gat import GAT, TrainConfig as GATConfig, train_gat
from src.gin import GIN, TrainConfig as GINConfig, train_gin
from src.sage import SAGE, TrainConfig as SAGEConfig, train_sage
from src.utils import load_npz_graph, split_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Postprocess suspect model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID, -1 for CPU")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--suspect-model", type=str, default="gat")
    parser.add_argument("--suspect-ckpt", type=str, required=True)
    parser.add_argument("--suspect-hidden", type=int, default=256)
    parser.add_argument("--suspect-layers", type=int, default=3)
    parser.add_argument("--suspect-heads", type=int, default=4)
    parser.add_argument("--suspect-dropout", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--finetune-lr", type=float, default=0.001)
    parser.add_argument("--finetune-wd", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--prune-ratio", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./suspect_models")
    parser.add_argument("--output-tag", type=str, default="")
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


def _apply_pruning(model: torch.nn.Module, prune_ratio: float) -> int:
    if prune_ratio <= 0:
        return 0
    parameters = [param for param in model.parameters() if param.requires_grad and param.dim() > 1]
    if not parameters:
        return 0
    with torch.no_grad():
        all_weights = torch.cat([param.detach().abs().flatten() for param in parameters]).cpu()
        threshold = torch.quantile(all_weights, prune_ratio).item()
        pruned = 0
        for param in parameters:
            mask = param.detach().abs() <= threshold
            pruned += int(mask.sum().item())
            param[mask] = 0.0
    return pruned


def _fine_tune(model, data, train_idx, args, device: torch.device) -> None:
    if args.suspect_model == "gat":
        config = GATConfig(args.finetune_epochs, args.finetune_lr, args.finetune_wd, args.suspect_dropout, log_every=args.log_every)
        train_gat(model, data, train_idx, config, device)
    elif args.suspect_model == "gin":
        config = GINConfig(args.finetune_epochs, args.finetune_lr, args.finetune_wd, args.suspect_dropout, log_every=args.log_every)
        train_gin(model, data, train_idx, config, device)
    elif args.suspect_model == "sage":
        config = SAGEConfig(args.finetune_epochs, args.finetune_lr, args.finetune_wd, args.suspect_dropout, log_every=args.log_every)
        train_sage(model, data, train_idx, config, device)
    else:
        raise ValueError("suspect-model should be gat, gin, or sage")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)

    data, _ = load_npz_graph(args.dataset)
    train_idx, _, _ = split_graph(data, frac_list=[0.6, 0.2, 0.2])

    model = _build_model(
        args.suspect_model,
        data.num_features,
        args.suspect_hidden,
        args.num_classes,
        args.suspect_layers,
        args.suspect_heads,
        args.suspect_dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.suspect_ckpt, map_location=device))

    if args.finetune_epochs > 0:
        print(f"[Postprocess] fine-tune epochs={args.finetune_epochs}")
        _fine_tune(model, data, train_idx, args, device)

    pruned = _apply_pruning(model, args.prune_ratio)
    if args.prune_ratio > 0:
        print(f"[Postprocess] prune_ratio={args.prune_ratio:.2f} pruned_params={pruned}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    output_path = output_dir / f"suspect_{args.suspect_model}_{args.dataset}{tag}.pt"
    torch.save(model.state_dict(), output_path)
    print(f"[Postprocess] saved suspect checkpoint to {output_path}")


if __name__ == "__main__":
    main()
