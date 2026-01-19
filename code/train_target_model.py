"""Train target GNN models on NPZ datasets without GraphGallery.

This script focuses on training the target model that will later be queried by
the model-stealing attack pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import torch

from src.gat import run_gat_target
from src.gin import run_gin_target
from src.sage import run_sage_target
from src.utils import load_npz_graph, split_graph


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for target model training."""
    parser = argparse.ArgumentParser("Target model training")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device ID, -1 for CPU")
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
    parser.add_argument("--fan-out", type=str, default="10,10,10")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--save-pred", type=str, default="")
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--wd", type=float, default=0)
    args, _ = parser.parse_known_args()

    if args.target_model == "sage":
        # GraphSAGE uses inductive sampling with a different default fan-out.
        args.inductive = True
        args.fan_out = "10,25"

    return args


def _resolve_device(gpu_id: int) -> torch.device:
    """Return a CUDA or CPU device based on the requested GPU id."""
    return torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")


def main() -> None:
    """Entry point for training target models."""
    args = _parse_args()
    device = _resolve_device(args.gpu)

    # Load dataset from the NPZ archive and prepare splits.
    graph, n_classes = load_npz_graph(args.dataset)
    in_feats = graph.ndata["features"].shape[1]
    labels = graph.ndata["labels"]

    train_g, val_g, test_g = split_graph(graph, frac_list=[0.6, 0.2, 0.2])
    print(train_g.number_of_nodes(), val_g.number_of_nodes(), test_g.number_of_nodes())

    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    save_dir = Path(f"./target_model_{args.target_model}_{args.num_hidden}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"target_model_{args.target_model}_{args.dataset}"

    # Train the requested target model and save the checkpoint.
    if args.target_model == "gat":
        data = train_g, val_g, test_g, in_feats, labels, n_classes, graph, args.head
        target_model = run_gat_target(args, device, data)
        torch.save(target_model, save_dir / save_name)
    elif args.target_model == "gin":
        data = train_g, val_g, test_g, in_feats, labels, n_classes
        target_model = run_gin_target(args, device, data)
        torch.save(target_model.state_dict(), save_dir / save_name)
    elif args.target_model == "sage":
        data = in_feats, n_classes, train_g, val_g, test_g
        target_model = run_sage_target(args, device, data)
        torch.save(target_model, save_dir / save_name)
    else:
        raise ValueError("target-model should be gat, gin, or sage")

    pickle.dump(args, open(save_dir / "model_args", "wb"))
    print("Finish")


if __name__ == "__main__":
    main()
