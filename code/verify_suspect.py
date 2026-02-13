"""Verify suspect models using trained GNNFingers artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnnfingers.fingerprints import GraphFingerprintSet
from gnnfingers.univerifier import UniverifierConfig, UniverifierMLP
from gnnfingers.utils import flatten_outputs, load_config
from gnnfingers.wrappers import HttpSuspectWrapper, LocalPyGSuspectWrapper
from src.gat import GAT
from src.gin import GIN
from src.sage import SAGE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Verify suspect with GNNFingers")
    parser.add_argument("--fingerprints", type=str, required=True)
    parser.add_argument("--univerifier", type=str, required=True)
    parser.add_argument("--lambda-threshold", type=float, default=-1.0)
    parser.add_argument("--task", type=str, default="node_classification")
    parser.add_argument("--suspect-model", type=str, default="gat")
    parser.add_argument("--suspect-ckpt", type=str, default="")
    parser.add_argument("--suspect-hidden", type=int, default=256)
    parser.add_argument("--suspect-layers", type=int, default=2)
    parser.add_argument("--suspect-heads", type=int, default=4)
    parser.add_argument("--suspect-dropout", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--api-endpoint", type=str, default="")
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


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)

    fingerprint_set = GraphFingerprintSet.load(args.fingerprints, device)
    graphs = fingerprint_set.build_graphs()
    meta_list = fingerprint_set.get_query_meta()

    config_path = Path(args.univerifier).with_name("config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        input_dim = cfg.get("univerifier_input_dim")
        trained_lambda = cfg.get("lambda_threshold")
    else:
        input_dim = None
        trained_lambda = None

    lambda_threshold = args.lambda_threshold
    if lambda_threshold < 0:
        lambda_threshold = float(trained_lambda) if trained_lambda is not None else 0.5

    if input_dim is None:
        sample_output = flatten_outputs(
            {"logits": torch.zeros(fingerprint_set.num_nodes, 2), "embeddings": torch.zeros(fingerprint_set.num_nodes, 2)},
            args.task,
            meta_list[0],
        )
        input_dim = sample_output.numel()

    config = UniverifierConfig(input_dim=input_dim, hidden_dims=[128, 64, 32])
    univerifier = UniverifierMLP(config).to(device)
    univerifier.load_state_dict(torch.load(args.univerifier, map_location=device))

    if args.api_endpoint:
        wrapper = HttpSuspectWrapper(args.api_endpoint)
        outputs = wrapper.predict(graphs)
    else:
        model = _build_model(
            args.suspect_model,
            fingerprint_set.feat_dim,
            args.suspect_hidden,
            args.num_classes,
            args.suspect_layers,
            args.suspect_heads,
            args.suspect_dropout,
        ).to(device)
        if args.suspect_ckpt:
            model.load_state_dict(torch.load(args.suspect_ckpt, map_location=device))
        wrapper = LocalPyGSuspectWrapper(model, device, task_type=args.task)
        outputs = wrapper.predict(graphs)

    vectors = [flatten_outputs(outputs[i], args.task, meta_list[i]) for i in range(len(meta_list))]
    batch = torch.stack(vectors).to(device)
    probs = univerifier(batch)
    o_plus = probs[:, 0].mean().item()

    verdict = "pirated" if o_plus >= lambda_threshold else "irrelevant"
    print(f"o_plus={o_plus:.4f}")
    print(f"lambda={lambda_threshold:.4f}")
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
