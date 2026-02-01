"""Benchmark GNNFingers detection against common model obfuscations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import prune

from gnnfingers.fingerprints import GraphFingerprintSet
from gnnfingers.univerifier import UniverifierConfig, UniverifierMLP
from gnnfingers.utils import flatten_outputs, load_config
from gnnfingers.wrappers import LocalPyGSuspectWrapper
from src.gat import GAT, evaluate_gat
from src.gin import GIN, evaluate_gin
from src.sage import SAGE, evaluate_sage
from src.utils import load_npz_graph, split_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark GNNFingers defense on suspect variants")
    parser.add_argument("--dataset", type=str, default="citeseer_full")
    parser.add_argument("--target-model", type=str, default="gat")
    parser.add_argument("--target-model-dim", type=int, default=256)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--task", type=str, default="node_classification")
    parser.add_argument("--fingerprints", type=str, required=True)
    parser.add_argument("--univerifier", type=str, required=True)
    parser.add_argument("--lambda-threshold", type=float, default=0.5)
    parser.add_argument("--variants", type=str, default="finetune_last,finetune_full,reset_finetune,distill,prune")
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--distill-epochs", type=int, default=10)
    parser.add_argument("--distill-arch", type=str, default="")
    parser.add_argument("--prune-amount", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="./suspect_models")
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


def _train_model(model, data, train_idx, num_epochs: int) -> None:
    model.to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()


def _fine_tune_last_layer(model, data, train_idx, num_epochs: int) -> None:
    last_layer_idx = len(model.layers) - 1
    for name, param in model.named_parameters():
        param.requires_grad = f"layers.{last_layer_idx}" in name
    _train_model(model, data, train_idx, num_epochs)


def _reset_parameters(model) -> None:
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _distill_student(teacher, student, data, train_idx, num_epochs: int) -> None:
    student.to(data.x.device)
    teacher.to(data.x.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    teacher.eval()
    with torch.no_grad():
        teacher_logits, _ = teacher(data.x, data.edge_index)
        soft_targets = teacher_logits.softmax(dim=1)
    for _ in range(num_epochs):
        student.train()
        optimizer.zero_grad()
        logits, _ = student(data.x, data.edge_index)
        loss = F.kl_div(F.log_softmax(logits[train_idx], dim=1), soft_targets[train_idx], reduction="batchmean")
        loss.backward()
        optimizer.step()


def _prune_model(model, amount: float) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")


def _load_univerifier(univerifier_path: Path, fingerprint_set: GraphFingerprintSet, task: str, device: torch.device) -> UniverifierMLP:
    config_path = univerifier_path.with_name("config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        input_dim = cfg.get("univerifier_input_dim")
    else:
        input_dim = None

    if input_dim is None:
        sample_output = flatten_outputs(
            {"logits": torch.zeros(fingerprint_set.num_nodes, 2), "embeddings": torch.zeros(fingerprint_set.num_nodes, 2)},
            task,
            fingerprint_set.get_query_meta()[0],
        )
        input_dim = sample_output.numel()

    config = UniverifierConfig(input_dim=input_dim, hidden_dims=[128, 64, 32])
    univerifier = UniverifierMLP(config).to(device)
    univerifier.load_state_dict(torch.load(univerifier_path, map_location=device))
    return univerifier


def _verify_model(model, fingerprint_set: GraphFingerprintSet, univerifier: UniverifierMLP, task: str, device: torch.device) -> float:
    graphs = fingerprint_set.build_graphs()
    meta_list = fingerprint_set.get_query_meta()
    wrapper = LocalPyGSuspectWrapper(model, device, task_type=task)
    outputs = wrapper.predict(graphs)
    vectors = [flatten_outputs(outputs[i], task, meta_list[i]) for i in range(len(meta_list))]
    batch = torch.stack(vectors).to(device)
    probs = univerifier(batch)
    return probs[:, 0].mean().item()


def _log_accuracy(model, model_name: str, data, train_idx, test_idx, device) -> None:
    if model_name == "gat":
        evaluate_gat(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    elif model_name == "gin":
        evaluate_gin(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)
    else:
        evaluate_sage(model, data, test_idx, device, train_idx=train_idx, log_metrics=True)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.gpu)

    data, n_classes = load_npz_graph(args.dataset)
    train_idx, _, test_idx = split_graph(data, frac_list=[0.6, 0.2, 0.2])
    data = data.to(device)

    target_dir = Path(f"./target_model_{args.target_model}_{args.target_model_dim}")
    target_path = target_dir / f"target_model_{args.target_model}_{args.dataset}"
    target_model = _build_model(
        args.target_model,
        data.num_features,
        args.target_model_dim,
        n_classes,
        args.num_layers,
        args.head,
        args.dropout,
    ).to(device)
    target_model.load_state_dict(torch.load(target_path, map_location=device))

    fingerprint_set = GraphFingerprintSet.load(args.fingerprints, device)
    univerifier = _load_univerifier(Path(args.univerifier), fingerprint_set, args.task, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    distill_arch = args.distill_arch or args.target_model

    for variant in variants:
        print(f"[Defense] variant={variant}")
        if variant == "finetune_last":
            model = copy.deepcopy(target_model)
            _fine_tune_last_layer(model, data, train_idx, args.finetune_epochs)
        elif variant == "finetune_full":
            model = copy.deepcopy(target_model)
            _train_model(model, data, train_idx, args.finetune_epochs)
        elif variant == "reset_finetune":
            model = copy.deepcopy(target_model)
            _reset_parameters(model)
            _train_model(model, data, train_idx, args.finetune_epochs)
        elif variant == "distill":
            student = _build_model(
                distill_arch,
                data.num_features,
                args.num_hidden,
                n_classes,
                args.num_layers,
                args.head,
                args.dropout,
            ).to(device)
            _distill_student(target_model, student, data, train_idx, args.distill_epochs)
            model = student
        elif variant == "prune":
            model = copy.deepcopy(target_model)
            _prune_model(model, args.prune_amount)
            _train_model(model, data, train_idx, args.finetune_epochs)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        _log_accuracy(model, args.target_model if variant != "distill" else distill_arch, data, train_idx, test_idx, device)
        o_plus = _verify_model(model, fingerprint_set, univerifier, args.task, device)
        verdict = "pirated" if o_plus >= args.lambda_threshold else "irrelevant"
        ckpt_path = output_dir / f"suspect_{variant}_{args.dataset}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Defense] o_plus={o_plus:.4f} verdict={verdict} ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
