# Model Stealing Attacks Against Inductive Graph Neural Networks

This is a PyTorch implementation of Model Stealing Attacks Against Inductive Graph Neural Networks, as described in our paper:

Yun Shen, Xinlei He, Yufei Han, Yang Zhang, [Model Stealing Attacks Against Inductive Graph Neural Networks](https://arxiv.org/abs/2112.08331) (IEEE S&P 2022)

## Step 0: Setup the environment (PyTorch Geometric, CPU-only)

From the repository root:

```
conda env create --file environment.yaml
conda activate gnn_model_stealing
```

This environment installs **PyTorch Geometric (PyG)** and all required dependencies for CPU-only
training and attacks. No GraphGallery or DGL setup is needed.

If you see pip attempting to build `torch-geometric` from source, it means you are using an
older `environment.yaml`. Make sure the file contains `pyg=2.3.1` under conda dependencies
and **does not** list `torch-geometric` under `pip:` before recreating the environment.

## Step 1: Train the target models

```
cd code
python train_target_model.py --dataset citeseer_full --target-model gat --num-hidden 256 --gpu -1

# Note: use --gpu -1 for CPU.
```
Note that we use the following datasets, target model architectures, and numbers of hidden neurons in our paper:
```
--dataset:      ['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']
--target-model: ['gat', 'gin', 'sage']
--num-hidden:   [64, 128, 256]
```

## Step 2: Conduct the model stealing attacks

```
# Type I attack:
python attack.py --dataset citeseer_full --target-model-dim 256 --num-hidden 256 --target-model gat --surrogate-model gin --recovery-from prediction --query_ratio 1.0 --structure original --gpu -1

# Save the surrogate for downstream verification:
python attack.py --dataset citeseer_full --target-model-dim 256 --num-hidden 256 --target-model gat --surrogate-model gin --recovery-from prediction --query_ratio 1.0 --structure original --save-surrogate --gpu -1
```

Explainations:
```
--dataset:          ['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']  # Datasets used to train the surrogate model
--target-model-dim: [64, 128, 256]                                                              # Numbers of hidden neurons for the target model
--num-hidden:       [64, 128, 256]                                                              # Numbers of hidden neurons for the surrogate model
--target-model:     ['gat', 'gin', 'sage']                                                      # Target model's architecuture
--surrogate-model:  ['gat', 'gin', 'sage']                                                      # Surrogate model's architecuture
--recovery-from:    ['prediction', 'embedding', 'projection']                                   # Target model's response
--query_ratio:      [0.1, 0.2, ..., 1.0]                                                        # Ratio of query graph used to train the surrogate model, e.g., 1.0 means we use the whole query graph (30% of the whole dataset); 0.5 means we use half of the query graph (15% of the whole dataset);
--structure:        ['original']                                                                # Type I attacks using the original graph structure.
--save-surrogate:   Save the surrogate checkpoint for downstream verification.
```


## Notes

1. To train the target model, we randomly sample 60% of the nodes to construct the training graph;
2. To train the surrogate model, for each dataset, we split them into three parts.
    - The first part consists of 20\% randomly sampled nodes that are left;
    - The second part consists of 30\% randomly sampled nodes, forming our query graph $\mathbf{G}_Q$.
    - The third part consists of the rest 50\% of the nodes, functioning as the testing data for both $\mathcal{M}_T$ and $\mathcal{M}_S$.
3. We follow the official IDGL implementation from [IDGL](https://github.com/hugochan/IDGL). The core IDGL module lives in `code/core/idgl.py`.

## GNNFingers Ownership Verification

This repository includes a **GNNFingers-style ownership verification** pipeline to determine
whether a suspect model is stolen from a source model. The verification is based on **fingerprint
queries** (small subgraphs with motif + feature triggers) and **response similarity** between
source and suspect models. This defense is intended for **attribution**, not preventing theft.

### Minimal end-to-end example

From the repository root:

```
# 1) Train a source model.
cd code
python train_target_model.py --dataset citeseer_full --target-model gat --num-hidden 256 --gpu -1

# 2) Run StealGNN to obtain a surrogate model checkpoint.
python attack.py --dataset citeseer_full --target-model-dim 256 --num-hidden 256 --target-model gat \
  --surrogate-model gin --recovery-from prediction --query_ratio 1.0 --structure original --save-surrogate --gpu -1

# 3) Build fingerprint queries and source signatures.
python scripts/build_fingerprints.py --dataset citeseer_full --source-model gat \
  --source-model-dir ./target_model_gat_256 --output-dir ./fingerprints --mode embedding

# 4) Verify a suspect model (e.g., the surrogate).
python scripts/verify_ownership.py --dataset citeseer_full --fingerprints ./fingerprints \
  --suspect-ckpt ./surrogate_models/surrogate_gin_citeseer_full.pt --suspect-model gin \
  --suspect-hidden 256 --suspect-layers 2 --suspect-heads 4 --suspect-dropout 0.5 --mode embedding \
  --source-aggregates ./source_scores.txt
```

To build the one-class threshold file (`source_scores.txt`), train multiple source models with
different random seeds and run:

```
python scripts/compute_source_scores.py --dataset citeseer_full --fingerprints ./fingerprints \
  --source-list ./source_list.csv --mode embedding --output ./source_scores.txt
```

Example `source_list.csv` format:

```
./target_model_gat_256/target_model_gat_citeseer_full,gat,256,2,4,0.5
./target_model_gin_256/target_model_gin_citeseer_full,gin,256,2,4,0.5
```

### One-click pipeline (JSON config)

You can run the full training → stealing → fingerprinting → verification flow using a single
launcher script with a JSON configuration file:

```
python scripts/run_gnnfingers_pipeline.py --config ./pipeline_config.json
```

Example `pipeline_config.json` (all step parameters are passed as JSON):

```
{
  "suspect_list": {
    "path": "./suspect_list.csv",
    "rows": [
      ["surrogate", "./surrogate_models/surrogate_gin_citeseer_full.pt", "gin", 256, 2, 4, 0.5, 1],
      ["neg_1", "./negative_models/neg_gat.pt", "gat", 256, 2, 4, 0.5, 0]
    ]
  },
  "source_list": {
    "path": "./source_list.csv",
    "rows": [
      ["./target_model_gat_256/target_model_gat_citeseer_full", "gat", 256, 2, 4, 0.5]
    ]
  },
  "steps": [
    {
      "name": "train_target",
      "script": "train_target_model.py",
      "args": {
        "dataset": "citeseer_full",
        "target_model": "gat",
        "num_hidden": 256,
        "num_layers": 2,
        "gpu": -1
      }
    },
    {
      "name": "steal_gnn",
      "script": "attack.py",
      "args": {
        "dataset": "citeseer_full",
        "target_model_dim": 256,
        "num_hidden": 256,
        "target_model": "gat",
        "surrogate_model": "gin",
        "recovery_from": "prediction",
        "query_ratio": 1.0,
        "structure": "original",
        "save_surrogate": true,
        "gpu": -1
      }
    },
    {
      "name": "build_fingerprints",
      "script": "scripts/build_fingerprints.py",
      "args": {
        "dataset": "citeseer_full",
        "source_model": "gat",
        "source_model_dir": "./target_model_gat_256",
        "output_dir": "./fingerprints",
        "mode": "embedding"
      }
    },
    {
      "name": "compute_source_scores",
      "script": "scripts/compute_source_scores.py",
      "args": {
        "dataset": "citeseer_full",
        "fingerprints": "./fingerprints",
        "source_list": "./source_list.csv",
        "mode": "embedding",
        "output": "./source_scores.txt"
      }
    },
    {
      "name": "verify_ownership",
      "script": "scripts/verify_ownership.py",
      "args": {
        "dataset": "citeseer_full",
        "fingerprints": "./fingerprints",
        "suspect_ckpt": "./surrogate_models/surrogate_gin_citeseer_full.pt",
        "suspect_model": "gin",
        "suspect_hidden": 256,
        "suspect_layers": 2,
        "suspect_heads": 4,
        "suspect_dropout": 0.5,
        "mode": "embedding",
        "source_aggregates": "./source_scores.txt"
      }
    },
    {
      "name": "eval_report",
      "script": "scripts/eval_gnnfingers_on_stealgnn.py",
      "args": {
        "dataset": "citeseer_full",
        "fingerprints": "./fingerprints",
        "suspect_list": "./suspect_list.csv",
        "mode": "embedding",
        "output_csv": "./gnnfingers_report.csv"
      }
    }
  ]
}
```

### Evaluation script

Prepare a CSV-like file listing suspects for evaluation (one line per model):

```
surrogate,./surrogate_models/surrogate_gin_citeseer_full.pt,gin,256,2,4,0.5,1
neg_1,./negative_models/neg_gat.pt,gat,256,2,4,0.5,0
```

Then run:

```
python scripts/eval_gnnfingers_on_stealgnn.py --dataset citeseer_full --fingerprints ./fingerprints \
  --suspect-list ./suspect_list.csv --mode embedding --output-csv ./gnnfingers_report.csv
```

The output CSV includes per-model scores, labels, and verdicts. The script also prints AUC/TPR when
labels include both positives and negatives.

### Design details

- **Anchor selection:** by default, anchors are sampled uniformly at random. You can switch to
  high-degree anchors via `FingerprintConfig.anchor_strategy = "high_degree"` in
  `gnnfingers/fingerprint_builder.py`.
- **Subgraph queries:** for node classification, each anchor yields a `k`-hop induced subgraph.
  Motifs and trigger features are injected into this subgraph while preserving the original
  feature dimensionality, so it stays compatible with PyG `Data` inputs.
- **Robustness intuition:** ownership verification compares **response consistency** on the
  fingerprint queries rather than matching parameters. This makes it more robust to fine-tuning,
  distillation, or architecture changes, because stolen models are trained to mimic the source
  model’s behavior on query inputs.


## Cite

If you use this code, please consider citing the following papers:

```
@inproceedings{SHHZ22,
author = {Yun Shen and Xinlei He and Yufei Han and Yang Zhang},
title = {{Model Stealing Attacks Against Inductive Graph Neural Networks}},
booktitle = {{IEEE Symposium on Security and Privacy (S\&P)}},
publisher = {IEEE},
year = {2022}
}

@inproceedings{CWZ20,
author = {Yu Chen and Lingfei Wu and Mohammed J. Zaki},
title = {{Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings}},
booktitle = {{Annual Conference on Neural Information Processing Systems (NeurIPS)}},
publisher = {NeurIPS},
year = {2020}
}
```
