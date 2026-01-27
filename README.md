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

## GNNFingers Ownership Verification (paper-style)

This repository now includes a **paper-aligned GNNFingers pipeline** that trains learnable
fingerprint graphs, a Univerifier MLP, and positive/negative model ensembles to verify GNN
ownership in a black-box setting. The defense focuses on **piracy attribution**, not prevention.
Supported tasks: node classification, link prediction, graph classification, and graph matching.

### Minimal end-to-end example (node classification)

From the repository root:

```
cd code
python train_gnnfingers.py --task node_classification --dataset citeseer_full --target-model gat \
  --hidden-dim 256 --num-layers 2 --num-fingerprints 64 --num-nodes 32 --epsilon 0.1 --iterations 1000 \
  --output-dir ./gnnfingers_out --gpu -1

python verify_suspect.py --fingerprints ./gnnfingers_out/fingerprints.pt \
  --univerifier ./gnnfingers_out/univerifier.pt --lambda-threshold 0.5 \
  --suspect-model gin --num-classes 6 --gpu -1
```

### One-click runner (no CLI args)

If you want a PyCharm-friendly launcher, run:

```
python run_full_pipeline.py
```

This script lives next to `attack.py` and `train_target_model.py` and runs the default
GNNFingers training + verification steps with built-in parameters.

### Design details

- **Fingerprints:** a set of learnable graphs initialized with sparse Bernoulli edges and
  trainable features. Edges are updated using the Top-K sign-flip rule on adjacency gradients.
- **Univerifier:** a 3-layer MLP with LeakyReLU, trained jointly with fingerprints to separate
  pirated vs. irrelevant models.
- **Ensembles:** positive models come from fine-tuning/retraining/distillation variants of the
  target; negative models are trained from scratch with different seeds/architectures.


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
