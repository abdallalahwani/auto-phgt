"""HGB datasets: loading, HGB-style splits, feature regimes, label-free representations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch_geometric.datasets import HGBDataset

from auto_phgt.tokenization import impute_missing_features

DATASETS = {
    "acm": {"name": "ACM", "root": "data/acm", "target": "paper"},
    "dblp": {"name": "DBLP", "root": "data/hgb/dblp", "target": "author"},
    "freebase": {"name": "Freebase", "root": "data/hgb/freebase", "target": "book"},
    "imdb": {"name": "IMDB", "root": "data/hgb/imdb", "target": "movie"},
}


def load(ds: str):
    spec = DATASETS[ds]
    return HGBDataset(root=spec["root"], name=spec["name"])[0]


def target(ds: str) -> str:
    return DATASETS[ds]["target"]


def split(graph, target_type: str, seed: int, val_ratio: float = 0.2):
    """HGB protocol: official test mask; a random 20% of the official training nodes is
    validation (HGB utils/data.py uses int(n * 0.2) after a shuffle); seeded here."""
    node = graph[target_type]
    train = node.train_mask.nonzero().view(-1)
    test = node.test_mask.nonzero().view(-1)
    order = train[torch.randperm(train.numel(), generator=torch.Generator().manual_seed(seed))]
    n_val = int(train.numel() * val_ratio)
    val, train = order[:n_val].sort().values, order[n_val:].sort().values
    assert not (set(train.tolist()) & set(test.tolist()))
    return train, val, test


def features(graph, target_type: str, regime: str) -> dict:
    """x_dict for a feature regime. Types missing from the dict get learned ID embeddings in
    TypeAwareProjection, which is equivalent to HGB's one-hot input followed by a linear layer."""
    given = {t: graph[t].x.float() for t in graph.node_types if "x" in graph[t]}
    if regime == "given":
        return given
    if regime == "imputed":
        return {t: x.float() for t, x in impute_missing_features(graph).items()}
    if regime == "target_onehot":
        return {t: x for t, x in given.items() if t == target_type}
    if regime == "target_zero":
        out = {t: x for t, x in given.items() if t == target_type}
        for t in graph.node_types:
            if t != target_type:
                out[t] = torch.zeros(graph[t].num_nodes, 10)
        return out
    raise ValueError(f"unknown feature regime {regime!r}")


def _pca(x: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    x = x.float()
    if x.size(1) <= d:
        out = x - x.mean(0, keepdim=True)
    else:
        torch.manual_seed(seed)
        _, _, v = torch.pca_lowrank(x, q=d, center=True, niter=4)
        out = (x - x.mean(0, keepdim=True)) @ v[:, :d]
    out = out / out.std(0, keepdim=True).clamp(min=1e-6)
    if out.size(1) < d:
        out = torch.cat([out, out.new_zeros(out.size(0), d - out.size(1))], 1)
    return out


def label_free_repr(graph, x_dict: dict, d: int = 32, seed: int = 0) -> dict:
    """Fixed per-node vectors mirroring the input regime, computed without any label.

    Dense inputs are PCA-reduced to d dimensions; all-zero inputs (HGB feat 1) stay zero;
    ID-only types (one-hot input) get a fixed random Gaussian vector, i.e. a random
    projection of their one-hot features.
    """
    out = {}
    for i, t in enumerate(graph.node_types):
        n = graph[t].num_nodes
        x = x_dict.get(t)
        if x is not None and x.abs().sum() > 0:
            out[t] = _pca(x, d, seed + i)
        elif x is not None:
            out[t] = torch.zeros(n, d)
        else:
            g = torch.Generator().manual_seed(seed * 1000 + 17 * i + 1)
            out[t] = torch.randn(n, d, generator=g)
    return out


def record(ds: str, graph, seed: int = 0) -> dict:
    t = target(ds)
    train, val, test = split(graph, t, seed)
    processed = Path(DATASETS[ds]["root"]) / DATASETS[ds]["name"].lower() / "processed" / "data.pt"
    digest = hashlib.md5(processed.read_bytes()).hexdigest() if processed.exists() else None
    y = graph[t].y
    return {"dataset": ds, "target": t, "num_classes": int(y.max()) + 1,
            "nodes": {nt: graph[nt].num_nodes for nt in graph.node_types},
            "features": {nt: (list(graph[nt].x.shape) if "x" in graph[nt] else None)
                         for nt in graph.node_types},
            "edges": {"__".join(et): graph[et].num_edges for et in graph.edge_types},
            "official_train": int(graph[t].train_mask.sum()),
            "official_test": int(graph[t].test_mask.sum()),
            "split_seed0": {"train": train.numel(), "val": val.numel(), "test": test.numel()},
            "train_class_counts": torch.bincount(y[graph[t].train_mask]).tolist(),
            "processed_md5": digest}
