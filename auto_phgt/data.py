"""Dataset loading shared by discovery, training, and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch_geometric.transforms as T
from torch_geometric.datasets import HGBDataset, OGB_MAG


def load_acm(root: str | Path = "data/acm"):
    """Load HGB ACM, which already has directed reciprocal relation types.

    In particular, ``paper-to-author`` and ``author-to-paper`` are present in
    the source graph. Adding ``ToUndirected`` here would change its schema.
    The returned graph must be used for both discovery and model training.
    """
    return HGBDataset(root=str(root), name="ACM")[0]


def load_mag(root: str | Path = "data/ogb_mag", preprocess=None):
    """Load ogbn-mag as HeteroData with the reverse relations used in notebook 01.

    ``ToUndirected`` also symmetrizes same-type ``paper-cites-paper`` edges.
    Use this returned graph for discovery, path extraction, and HGT training.
    """
    return OGB_MAG(root=str(root), preprocess=preprocess, transform=T.ToUndirected())[0]


def load_dataset(name: str, root: str | Path | None = None):
    normalized = name.lower().replace("_", "-")
    if normalized == "acm":
        return load_acm(root or "data/acm")
    if normalized in {"mag", "ogbn-mag"}:
        return load_mag(root or "data/ogb_mag")
    raise ValueError(f"Unknown dataset {name!r}; expected 'acm' or 'ogbn-mag'")
