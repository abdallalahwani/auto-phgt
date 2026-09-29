"""Frozen experimental protocol shared by the entry points and the final suite.

This module deliberately avoids importing torch so the suite orchestrator stays light.
The values are the settings the ACM and OGB-MAG entry points already used; comparable
modes of one dataset share every setting that applies to them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON = {
    "d_model": 64, "k": 5, "max_hops": 4, "instances_per_path": 4,
    "hgt_layers": 2, "hgt_heads": 4, "fusion_layers": 2, "fusion_heads": 4, "ffn_mult": 2,
    "dropout": 0.3, "pooling": "mean", "token_dropout": 0.1, "weight_decay": 1e-3,
    "epochs": 100, "patience": 20,
}

ACM_HYPERPARAMETERS = {**COMMON, "lr": 2e-3, "acm_validation_size": 180,
                       "training": "full_graph"}

MAG_HYPERPARAMETERS = {**COMMON, "lr": 1e-3, "batch_size": 512, "num_neighbors": [10, 5],
                       "training": "sampled_subgraph"}

PROTOCOL = {
    "model_selection": ("best validation Accuracy; an epoch must strictly improve it, so ties "
                        "keep the earliest epoch; its weights are restored before the single "
                        "test evaluation. Test metrics are never used for selection."),
    "early_stopping": "stop after `patience` epochs without a validation-Accuracy improvement",
    "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                  "lr_schedule": None, "gradient_clipping": None, "precision": "float32"},
    "metrics": ["accuracy", "macro_f1"],
    "evaluation_path_sampling": ("validation and test path instances are drawn with a fixed "
                                 "seed (1) for every run; training draws fresh instances every "
                                 "epoch from a generator seeded with the run seed"),
    "seed_effects": ("set_seed(seed) seeds python/numpy/torch/CUDA; the seed also drives the "
                     "ACM validation carve-out, training path sampling, MAG neighbour "
                     "sampling order, and random meta-path selection"),
    "datasets": {
        "acm": {"source": "PyG HGBDataset('ACM'), used unchanged (reciprocal relations present)",
                "split": ("official HGB masks: 907 train / 2118 test papers; 180 train papers "
                          "held out for validation with torch.Generator(seed) -> 727/180/2118"),
                "hyperparameters": ACM_HYPERPARAMETERS,
                "batching": "full batch: one optimizer step per epoch on all training papers"},
        "ogbn-mag": {"source": "PyG OGB_MAG with ToUndirected()",
                     "split": "official OGB split: 629571 train / 64879 val / 41939 test papers",
                     "hyperparameters": MAG_HYPERPARAMETERS,
                     "batching": ("sampled subgraphs (with-replacement fan-out, de-duplicated); "
                                  "full graph, features, CSR and samplers stay on CPU; only "
                                  "per-batch tensors move to the GPU")},
    },
    "discovery": {"canonical": ("StatisticalDiscoveryModule top-k by "
                                "expected_structural_volume_geometric_mean_v1"),
                  "random": ("k distinct paths drawn uniformly without replacement, with "
                             "random.Random(seed), from the same candidate space "
                             "(discover_candidate_paths, same max_hops)")},
}


def hyperparameters(dataset: str) -> dict:
    table = {"acm": ACM_HYPERPARAMETERS, "ogbn-mag": MAG_HYPERPARAMETERS}
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in table[dataset].items()}


def read_git_head(root) -> str | None:
    """Commit of HEAD read from ``.git`` itself; compute nodes may lack the git binary."""
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def git_state(root: Path = ROOT) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"], check=True, capture_output=True,
                               text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": read_git_head(root), "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}
