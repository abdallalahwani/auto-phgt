"""Frozen V3 protocol: datasets, tuning grid, RCMS settings, conditions, and seeds.

Everything that can influence a V3 result is defined here and hashed into
artifacts/v3/protocol/protocol_lock.json before the campaign starts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = Path("artifacts/v3")

DATASETS = {
    "acm": {"name": "ACM", "root": "data/acm", "target": "paper",
            "hgb_reference": {"feat": "given", "layers": 2, "lr": 1e-3, "wd": 1e-4,
                              "dropout": 0.2, "use_norm": False}},
    "dblp": {"name": "DBLP", "root": "data/hgb/dblp", "target": "author",
             "hgb_reference": {"feat": "target_onehot", "layers": 3, "lr": 1e-3, "wd": 1e-4,
                               "dropout": 0.2, "use_norm": True}},
    "freebase": {"name": "Freebase", "root": "data/hgb/freebase", "target": "book",
                 "hgb_reference": {"feat": "target_onehot", "layers": 3, "lr": 1e-3, "wd": 0.0,
                                   "dropout": 0.2, "use_norm": True}},
}
HGB_DATASETS = tuple(DATASETS)

# Feature regimes (HGB --feats-type): given = 0 (featureless types one-hot), target_zero = 1,
# target_onehot = 2; imputed = the V1 mean-of-neighbours imputation.
FEATURE_REGIMES = {"acm": ["given", "target_zero", "target_onehot", "imputed"],
                   "dblp": ["given", "target_zero", "target_onehot", "imputed"],
                   "freebase": ["target_onehot"]}

GRID = {"layers": [2, 3], "lr": [5e-4, 1e-3, 5e-3], "dropout": [0.2, 0.5],
        "wd": {"acm": [1e-4, 1e-3], "dblp": [1e-4, 1e-3], "freebase": [0.0, 1e-4, 1e-3]}}
TUNE_SEEDS = [100, 101]
BACKBONE_FIXED = {"d_model": 64, "heads": 8, "max_epochs": 300, "patience": 30,
                  "schedule": "onecycle", "pct_start": 0.05, "val_ratio": 0.2,
                  "early_stopping": "validation loss", "optimizer": "AdamW"}
AUTO_PHGT = {"fusion_layers": 2, "fusion_heads": 4, "ffn_mult": 2, "pooling": "mean",
             "token_dropout": 0.1, "instances_per_path": 4}
REPRODUCTION_TARGET = {"acm": 91.0, "warning_below": 90.5}

RCMS = {"folds": 3, "search_instances": 8, "d_z": 32, "probe_l2": 1e-2, "probe_lr": 0.05,
        "probe_steps": 300, "pool_after_step1": 200, "coverage_probe_nodes": 512,
        "min_coverage": 1e-3, "search_seed_offset": 1000}

MAG = {"epochs": 150, "continue_to": 250, "continue_min_gain": 0.001, "continue_window": 20,
       "search_holdout": 0.2, "search_subset": 30000, "search_epochs": 30,
       "config": "V1/V2 ogbn-mag configuration (experiments/protocol.py), not re-tuned"}

SEEDS = {"acm": [0, 1, 2, 3, 4], "dblp": [0, 1, 2, 3, 4], "freebase": [0, 1, 2, 3, 4],
         "ogbn-mag": [0, 1, 2]}
FREEBASE_EXTENSION = {"seeds": [5, 6, 7, 8, 9], "sd_threshold": 0.015,
                      "rule": ("extend all Freebase conditions to 10 seeds if the mean over "
                               "Freebase P0 conditions of the sample SD of VALIDATION micro-F1 "
                               "(5 seeds) is >= 1.5 points; test results are never read")}

MANUAL_PATHS = {
    "acm": [["paper", "to", "author", "to", "paper"], ["paper", "to", "subject", "to", "paper"]],
    "dblp": [["author", "to", "paper", "to", "author"],
             ["author", "to", "paper", "to", "term", "to", "paper", "to", "author"],
             ["author", "to", "paper", "to", "venue", "to", "paper", "to", "author"]],
}
LMSPS_COMMIT = "1a1e2e8a087fb1324c1faa8d12a82c5e8580666b"
LMSPS_DBLP = ['AP', 'APT', 'APVP', 'APAPA', 'APTPA', 'APTPT', 'APTPV', 'APVPA', 'APVPV', 'APAPAP',
              'APAPTP', 'APAPVP', 'APTPTP', 'APTPVP', 'APVPTP', 'APAPAPV', 'APAPTPA', 'APAPTPV',
              'APAPVPA', 'APAPVPT', 'APAPVPV', 'APTPAPA', 'APTPAPT', 'APTPTPT', 'APTPVPV',
              'APVPAPT', 'APVPTPA', 'APVPVPA', 'APVPVPT', 'APVPVPV']
LMSPS_LETTERS = {"A": "author", "P": "paper", "T": "term", "V": "venue"}


def condition(ds, name, mode, *, selector=None, k=None, lmax=4, token_control=None,
              width=None, priority=0, group):
    return {"id": f"{ds}__{name}", "dataset": ds, "name": name, "mode": mode,
            "selector": selector, "k": k, "lmax": lmax, "token_control": token_control,
            "width": width, "priority": priority, "group": group}


def hgb_conditions() -> list[dict]:
    """Final-run conditions (each run for every seed of its dataset)."""
    out = []
    for ds in HGB_DATASETS:
        out.append(condition(ds, "hgt_strong", "hgt", group="backbone"))
    for ds in ("acm", "dblp"):
        out.append(condition(ds, "hgt_param_matched", "hgt", width="pm", group="capacity"))
    for sel in ("canonical", "random", "diverse", "homophily", "hybrid", "rcms", "rcms_indep",
                "rcms_nohgt"):
        out.append(condition("acm", f"{sel}_k5", "auto_phgt", selector=sel, k=5,
                             group="selectors_k5"))
    out.append(condition("acm", "rcms_k5_dummy", "auto_phgt", selector="rcms", k=5,
                         token_control="dummy", group="capacity"))
    out.append(condition("acm", "rcms_k5_shuffle", "auto_phgt", selector="rcms", k=5,
                         token_control="shuffle", group="capacity"))
    for sel in ("canonical", "random", "hybrid", "manual", "rcms", "rcms_indep", "rcms_nohgt"):
        out.append(condition("acm", f"{sel}_k2", "auto_phgt", selector=sel, k=2,
                             group="selectors_k2"))
    out.append(condition("acm", "rcms_k5_L6", "auto_phgt", selector="rcms", k=5, lmax=6,
                         priority=1, group="long_range"))
    # DBLP has only 4 target-to-target candidates within 4 hops: k=5 is infeasible there.
    out.append(condition("dblp", "all4_L4", "auto_phgt", selector="all", k=4,
                         group="selectors_k5"))
    for sel in ("canonical", "random", "hybrid", "rcms", "rcms_indep", "rcms_nohgt"):
        out.append(condition("dblp", f"{sel}_k2", "auto_phgt", selector=sel, k=2,
                             group="selectors_k2"))
        out.append(condition("dblp", f"{sel}_k5_L6", "auto_phgt", selector=sel, k=5, lmax=6,
                             group="selectors_k5"))
    out.append(condition("dblp", "rcms_k5_L6_dummy", "auto_phgt", selector="rcms", k=5, lmax=6,
                         token_control="dummy", group="capacity"))
    out.append(condition("dblp", "expert_k3", "auto_phgt", selector="manual", k=3, priority=1,
                         group="external_paths"))
    out.append(condition("dblp", "lmsps_paths_k30", "auto_phgt", selector="lmsps", k=30,
                         lmax=6, priority=1, group="external_paths"))
    for sel in ("canonical", "random", "hybrid", "rcms", "rcms_indep", "rcms_nohgt"):
        out.append(condition("freebase", f"{sel}_k5", "auto_phgt", selector=sel, k=5,
                             group="selectors_k5"))
    for sel in ("canonical", "random", "hybrid", "rcms"):
        out.append(condition("freebase", f"{sel}_k2", "auto_phgt", selector=sel, k=2,
                             group="selectors_k2"))
    out.append(condition("freebase", "rcms_k5_L6", "auto_phgt", selector="rcms", k=5, lmax=6,
                         priority=1, group="long_range"))
    return out


def search_lmax(ds) -> list[int]:
    """Search spaces per dataset: L<=4 always; L<=6 where a condition needs it."""
    return sorted({c["lmax"] for c in hgb_conditions()
                   if c["dataset"] == ds and c["selector"] and c["selector"].startswith("rcms")})


MAG_CONDITIONS = [
    {"id": "mag__hgt", "name": "hgt", "mode": "hgt", "selection": "discovered", "token_control": None},
    {"id": "mag__canonical_k5", "name": "canonical_k5", "mode": "auto_phgt", "selection": "discovered", "token_control": None},
    {"id": "mag__random_k5", "name": "random_k5", "mode": "auto_phgt", "selection": "random", "token_control": None},
    # Dummy tokens ignore path content, so the path set only fixes the token count (5 x 4).
    {"id": "mag__dummy_k5", "name": "dummy_k5", "mode": "auto_phgt", "selection": "discovered", "token_control": "dummy"},
    {"id": "mag__rcms_k5", "name": "rcms_k5", "mode": "auto_phgt", "selection": "rcms", "token_control": None},
]

HYPOTHESES = {
    "H1": "Strong HGT calibration materially reduces the baseline gap relative to HGB published HGT.",
    "H2": "RCMS selects better path sets than the original structural-volume selector.",
    "H3": "RCMS outperforms the strongest V2 heuristic (hybrid) on clean confirmation datasets where applicable.",
    "H4": "RCMS-Auto-PHGT improves over strong HGT, showing semantic paths add information beyond the backbone.",
    "H5": "Full RCMS outperforms independent residual ranking, supporting set-conditioning.",
    "H6": "Full RCMS outperforms conditional selection without HGT, supporting backbone-conditioning.",
    "H7": "Real semantic tokens outperform dummy/shuffled capacity controls.",
    "H8": "RCMS provides a favorable accuracy/search-cost tradeoff relative to task-driven automatic path-search competitors.",
}


def protocol_record() -> dict:
    return {"datasets": DATASETS, "feature_regimes": FEATURE_REGIMES, "grid": GRID,
            "tune_seeds": TUNE_SEEDS, "backbone_fixed": BACKBONE_FIXED, "auto_phgt": AUTO_PHGT,
            "rcms": RCMS, "mag": MAG, "seeds": SEEDS, "freebase_extension": FREEBASE_EXTENSION,
            "manual_paths": MANUAL_PATHS, "lmsps": {"commit": LMSPS_COMMIT, "dblp": LMSPS_DBLP},
            "hgb_conditions": hgb_conditions(), "mag_conditions": MAG_CONDITIONS,
            "hypotheses": HYPOTHESES, "reproduction_target": REPRODUCTION_TARGET}


def protocol_hash() -> str:
    blob = json.dumps(protocol_record(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


if os.environ.get("V3_SMOKE") == "1":  # tiny settings for the pre-launch GPU smoke test only
    V3 = Path("artifacts/v3_smoke")
    GRID = {"layers": [2], "lr": [1e-3], "dropout": [0.2],
            "wd": {"acm": [1e-4], "dblp": [1e-4], "freebase": [0.0]}}
    FEATURE_REGIMES = {"acm": ["given"], "dblp": ["target_onehot"], "freebase": ["target_onehot"]}
    TUNE_SEEDS = [100]
    BACKBONE_FIXED = {**BACKBONE_FIXED, "max_epochs": 6, "patience": 3}
    RCMS = {**RCMS, "probe_steps": 20}
    MAG = {**MAG, "epochs": 1, "search_epochs": 1, "search_subset": 2000, "continue_to": 2}
    SEEDS = {"acm": [0], "dblp": [0], "freebase": [0], "ogbn-mag": [0]}
    FREEBASE_EXTENSION = {**FREEBASE_EXTENSION, "seeds": [5]}
