"""Frozen V4-Hedge protocol: datasets, backbone grid, selector constants, conditions, seeds,
hypotheses and decision rules.

Everything that can influence a V4 result is defined here and hashed into
artifacts/v4_hedge/protocol/protocol_lock.json before the campaign starts. V4 reuses V3 *code*
(the HGB loaders/splits and the HGB-style trainer) but never reads a V3 output file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from experiments.residual_selection.plan import AUTO_PHGT as V3_AUTO_PHGT
from experiments.residual_selection.plan import BACKBONE_FIXED as V3_BACKBONE_FIXED

ROOT = Path(__file__).resolve().parents[2]
V4 = Path("artifacts/v4_hedge")

# Frozen V2 inputs, read-only: the ACM single-path utilities (diagnostic only) and the V2
# ACM result files that the V2-backbone bridge compares against (reused, never re-run).
V2_SINGLE_PATHS = Path("artifacts/v2/summary/acm_single_paths.csv")
V2_ACM_RESULTS = Path("artifacts/v2/acm/results")

DATASETS = {
    "acm": {"target": "paper",
            "role": "diagnostic / mechanism benchmark; exploratory because its test set was "
                    "already seen in V1/V2"},
    "freebase": {"target": "book", "role": "main clean confirmation benchmark"},
}
LMAX = 4
KS = (2, 5)
METRIC = "test micro-F1 (= accuracy for single-label data); macro-F1 secondary"

# --------------------------------------------------------------------------- backbone
# V4's own validation-only HGT grid. Training uses the HGB-style trainer implemented in
# experiments/residual_selection/train.py (AdamW + OneCycle, early stopping on validation loss); its fixed
# settings are copied into the protocol record so the lock detects any change.
BACKBONE_FIXED = dict(V3_BACKBONE_FIXED)
AUTO_PHGT = dict(V3_AUTO_PHGT)
TUNE_SEEDS = [200, 201]
GRID = {
    "acm": {"feat": ["given"], "layers": [2, 3], "lr": [1e-3, 5e-3], "wd": [1e-4, 1e-3],
            "dropout": [0.2, 0.5]},
    "freebase": {"feat": ["target_onehot"], "layers": [2, 3], "lr": [1e-3, 5e-3],
                 "wd": [0.0, 1e-4], "dropout": [0.2, 0.5]},
}
HGB_REFERENCE = {  # HGB's published HGT settings; both lie inside GRID
    "acm": {"feat": "given", "layers": 2, "lr": 1e-3, "wd": 1e-4, "dropout": 0.2},
    "freebase": {"feat": "target_onehot", "layers": 3, "lr": 1e-3, "wd": 0.0, "dropout": 0.2},
}
PUBLISHED_HGT = {"acm": {"micro_f1": 91.00, "macro_f1": 91.12},
                 "freebase": {"micro_f1": 60.51, "macro_f1": 29.28}}
SELECTION_RULE = ("highest mean validation micro-F1 over the tuning seeds; ties by lower mean "
                  "validation loss. Auto-PHGT runs reuse the chosen HGT configuration.")

# --------------------------------------------------------------------------- Transition-Info
TI = {
    "variants": ["TI_raw", "TI_cov", "TI_norm"],
    "formulas": {
        "TI_raw": "I(p)",
        "TI_cov": "C(p) * I(p)",
        "TI_norm": "C(p) * I(p) / (H(q_p) + eps)",
    },
    "definitions": {
        "P_r": "row-normalised relation adjacency (duplicate edges counted; zero rows stay zero)",
        "T_p": "P_r1 @ ... @ P_rh, computed row-chunk-wise with sparse products (never a dense "
               "N x N matrix); T_p(.|u) is the endpoint distribution of complete uniform random "
               "walks from u (the Auto-PHGT instance sampler), i.e. the row renormalised",
        "self": "the source's own entry T_p(u|u) is removed and the row renormalised (a path "
                "that only returns to its source carries no relational information; same echo "
                "rule as FastPath). Self-inclusive values are recorded as diagnostics",
        "C": "fraction of target nodes with at least one complete instance ending at a node "
             "other than the source (C_any, which also counts self-returns, is recorded)",
        "q": "q_p(v) = mean over covered sources u of T_p(v|u)",
        "I": "mean over covered sources of KL(T_p(.|u) || q_p) = H(q_p) - mean_u H(T_p(.|u)) "
             "(exact over ALL target nodes; nats)",
    },
    "eps": 1e-12,
    "chunk_rows": 512,
    "dense_fraction": 0.1,
    "primary_rule": ("the variant with the highest Spearman correlation with the frozen V2 ACM "
                     "single-path VALIDATION accuracy (mean over V2 seeds 0-2); test utilities "
                     "are never used for this choice; ties in the order of tie_order"),
    "tie_order": ["TI_cov", "TI_norm", "TI_raw"],
}
TI_SET = {
    "fingerprint": ("F_p = rows (T_p(.|u) - q_p) of every covered source u (zero rows for "
                    "uncovered sources) projected by a fixed Gaussian matrix R [N_target x d] / "
                    "sqrt(d)"),
    "fingerprint_dim": 64, "fingerprint_seed": 0,
    "novelty": "novelty(p|S) = 1 - max_{s in S} max(0, cos(F_p, F_s))",
    "score": "first path = argmax TI(p); then argmax TI(p) * novelty(p|S)",
}
BEAM = {
    "width": 20, "lmax": 6, "min_hops": 2, "sample_sources": 1024, "sample_seed": 0,
    "min_coverage": 1e-3, "min_effective_endpoints": 2.0, "duplicate_cosine": 0.9999,
    "fingerprint_dim": 32, "rescore_top": 40,
    "rule": ("depth-wise: extend surviving schema-valid prefixes by every outgoing relation; "
             "statistics on a fixed sample of source nodes; discard prefixes that cannot reach "
             "the target type within the remaining hops (schema reachability), coverage < "
             "min_coverage, endpoint collapse (exp H(q) < min_effective_endpoints) and "
             "duplicates (fingerprint cosine > duplicate_cosine with a better prefix of the same "
             "end type); keep the top `width` prefixes by the primary TI variant; every "
             "generated target-ending path with >= 2 hops that survives the filters is a "
             "discovered candidate; the `rescore_top` best discovered paths are re-scored with "
             "exact TI (primary variant) over all target nodes and the top-k is selected"),
}

# --------------------------------------------------------------------------- FastPath
FASTPATH = {
    "folds": 3, "fold_seed_offset": 4000,
    "operator": ("T_p as in Transition-Info, restricted to official training rows/columns; the "
                 "diagonal (self-return) is removed"),
    "cross_fitting": ("stratified 3-fold split of the run's TRAINING nodes (seeded); for fold j "
                      "the known labels are TRAIN minus fold j, fold-j labels are completely "
                      "masked; every node's evidence m_p(u) = sum_{v known, v != u} T_p(v|u) "
                      "onehot(y_v); L_p(u) = m_p(u) / sum(m_p(u)), or the fold's known-label "
                      "class prior if the path reaches no known label"),
    "score": ("Q_FP(p) = -mean_u CE(y_u, (1 - eps) L_p(u) + eps * prior_j) over out-of-fold "
              "training nodes"),
    "smoothing_eps": 0.1,
    "set_classifier": {"model": "multinomial logistic regression, sklearn lbfgs", "C": 1.0,
                       "max_iter": 1000,
                       "features": ("per path: fold-specific L_p (C values) and an evidence "
                                    "indicator; columns standardised on the fitting rows")},
    "set_rule": ("greedy: Gain(p|S) = CE(S) - CE(S+p), CE = held-out cross-entropy of the probe "
                 "fitted on the known rows of fold j and evaluated on fold j, averaged over folds "
                 "(weights = fold sizes); CE({}) = the known-label class prior; add argmax Gain "
                 "until k=5 (k=2 is the first two picks); no GNN is trained"),
    "tie_break": "canonical structural rank",
    "labels": "official TRAINING labels of the run's split only; validation/test labels never",
}

# --------------------------------------------------------------------------- seeds / runs
SEEDS = {"acm": [0, 1, 2, 3, 4], "freebase": [0, 1, 2, 3, 4]}
FREEBASE_EXTENSION = {
    "seeds": [5, 6, 7, 8, 9], "sd_threshold": 0.01, "top_v4": 2,
    "v4_pool": ["ti_k5", "fastpath_k5", "fastpath_set_k5"],
    "comparators": ["hgt", "canonical_k5", "random_k5", "hybrid_k5"],
    "rule": ("P1, after every Freebase P0 condition finished seeds 0-4: extend to seeds 5-9 if "
             "the mean over Freebase P0 conditions of the sample SD of VALIDATION micro-F1 is >= "
             "1 point; extended = the 2 V4 methods of v4_pool with the highest mean VALIDATION "
             "micro-F1 plus the comparators (needed for paired tests); test results are never "
             "read"),
}
V2_BRIDGE = {
    "seeds": list(range(10)), "selectors": ["ti", "fastpath", "fastpath_set"], "ks": [5, 2],
    "protocol": ("the unchanged V1/V2 ACM protocol (experiments.run_acm.run, "
                 "experiments/protocol.py ACM_HYPERPARAMETERS, V2 split with 180 validation "
                 "papers); V4 selections on each seed's V2 training split; compared with the "
                 "frozen V2 result files, which are reused and not re-run"),
}
DIAGNOSTIC = {
    "v2_seeds": [0, 1, 2],
    "v2_benchmarks": {"structural_rho": -0.27, "homophily_rho": 0.37},
    "named_paths": ["P-A-P", "P-A-P-cite-P", "P-T-P", "P-T-P-T-P"],
    "fastpath_split": "the V2 training splits of seeds 0-2 (the splits of the V2 utilities); "
                      "per-path scores averaged over the three splits, like V2's homophily",
    "utility": "V2 mean test accuracy over seeds 0-2 (post-hoc diagnostic only)",
    "decision_utility": "V2 mean validation accuracy over seeds 0-2 (TI variant choice only)",
}
STATS = {"bootstrap": 10000, "seed": 0, "ci": 0.95, "noninferiority_margin": 0.01,
         "paired": ("difference over matched seeds; mean, sample SD, paired bootstrap 95% CI, "
                    "exact sign-flip permutation p (two-sided), paired t p; with n=5 p-values "
                    "are descriptive"),
         "correlation": "Spearman rho; differences via a paired bootstrap over the 88 paths"}

SELECTORS = {
    "canonical": "V1 structural-volume top-k (unchanged)",
    "random": "k distinct candidates, random.Random(seed) (V2 semantics)",
    "hybrid": "V2: structural score x training-label homophily lift (no relation rules)",
    "diverse": "V2: label-free relation-family coverage (ACM only)",
    "ti": "Transition-Info top-k (primary variant)",
    "ti_set": "Transition-Info greedy with fingerprint novelty (primary variant)",
    "fastpath": "FastPath top-k by Q_FP",
    "fastpath_set": "FastPath-Set greedy conditional gain",
    "ti_beam": "TI-Beam discovery up to 6 hops, top-k by exact TI (primary variant)",
    "ti_raw": "explicit TI_raw top-k", "ti_cov": "explicit TI_cov top-k",
    "ti_norm": "explicit TI_norm top-k",
}
V4_METHODS = ("ti", "ti_set", "fastpath", "fastpath_set", "ti_beam")


def condition(ds, name, mode, *, selector=None, k=None, lmax=LMAX, token_control=None,
              priority=0, group, gate=None):
    return {"id": f"{ds}__{name}", "dataset": ds, "name": name, "mode": mode,
            "selector": selector, "k": k, "lmax": lmax, "token_control": token_control,
            "priority": priority, "group": group, "gate": gate}


def conditions() -> list[dict]:
    """Strong-backbone final-run conditions (each run for every seed of its dataset)."""
    out = []
    for ds in DATASETS:
        out.append(condition(ds, "hgt", "hgt", group="backbone"))
        base = ["canonical", "random", "hybrid"] + (["diverse"] if ds == "acm" else [])
        for sel in base + ["ti", "fastpath", "fastpath_set"]:
            out.append(condition(ds, f"{sel}_k5", "auto_phgt", selector=sel, k=5,
                                 group="selectors_k5"))
        out.append(condition(ds, "ti_set_k5", "auto_phgt", selector="ti_set", k=5, priority=1,
                             group="selectors_k5"))
        for sel in ["canonical", "random", "hybrid", "ti", "ti_set", "fastpath", "fastpath_set"]:
            out.append(condition(ds, f"{sel}_k2", "auto_phgt", selector=sel, k=2, priority=1,
                                 group="selectors_k2"))
        out.append(condition(ds, "ti_beam_k5", "auto_phgt", selector="ti_beam", k=5,
                             lmax=BEAM["lmax"], priority=2, group="ti_beam"))
        out.append(condition(ds, "dummy_k5", "auto_phgt", selector="canonical", k=5,
                             token_control="dummy", priority=2, group="controls"))
        for variant in ("ti_raw", "ti_cov", "ti_norm"):
            out.append(condition(ds, f"{variant}_k5", "auto_phgt", selector=variant, k=5,
                                 priority=2, group="ti_variants", gate="not_primary_variant"))
    return out


def bridge_conditions() -> list[dict]:
    return [condition("acm", f"v2b_{sel}_k{k}", "auto_phgt", selector=sel, k=k, priority=1,
                      group="v2_backbone_bridge")
            for k in V2_BRIDGE["ks"] for sel in V2_BRIDGE["selectors"]]


HYPOTHESES = {
    "H1": "Empirical Transition-Info correlates more positively with downstream path utility "
          "than V1 structural volume on ACM.",
    "H2": "FastPath correlates more strongly with downstream path utility than raw homophily "
          "on ACM.",
    "H3": "FastPath-Set outperforms independent FastPath top-k selection, showing that "
          "complementary path sets matter.",
    "H4": "At least one lightweight V4 selector outperforms the canonical V1 selector on "
          "Freebase.",
    "H5": "At least one lightweight V4 selector is competitive with or better than "
          "hybrid/random while requiring negligible selector-training GPU cost.",
    "H6": "Selector behavior differs across datasets, supporting dataset-adaptive path "
          "discovery rather than a universal fixed structural heuristic.",
}
DECISION_RULES = {
    "H1": ("rho(TI primary, V2 test utility) - rho(structural, V2 test utility) over the 88 ACM "
           "paths; supported if its paired path-bootstrap 95% CI lies above 0, contradicted if "
           "below 0, otherwise inconclusive (all three variants are reported)"),
    "H2": ("rho(Q_FP, V2 test utility) - rho(homophily, V2 test utility); same bootstrap rule"),
    "H3": ("fastpath_set_k5 - fastpath_k5 (test micro-F1, matched seeds) on ACM and Freebase: "
           "supported if the CI lies above 0 on at least one dataset and below 0 on none; "
           "contradicted if below 0 on at least one and above 0 on none; otherwise inconclusive. "
           "A dataset where both selectors pick identical sets for every seed is uninformative. "
           "k=2 is reported as secondary evidence"),
    "H4": ("for each of ti_k5, ti_set_k5, fastpath_k5, fastpath_set_k5 on Freebase: difference "
           "to canonical_k5; supported if at least one CI lies above 0 (Holm-adjusted "
           "permutation p reported descriptively)"),
    "H5": ("Freebase k=5: supported if at least one V4 selector (ti, ti_set, fastpath, "
           "fastpath_set) is non-inferior to BOTH hybrid_k5 and random_k5 (CI lower bound > "
           "-1 point) with zero selector GPU seconds; 'better' if both CIs lie above 0; ACM and "
           "the V2-backbone bridge are reported as secondary"),
    "H6": ("supported if some pair of k=5 selectors common to both datasets has paired CIs "
           "excluding 0 in opposite directions on ACM and Freebase; otherwise not supported; "
           "per-dataset rankings and Kendall tau are reported descriptively"),
    "significance": "a CI 'lies above 0' when its lower bound is > 0 (95% paired bootstrap)",
}
OUTCOMES = {
    "A": "Transition-Info works (H1 supported and ti_k5 or ti_set_k5 beats canonical_k5 on "
         "Freebase): V1 used the wrong statistical approximation.",
    "B": "FastPath works but TI does not (fastpath_k5 or fastpath_set_k5 beats canonical_k5 on "
         "Freebase and A does not hold): task awareness is required, expensive neural search "
         "is not.",
    "C": "FastPath-Set > FastPath (H3 supported): path complementarity is important.",
    "D": "V3 later wins strongly: V4 becomes the lightweight baseline / efficiency comparison "
         "(needs V3 results; not decided inside V4).",
    "E": "V3 fails and V4 works: V4 becomes the main selector contribution (needs V3 results; "
         "not decided inside V4).",
    "F": "Both fail (no V4 selector beats canonical_k5 on Freebase and neither H1 nor H2 is "
         "supported): path selection needs a richer approach.",
}
PRIORITIES = {
    "P0": "protocol freeze, ACM 88-path diagnostics, Transition-Info, FastPath, FastPath-Set, "
          "Freebase loader verification, strong HGT reference, ACM and Freebase k=5 selector "
          "runs, summary/statistics",
    "P1": "TI set-aware, k=2, V2-backbone bridge, Freebase 10-seed extension, efficiency",
    "P2": "TI-Beam (Lmax=6) final runs, dummy-token control, explicit non-primary TI variants",
    "not_run": "DBLP and ogbn-mag are not part of V4 (the directive allows them only after "
               "everything else; V4 stays lightweight)",
}
TEST_POLICY = ("training folds for selector construction; validation for model/hyperparameter "
               "and seed-extension decisions; each final run evaluates test exactly once after "
               "its selection is frozen, logged in logs/test_evaluations.jsonl; V2 ACM test "
               "utilities are used only for post-hoc diagnostic correlations; no V3 file is read")


def protocol_record() -> dict:
    return {"datasets": DATASETS, "lmax": LMAX, "ks": KS, "metric": METRIC,
            "backbone_fixed": BACKBONE_FIXED, "auto_phgt": AUTO_PHGT, "tune_seeds": TUNE_SEEDS,
            "grid": GRID, "hgb_reference": HGB_REFERENCE, "published_hgt": PUBLISHED_HGT,
            "selection_rule": SELECTION_RULE, "ti": TI, "ti_set": TI_SET, "beam": BEAM,
            "fastpath": FASTPATH, "seeds": SEEDS, "freebase_extension": FREEBASE_EXTENSION,
            "v2_bridge": V2_BRIDGE, "diagnostic": DIAGNOSTIC, "stats": STATS,
            "selectors": SELECTORS, "conditions": conditions(),
            "bridge_conditions": bridge_conditions(), "hypotheses": HYPOTHESES,
            "decision_rules": DECISION_RULES, "outcomes": OUTCOMES, "priorities": PRIORITIES,
            "test_policy": TEST_POLICY, "v2_inputs": [str(V2_SINGLE_PATHS), str(V2_ACM_RESULTS)]}


def protocol_hash() -> str:
    blob = json.dumps(protocol_record(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


if os.environ.get("V4_SMOKE") == "1":  # tiny settings for the pre-launch smoke test only
    V4 = Path("artifacts/v4_hedge_smoke")
    TUNE_SEEDS = [200]
    GRID = {"acm": {"feat": ["given"], "layers": [2], "lr": [1e-3], "wd": [1e-4],
                    "dropout": [0.2]},
            "freebase": {"feat": ["target_onehot"], "layers": [3], "lr": [1e-3], "wd": [0.0],
                         "dropout": [0.2]}}
    BACKBONE_FIXED = {**BACKBONE_FIXED, "max_epochs": 4, "patience": 2}
    SEEDS = {"acm": [0], "freebase": [0]}
    FREEBASE_EXTENSION = {**FREEBASE_EXTENSION, "seeds": [5], "sd_threshold": 0.0}
    V2_BRIDGE = {**V2_BRIDGE, "seeds": [0]}
    BEAM = {**BEAM, "width": 4, "lmax": 5, "sample_sources": 256, "rescore_top": 6}
    STATS = {**STATS, "bootstrap": 500}
