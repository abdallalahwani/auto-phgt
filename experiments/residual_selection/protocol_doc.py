"""Writes artifacts/v3/protocol/V3_PROTOCOL.md at freeze time (before any V3 run)."""

from __future__ import annotations

import json

from .plan import (AUTO_PHGT, BACKBONE_FIXED, DATASETS, FEATURE_REGIMES, FREEBASE_EXTENSION, GRID,
                   HYPOTHESES, MAG, MANUAL_PATHS, RCMS, SEEDS, TUNE_SEEDS, V3, hgb_conditions)

RULES = {
    "H1": "Supported if the V3 strong-HGT ACM test micro-F1 gap to published HGB HGT (91.00) is at "
          "most half of the V2 gap (91.00 - 86.44 = 4.56 points), i.e. strong HGT >= 88.72.",
    "H2": "Paired RCMS - canonical on each HGB dataset's primary setting (ACM k=5 L<=4; DBLP k=5 "
          "L<=6; Freebase k=5 L<=4). Supported if supported on >= 2 datasets and contradicted on none.",
    "H3": "Paired RCMS - hybrid on the clean datasets (DBLP, Freebase). Supported if supported on >= 1 "
          "and contradicted on none. ACM is reported but exploratory.",
    "H4": "Paired RCMS-Auto-PHGT - strong HGT on the three HGB datasets. Supported if supported on >= 2 "
          "and contradicted on none. MAG reported separately.",
    "H5": "Paired full RCMS - independent residual ranking (primary settings). Supported if supported on "
          ">= 1 dataset and contradicted on none.",
    "H6": "Paired full RCMS - conditional selection without HGT (primary settings). Same rule as H5.",
    "H7": "Paired RCMS real tokens - dummy tokens (ACM, DBLP) and - shuffled tokens (ACM). Supported if "
          "supported on >= 2 of these comparisons and contradicted on none.",
    "H8": "Requires the search cost of a task-driven competitor measured under our protocol. LMSPS "
          "search is not reproduced in V3 (see limitations), so H8 is reported as not testable; RCMS "
          "search cost and Auto-PHGT with LMSPS-published DBLP paths are reported descriptively.",
}


def write_protocol_doc(lock: dict, tasks: list[dict]) -> None:
    conds = hgb_conditions()
    counts = {}
    for t in tasks:
        key = (t["fn"], t["priority"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# V3 protocol (frozen before any V3 run)", "",
        f"- frozen at: {lock['frozen_at']} on {lock['host']}",
        f"- git: {lock['git']}", f"- protocol hash: `{lock['protocol_hash']}`; task graph "
        f"`{lock['task_graph_sha256']}`; code hashes in protocol_lock.json", "",
        "V1 and V2 code and outputs are not modified; V3 lives in `experiments/residual_selection/`, "
        "`experiments/residual_campaign.py`, `slurm/residual_campaign.sbatch`, and `artifacts/v3/`.", "",
        "## 1. Goal", "",
        "Test whether Residual-Conditional Meta-Path Selection (RCMS) defines meta-path usefulness "
        "better than the V1 structural-volume score, inside the unchanged Auto-PHGT pipeline "
        "(candidate enumeration -> selection -> instance sampling -> semantic tokens -> HGT -> "
        "attention fusion -> classifier), on a strong, validation-tuned HGB-style HGT backbone.", "",
        "## 2. Datasets and splits", "",
        "HGB ACM, DBLP, Freebase via PyG `HGBDataset` (official train/test files). Validation is a "
        "seeded random 20% of the official training nodes (HGB utils/data.py uses int(0.2 n)). "
        "Metrics: micro-F1 (= accuracy, single-label) and macro-F1 on the official test nodes. "
        "OGBN-MAG uses the official OGB split and the V1/V2 loader and model configuration.", "",
        "| dataset | target | classes | official train | official test |", "|---|---|---|---|---|",
        "| ACM | paper | 3 | 907 | 2118 |", "| DBLP | author | 4 | 1217 | 2840 |",
        "| Freebase | book | 7 | 2386 | 5568 |", "", "ACM is exploratory (its test set was inspected in "
        "V1/V2). DBLP and Freebase are the clean confirmation datasets.", "",
        "## 3. Strong HGT backbone (step 0)", "",
        f"Fixed: {json.dumps(BACKBONE_FIXED)}. Validation-only grid per dataset: feature regimes "
        f"{json.dumps(FEATURE_REGIMES)} (HGB feats-type 0/1/2 and the V1 imputation), layers "
        f"{GRID['layers']}, OneCycle max lr {GRID['lr']}, weight decay {json.dumps(GRID['wd'])}, dropout "
        f"{GRID['dropout']}, tuning seeds {TUNE_SEEDS} (disjoint from final seeds). The grid contains "
        "each dataset's HGB reference configuration. Selection: highest mean validation micro-F1 "
        "(ties: lower validation loss). The test split is never evaluated during tuning. "
        f"Reproduction target ACM ~91 micro-F1; below 90.5 -> BACKBONE_REPRODUCTION_WARNING (the "
        "campaign continues with the validation-selected configuration). Auto-PHGT variants use the "
        f"same backbone configuration and training procedure; fusion/tokenizer settings stay fixed: "
        f"{json.dumps(AUTO_PHGT)}.", "",
        "## 4. RCMS", "",
        "- Candidates: every schema-valid target-to-target path with 2..L hops (L = 4 primary, 6 "
        "extended), in canonical structural-score order. Screening removes only candidates whose "
        f"completion rate on {RCMS['coverage_probe_nodes']} training nodes is below "
        f"{RCMS['min_coverage']}; every removal is recorded.",
        "- H (backbone): 3-fold stratified cross-fitting inside the seed's training nodes. For fold j an "
        "HGT with the chosen configuration is trained on the other folds (official validation only for "
        "early stopping) and predicts fold j; H is the out-of-fold class log-probability vector "
        "(aligned across fold models, unlike raw embeddings).",
        f"- Z_p (path): label-free. Per node type a fixed d={RCMS['d_z']} vector mirrors the input regime "
        "(PCA of dense features; zero for HGB feat-1 zero inputs; a fixed random Gaussian vector, i.e. a "
        "random projection of the one-hot input, for ID-only types). For each target, Z_p is the mean "
        f"over {RCMS['search_instances']} sampled complete instances of the mean vector of the path "
        "positions after the target, plus the completion rate (d+1 dims). All blocks are standardised "
        "over nodes (label-free statistics).",
        f"- Probe: multinomial logistic regression with identical capacity for every candidate at a step, "
        f"L2 {RCMS['probe_l2']}, Adam lr {RCMS['probe_lr']}, {RCMS['probe_steps']} full-batch steps from "
        "zero; held-out CE on each cross-fitting fold (fit on the other two folds), averaged over folds.",
        "- U(p | H, S) = CE(H, Z_S) - CE(H, Z_S, Z_p). Greedy: S = {}; add argmax U until |S| = k "
        f"(k in {{2, 5}}; nested). Step 1 evaluates every candidate; later steps evaluate the "
        f"{RCMS['pool_after_step1']} best step-1 candidates (a documented approximation that only binds "
        "when more candidates exist). Ties break by canonical rank.",
        "- Ablations: independent residual ranking (top-k of U(p | H, {})); conditional without HGT "
        "(greedy on U(p | S)); full RCMS. One search per (dataset, seed, L); test never read.",
        "- MAG: a pre-declared internal 80/20 stratified holdout of the official training set replaces "
        f"3-fold HGT cross-fitting; the search HGT uses the V1 configuration for at most "
        f"{MAG['search_epochs']} epochs; utilities use a {MAG['search_subset']}-paper stratified subset "
        "of the holdout with 3-fold probe cross-validation; one search (seed 0) serves all MAG seeds.",
        "- Analysis only (never used for selection): residual linear CKA of HGT-residualised path "
        "representations within each selector's set; training-label homophily of every candidate.", "",
        "## 5. Conditions", "",
        "| condition | mode | selector | k | L | token control | priority | group |",
        "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        lines.append(f"| {c['id']} | {c['mode']} | {c['selector']} | {c['k']} | {c['lmax']} | "
                     f"{c['token_control']} | P{c['priority']} | {c['group']} |")
    lines += [
        "", "MAG: strong/current HGT, canonical, random, dummy tokens, RCMS (k=5), 3 seeds, "
        f"{MAG['epochs']}-epoch cap with validation early stopping (patience 20). Continuation to "
        f"{MAG['continue_to']} epochs (P1) only if a run reaches the cap and its best validation accuracy "
        f"improved by >= {MAG['continue_min_gain']} over the last {MAG['continue_window']} epochs, and "
        "the remaining walltime suffices. Dummy tokens ignore path content, so their path set only "
        "fixes the token count.", "",
        "Amendment (fixed before any V3 result): DBLP has only 4 target-to-target candidates within 4 "
        "hops, so k=5 is infeasible there; every selector would return the same set. DBLP's selector "
        "comparison therefore uses k=2 at L<=4 and k=5 at L<=6 (13 candidates); `all4_L4` (all 4 "
        "candidates) is reported once.", "",
        f"Manual/expert paths: {json.dumps(MANUAL_PATHS)} (exploratory). Parameter-matched HGT: the "
        "HGT width (multiple of 8) whose parameter count is closest to the main Auto-PHGT model "
        "(ACM, DBLP; Freebase is dominated by ID embeddings).", "",
        "## 6. External comparisons", "",
        "Published HGB, LMSPS, SeHGNN, HINormer and PHGT numbers are included and labelled "
        "PUBLISHED RESULT — NOT OUR RUN; no paired statistics are computed against them. Our own run: "
        "Auto-PHGT with the meta-paths LMSPS published for DBLP (official repository, commit in "
        "plan.py), which isolates path quality from our architecture. Not reproduced in V3: official "
        "LMSPS/PHGT training (they require torch 1.x/DGL/CUDA-11 environments and, for PHGT, "
        "pre-generated cluster data; they do not fit the single 24-hour allocation next to the P0 "
        "work), AMP-HGNAS (no code verified). LMSPS's ACM paths use a merged paper-paper relation that "
        "our ACM graph represents as two relations, so they are not mapped.", "",
        "## 7. Hypotheses (verbatim) and decision rules", ""]
    for h, text in HYPOTHESES.items():
        lines += [f"- **{h}**: {text}", f"  - Rule: {RULES[h]}"]
    lines += [
        "", "Paired comparisons use matched seeds; the per-seed difference in test micro-F1 (accuracy on "
        "MAG); mean, sample SD, 95% paired bootstrap CI (10,000 resamples), exact sign-flip "
        "permutation p, and paired t p (descriptive). Supported = CI excludes 0 in the predicted "
        "direction; contradicted = CI excludes 0 in the opposite direction; else inconclusive. "
        "Outcome classes A-D follow the task description (A: RCMS beats hybrid/random and strong HGT; "
        "B: RCMS ~ LMSPS but cheaper (not assessable without LMSPS search); C: selector gains without "
        "beating strong HGT; D: strong HGT removes the Auto-PHGT gain).", "",
        "## 8. Seeds", "",
        f"{json.dumps(SEEDS)}. Freebase extension: {FREEBASE_EXTENSION['rule']} (seeds "
        f"{FREEBASE_EXTENSION['seeds']}).", "",
        "## 9. Orchestration and priorities", "",
        "One Slurm job (`slurm/residual_campaign.sbatch`), no arrays or follow-up jobs. P0 must finish before P1 "
        "starts (P1 tasks start only once every P0 task has started); a task starts only if its "
        "estimate fits the remaining walltime. P0: strong-HGT tuning, all P0 conditions above, RCMS "
        "searches L<=4 (and DBLP L<=6), MAG HGT/canonical/random/dummy/RCMS, aggregation. P1: L<=6 "
        "RCMS on ACM/Freebase, DBLP expert and LMSPS paths, MAG continuation. P2 (published legacy "
        "baselines) needs no compute.", "",
        "Task counts (function, priority): " + ", ".join(f"{k[0]} P{k[1]}: {v}" for k, v in sorted(counts.items())),
        "", "## 10. No test-driven branching", "",
        "Allowed: training loss, internal cross-fitting folds, official validation (tuning, early "
        "stopping, selection), the variance-based Freebase extension (validation SD), and the MAG "
        "continuation rule (validation trend). Every test evaluation is logged to "
        "`artifacts/v3/logs/test_evaluations.jsonl`; no task reads test metrics before aggregation.", "",
        "## 11. Known limitations", "",
        "- PyG HGTConv differs from HGB's DGL HGT (see baseline_audit/HGT_REPRODUCTION.md).",
        "- MAG HGT is not re-tuned (cost); MAG is below leaderboard systems that use extra embeddings.",
        "- Official LMSPS/PHGT runs are not reproduced; H8 is therefore not testable.",
        "- The RCMS probe representation is a label-free proxy of the learned token content.",
        "- ACM results are exploratory because its test set was inspected before V3.", ""]
    path = V3 / "protocol" / "V3_PROTOCOL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
