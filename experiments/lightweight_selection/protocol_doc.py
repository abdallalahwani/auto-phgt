"""Write artifacts/v4_hedge/protocol/V4_HEDGE_PROTOCOL.md from the frozen lock record."""

from __future__ import annotations

import json

from . import plan


def compute_plan(tasks) -> dict:
    """Planned task counts and a GPU-hour estimate from the scheduler's own estimates."""
    gpu = [t for t in tasks if t["size"] != "cpu"]
    by = {}
    for t in gpu:
        key = f"P{t['priority']} {t['meta'].get('dataset')} {t['fn']}"
        by[key] = by.get(key, 0) + 1
    def runs_of(t):
        if t["fn"] == "bridge":
            return len(plan.V2_BRIDGE["seeds"])
        if t["fn"] == "tune":
            g = plan.GRID[t["meta"]["dataset"]]
            return len(plan.TUNE_SEEDS) * len(g["feat"]) * len(g["wd"]) * len(g["dropout"])
        return 1

    runs = sum(runs_of(t) for t in gpu)
    hours = {f"P{p}": round(sum(t["est_minutes"] for t in gpu if t["priority"] == p) / 60, 1)
             for p in (0, 1, 2)}
    return {"gpu_tasks": len(gpu), "cpu_tasks": len(tasks) - len(gpu), "gpu_training_runs": runs,
            "by_kind": by, "estimated_gpu_hours_upper": hours,
            "note": "estimates are conservative per-task upper bounds used by the scheduler"}


def write_protocol_doc(lock: dict, tasks: list[dict]) -> None:
    cp = compute_plan(tasks)
    lines = [
        "# V4-Hedge protocol (frozen before any V4 result)", "",
        "V4-Hedge is an independent, lightweight hedge against V3. It changes only how meta-"
        "paths are discovered/selected; the Auto-PHGT architecture (concrete path instances ->"
        " semantic tokens -> HGT + attention fusion) is unchanged. V4 reuses V3 *code* (HGB "
        "loaders/splits and the HGB-style trainer) and never reads a V3 output.", "",
        "## Lock", "",
        f"- protocol hash: `{lock['protocol_hash']}`",
        f"- git: {lock['git']}; tracked diff sha256 `{lock['git_diff_sha256']}`",
        f"- V3 code vs its launch commit: {lock['v3_code_check']}",
        f"- task graph: {lock['tasks']} tasks, sha256 `{lock['task_graph_sha256']}`",
        f"- versions: {json.dumps(lock['versions'])}",
        f"- frozen at {lock['frozen_at']} on {lock['host']}", "",
        "## Datasets", ""]
    for ds, rec in lock["datasets"].items():
        lines.append(f"- **{ds}** ({plan.DATASETS[ds]['role']}): observed {rec['observed']}; "
                     f"matches expected: {rec['matches_expected']}; train/test disjoint: "
                     f"{rec['train_test_disjoint']}; V4 split sizes {rec['v4_split_sizes']}")
    lines += ["", "Candidate space: all schema-valid target-to-target meta-paths with 2..4 hops "
              "(`StatisticalDiscoveryModule.discover_candidate_paths`, the V1-V3 space; ACM: the "
              "same 88 paths as V2). Freebase is used exactly as loaded by the repository (PyG "
              "HGBDataset, 36 directed relation types).", "",
              "## Backbone", "",
              f"- fixed: {json.dumps(plan.BACKBONE_FIXED)}",
              f"- Auto-PHGT fusion (unchanged): {json.dumps(plan.AUTO_PHGT)}",
              f"- V4's own validation grid: {json.dumps(plan.GRID)}; tuning seeds "
              f"{plan.TUNE_SEEDS}; HGB reference configs {json.dumps(plan.HGB_REFERENCE)} are "
              "inside the grid",
              f"- selection: {plan.SELECTION_RULE}", "",
              "## Transition-Info (label-free)", ""]
    for key, text in plan.TI["definitions"].items():
        lines.append(f"- {key}: {text}")
    for key, text in plan.TI["formulas"].items():
        lines.append(f"- {key}(p) = {text}")
    lines += [f"- eps = {plan.TI['eps']}; primary variant: {plan.TI['primary_rule']} "
              f"(tie order {plan.TI['tie_order']})",
              f"- set-aware TI: {plan.TI_SET['fingerprint']}; {plan.TI_SET['novelty']}; "
              f"{plan.TI_SET['score']}",
              f"- TI-Beam (P2 final runs): {plan.BEAM['rule']}; settings "
              f"{json.dumps({k: v for k, v in plan.BEAM.items() if k != 'rule'})}", "",
              "## FastPath (training labels only, echo-free, cross-fitted)", "",
              f"- operator: {plan.FASTPATH['operator']}",
              f"- cross-fitting: {plan.FASTPATH['cross_fitting']}",
              f"- individual score: {plan.FASTPATH['score']} with eps = "
              f"{plan.FASTPATH['smoothing_eps']}",
              f"- FastPath-Set: {plan.FASTPATH['set_rule']}; probe "
              f"{json.dumps(plan.FASTPATH['set_classifier'])}",
              f"- labels: {plan.FASTPATH['labels']}; ties: {plan.FASTPATH['tie_break']}", "",
              "## Selectors", ""]
    lines += [f"- `{k}`: {v}" for k, v in plan.SELECTORS.items()]
    lines += ["", "## Conditions (strong backbone; seeds " + json.dumps(plan.SEEDS) + ")", "",
              "| id | selector | k | Lmax | control | priority | gate |",
              "|---|---|---|---|---|---|---|"]
    for c in plan.conditions():
        lines.append(f"| {c['id']} | {c['selector'] or '-'} | {c['k'] or '-'} | {c['lmax']} | "
                     f"{c['token_control'] or '-'} | P{c['priority']} | {c['gate'] or '-'} |")
    lines += ["", f"Freebase extension: {plan.FREEBASE_EXTENSION['rule']}", "",
              f"V2-backbone bridge (P1): {plan.V2_BRIDGE['protocol']}; selectors "
              f"{plan.V2_BRIDGE['selectors']}, k {plan.V2_BRIDGE['ks']}, seeds "
              f"{plan.V2_BRIDGE['seeds']}", "",
              "## ACM diagnostic (P0, before model training)", "",
              f"- utilities: {plan.DIAGNOSTIC['utility']}; TI variant choice uses "
              f"{plan.DIAGNOSTIC['decision_utility']}",
              f"- FastPath on {plan.DIAGNOSTIC['fastpath_split']}",
              f"- V2 benchmarks: {plan.DIAGNOSTIC['v2_benchmarks']}; named paths "
              f"{plan.DIAGNOSTIC['named_paths']}", "",
              "## Hypotheses (written before any V4 test result)", ""]
    for key, text in plan.HYPOTHESES.items():
        lines.append(f"- **{key}**: {text}  \n  Decision rule: {plan.DECISION_RULES[key]}")
    lines += [f"- {plan.DECISION_RULES['significance']}", "", "## Outcomes", ""]
    lines += [f"- **{k}**: {v}" for k, v in plan.OUTCOMES.items()]
    lines += ["", "## Statistics and test policy", "", f"- {plan.STATS['paired']}",
              f"- {plan.STATS['correlation']}; bootstrap {plan.STATS['bootstrap']} resamples, "
              f"seed {plan.STATS['seed']}; non-inferiority margin "
              f"{plan.STATS['noninferiority_margin'] * 100:.0f} point",
              f"- metric: {plan.METRIC}", f"- {plan.TEST_POLICY}", "",
              "## Priorities and compute", ""]
    lines += [f"- {k}: {v}" for k, v in plan.PRIORITIES.items()]
    lines += [f"- plan: {json.dumps(cp)}", "",
              "Selector computations are CPU/sparse only (zero GPU seconds); GPU time is the "
              "final HGT/Auto-PHGT training. The master considers tasks in priority order: "
              "P1 only uses resources that no ready P0 task is waiting for, P2 starts only "
              "when no P0/P1 task is pending, and tasks that cannot finish before the "
              "deadline are not started.", ""]
    path = plan.V4 / "protocol" / "V4_HEDGE_PROTOCOL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
