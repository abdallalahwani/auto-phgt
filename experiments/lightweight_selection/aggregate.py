"""V4-Hedge aggregation: tables, paired statistics, hypothesis verdicts, plots and claims.

    python -m experiments.lightweight_selection.aggregate

Reads only V4 outputs and frozen V2 files (never a V3 file). Safe to run on partial results.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from . import plan
from .plan import V4
from .stats import holm, kendall_tau, paired, verdict

SUMMARY, PLOTS = V4 / "summary", V4 / "plots"
V4_SELECTORS = ["ti", "ti_set", "fastpath", "fastpath_set"]
BASELINES = ["canonical", "random", "hybrid", "diverse"]
V2_NAMES = {"canonical": "discovered", "random": "random", "diverse": "diverse",
            "homophily": "homophily", "hybrid": "hybrid"}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def pct(x):
    return None if x is None else 100.0 * x


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)


def fmt(m, sd=None, digits=2):
    if m is None:
        return "-"
    return f"{m:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "")


# --------------------------------------------------------------------------- loading
def load_strong() -> dict:
    """{(ds, name): {seed: record}} for completed strong-backbone runs."""
    out = {}
    for ds in plan.DATASETS:
        for path in sorted((V4 / "results" / ds).glob("*.json")):
            rec = _read(path)
            if rec and rec.get("status") == "completed":
                out.setdefault((ds, rec["name"]), {})[rec["seed"]] = rec
    return out


def load_bridge() -> dict:
    out = {}
    for path in sorted((V4 / "results" / "acm_v2backbone").glob("*.json")):
        rec = _read(path)
        if rec and rec.get("status") == "completed":
            name = rec["condition"].split("__", 1)[1].replace("v2b_", "")
            out.setdefault(name, {})[rec["seed"]] = rec
    return out


def load_v2_frozen() -> dict:
    """Frozen V2 ACM results (reused, never re-run): {name_k: {seed: (test acc, val acc)}}."""
    out = {}
    for name, v2 in V2_NAMES.items():
        for k in plan.KS:
            for seed in range(10):
                rec = _read(plan.V2_ACM_RESULTS / f"v2_acm_auto_phgt_seed{seed}_{v2}_k{k}.json")
                if rec and rec.get("status") == "completed":
                    out.setdefault(f"{name}_k{k}", {})[seed] = (rec["test"]["accuracy"],
                                                                rec["validation"]["accuracy"])
    for seed in range(10):
        rec = _read(plan.V2_ACM_RESULTS / f"v2_acm_hgt_seed{seed}_discovered_k5.json")
        if rec and rec.get("status") == "completed":
            out.setdefault("hgt", {})[seed] = (rec["test"]["accuracy"],
                                               rec["validation"]["accuracy"])
    return out


def by_seed(runs: dict, split="test", kind="micro_f1") -> dict:
    return {s: r[split][kind] for s, r in runs.items()}


def bridge_by_seed(runs: dict, split="test") -> dict:
    key = "test" if split == "test" else "validation"
    return {s: r[key]["accuracy"] for s, r in runs.items()}


# --------------------------------------------------------------------------- tables
def main_rows(strong) -> list[dict]:
    rows = []
    for cond in plan.conditions():
        runs = strong.get((cond["dataset"], cond["name"]), {})
        planned = len(plan.SEEDS[cond["dataset"]])
        row = {"dataset": cond["dataset"], "condition": cond["name"], "selector": cond["selector"],
               "k": cond["k"], "lmax": cond["lmax"], "control": cond["token_control"],
               "group": cond["group"], "priority": f"P{cond['priority']}",
               "seeds_planned_min": planned, "seeds_completed": len(runs),
               "seeds": " ".join(str(s) for s in sorted(runs))}
        for split, key in (("test", "test"), ("val", "validation")):
            for kind in ("micro_f1", "macro_f1"):
                m, sd = mean_sd([pct(r[key][kind]) for r in runs.values()])
                row[f"{split}_{kind}_mean"], row[f"{split}_{kind}_sd"] = m, sd
        row["parameters"] = mean_sd([r["parameters"] for r in runs.values()])[0]
        row["train_s_mean"] = mean_sd([r["fit"]["train_runtime_s"] for r in runs.values()])[0]
        row["best_epoch_mean"] = mean_sd([r["fit"]["best_epoch"] for r in runs.values()])[0]
        row["gpu_peak_mib_mean"] = mean_sd([(r["fit"].get("gpu_memory") or {}).get(
            "max_allocated_mib") for r in runs.values()])[0]
        row["selection_s_mean"] = mean_sd([r["selection_runtime_s"] for r in runs.values()])[0]
        rows.append(row)
    return rows


def bridge_rows(bridge, v2) -> list[dict]:
    rows = []
    for name, runs in sorted(v2.items()):
        tm, tsd = mean_sd([pct(t) for t, _ in runs.values()])
        vm, vsd = mean_sd([pct(v) for _, v in runs.values()])
        rows.append({"backbone": "v2 (frozen V2 results, reused)", "dataset": "acm",
                     "condition": name, "seeds_completed": len(runs), "test_micro_f1_mean": tm,
                     "test_micro_f1_sd": tsd, "val_micro_f1_mean": vm, "val_micro_f1_sd": vsd})
    for name, runs in sorted(bridge.items()):
        tm, tsd = mean_sd([pct(r["test"]["accuracy"]) for r in runs.values()])
        mm, msd = mean_sd([pct(r["test"]["macro_f1"]) for r in runs.values()])
        vm, vsd = mean_sd([pct(r["validation"]["accuracy"]) for r in runs.values()])
        rows.append({"backbone": "v2 (new V4 runs)", "dataset": "acm", "condition": name,
                     "seeds_completed": len(runs), "test_micro_f1_mean": tm,
                     "test_micro_f1_sd": tsd, "test_macro_f1_mean": mm, "test_macro_f1_sd": msd,
                     "val_micro_f1_mean": vm, "val_micro_f1_sd": vsd})
    return rows


def comparisons(strong, bridge, v2) -> list[dict]:
    rows = []

    def add(scope, ds, a_name, b_name, a, b, **extra):
        res = paired(a, b, plan.STATS["bootstrap"], plan.STATS["seed"])
        rows.append({"scope": scope, "dataset": ds, "a": a_name, "b": b_name, **extra,
                     "n": res["n"], "mean_diff_pts": res["mean"], "sd_pts": res["sd"],
                     "ci95_low": res["ci95"][0] if res["ci95"] else None,
                     "ci95_high": res["ci95"][1] if res["ci95"] else None,
                     "verdict": verdict(res["ci95"]), "p_perm": res["p_perm"], "p_t": res["p_t"],
                     "wins": res["wins"], "losses": res["losses"],
                     "per_seed_pts": " ".join(f"{d:.2f}" for d in res["per_seed"])})

    for ds in plan.DATASETS:
        get = lambda name: by_seed(strong.get((ds, name), {}))  # noqa: E731
        for k in plan.KS:
            for a in V4_SELECTORS + ["ti_beam", "ti_raw", "ti_cov", "ti_norm"]:
                an = f"{a}_k{k}"
                if (ds, an) not in strong:
                    continue
                for b in BASELINES + ["hgt"]:
                    bn = "hgt" if b == "hgt" else f"{b}_k{k}"
                    if (ds, bn) in strong:
                        add("strong", ds, an, bn, get(an), get(bn), k=k)
            for a, b in (("fastpath_set", "fastpath"), ("ti_set", "ti"),
                         ("fastpath", "ti"), ("fastpath_set", "ti_set")):
                an, bn = f"{a}_k{k}", f"{b}_k{k}"
                if (ds, an) in strong and (ds, bn) in strong:
                    add("strong", ds, an, bn, get(an), get(bn), k=k)
            for b in BASELINES:
                bn = f"{b}_k{k}"
                if (ds, bn) in strong and (ds, "hgt") in strong:
                    add("strong", ds, bn, "hgt", get(bn), get("hgt"), k=k)
        if (ds, "dummy_k5") in strong and (ds, "canonical_k5") in strong:
            add("strong", ds, "canonical_k5", "dummy_k5", get("canonical_k5"), get("dummy_k5"),
                k=5)
        if (ds, "ti_beam_k5") in strong and (ds, "ti_k5") in strong:
            add("strong", ds, "ti_beam_k5", "ti_k5", get("ti_beam_k5"), get("ti_k5"), k=5)
    for name, runs in bridge.items():
        k = int(name.rsplit("_k", 1)[1])
        a = bridge_by_seed(runs)
        for b in list(V2_NAMES) + ["hgt"]:
            bn = "hgt" if b == "hgt" else f"{b}_k{k}"
            if bn in v2:
                add("v2_backbone", "acm", f"v2b_{name}", f"v2_{bn}", a,
                    {s: t for s, (t, _) in v2[bn].items()}, k=k)
    return rows


def find(rows, ds, a, b, scope="strong"):
    return next((r for r in rows if r["scope"] == scope and r["dataset"] == ds and r["a"] == a
                 and r["b"] == b), None)


def identical_sets(strong, ds, a, b) -> bool | None:
    ra, rb = strong.get((ds, a), {}), strong.get((ds, b), {})
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        return None
    return all(ra[s]["templates"] == rb[s]["templates"] for s in seeds)


# --------------------------------------------------------------------------- hypotheses
def hypotheses(strong, stats_rows, diag) -> dict:
    out = {}
    primary = (_read(V4 / "decisions" / "ti_primary.json") or {}).get("primary")
    if diag and primary:
        d1 = diag["differences"].get(f"test_utility:{primary}-structural")
        out["H1"] = {"verdict": _h(d1["ci95"] if d1 else None), "primary_variant": primary,
                     "rho_ti": diag["test_utility"][primary]["rho"],
                     "rho_structural": diag["test_utility"]["structural"]["rho"],
                     "difference": d1,
                     "all_variants": {v: diag["differences"].get(f"test_utility:{v}-structural")
                                      for v in plan.TI["variants"]}}
        d2 = diag["differences"].get("test_utility:FastPath-homophily")
        out["H2"] = {"verdict": _h(d2["ci95"] if d2 else None),
                     "rho_fastpath": diag["test_utility"]["FastPath"]["rho"],
                     "rho_homophily": diag["test_utility"]["homophily"]["rho"], "difference": d2}
    else:
        out["H1"] = out["H2"] = {"verdict": "pending"}
    per = {}
    for ds in plan.DATASETS:
        row = find(stats_rows, ds, "fastpath_set_k5", "fastpath_k5")
        same = identical_sets(strong, ds, "fastpath_set_k5", "fastpath_k5")
        per[ds] = {"verdict": "uninformative (identical sets)" if same else
                   (row["verdict"] if row else "pending"), "row": row}
    verdicts = [v["verdict"] for v in per.values()]
    if "pending" in verdicts:
        h3 = "pending"
    elif "above" in verdicts and "below" not in verdicts:
        h3 = "supported"
    elif "below" in verdicts and "above" not in verdicts:
        h3 = "contradicted"
    else:
        h3 = "inconclusive"
    out["H3"] = {"verdict": h3, "per_dataset": per,
                 "k2": {ds: find(stats_rows, ds, "fastpath_set_k2", "fastpath_k2")
                        for ds in plan.DATASETS}}
    h4 = {m: find(stats_rows, "freebase", f"{m}_k5", "canonical_k5") for m in V4_SELECTORS}
    done = {m: r for m, r in h4.items() if r}
    p_holm = holm({m: r["p_perm"] for m, r in done.items()})
    out["H4"] = {"verdict": ("pending" if not done else
                             "supported" if any(r["verdict"] == "above" for r in done.values())
                             else "not supported"),
                 "comparisons": done, "holm_p_perm": p_holm}
    margin = -plan.STATS["noninferiority_margin"] * 100
    h5 = {}
    for m in V4_SELECTORS:
        vs = {b: find(stats_rows, "freebase", f"{m}_k5", f"{b}_k5") for b in ("hybrid", "random")}
        if all(vs.values()):
            noninf = all(r["ci95_low"] is not None and r["ci95_low"] > margin for r in vs.values())
            better = all(r["verdict"] == "above" for r in vs.values())
            h5[m] = {"noninferior_to_both": noninf, "better_than_both": better,
                     "vs": vs, "selector_gpu_s": 0.0}
    out["H5"] = {"verdict": ("pending" if not h5 else
                             "supported" if any(v["noninferior_to_both"] for v in h5.values())
                             else "not supported"),
                 "margin_pts": margin, "per_selector": h5}
    common = [n for n in ["canonical_k5", "random_k5", "hybrid_k5", "ti_k5", "ti_set_k5",
                          "fastpath_k5", "fastpath_set_k5"]
              if ("acm", n) in strong and ("freebase", n) in strong]
    flips = []
    for i, a in enumerate(common):
        for b in common[i + 1:]:
            ra = paired(by_seed(strong[("acm", a)]), by_seed(strong[("acm", b)]),
                        plan.STATS["bootstrap"], plan.STATS["seed"])
            rb = paired(by_seed(strong[("freebase", a)]), by_seed(strong[("freebase", b)]),
                        plan.STATS["bootstrap"], plan.STATS["seed"])
            va, vb = verdict(ra["ci95"]), verdict(rb["ci95"])
            if {va, vb} == {"above", "below"}:
                flips.append({"a": a, "b": b, "acm": ra["mean"], "freebase": rb["mean"]})
    means = {ds: [mean_sd([r["test"]["micro_f1"] for r in strong[(ds, n)].values()])[0]
                  for n in common] for ds in plan.DATASETS}
    tau = kendall_tau(means["acm"], means["freebase"]) if len(common) >= 3 else None
    out["H6"] = {"verdict": ("pending" if len(common) < 2 else
                             "supported" if flips else "not supported"),
                 "sign_flips": flips, "common_selectors": common, "kendall_tau": tau,
                 "ranking": {ds: [n for _, n in sorted(zip(means[ds], common), reverse=True)]
                             for ds in plan.DATASETS} if common else {}}
    return out


def _h(ci):
    v = verdict(ci)
    return {"above": "supported", "below": "contradicted", "overlaps": "inconclusive",
            "n/a": "pending"}[v]


def outcomes(hyp, stats_rows) -> dict:
    def beats(m):
        r = find(stats_rows, "freebase", f"{m}_k5", "canonical_k5")
        return None if r is None else r["verdict"] == "above"

    ti_beats = [beats("ti"), beats("ti_set")]
    fp_beats = [beats("fastpath"), beats("fastpath_set")]
    known = [x for x in ti_beats + fp_beats if x is not None]
    a = hyp["H1"]["verdict"] == "supported" and any(x for x in ti_beats if x)
    b = any(x for x in fp_beats if x) and not a
    c = hyp["H3"]["verdict"] == "supported"
    f = (bool(known) and not any(known) and hyp["H1"]["verdict"] != "supported"
         and hyp["H2"]["verdict"] != "supported")
    return {"A": a, "B": b, "C": c, "D": "requires V3 results (not read by V4)",
            "E": "requires V3 results (not read by V4)", "F": f,
            "complete": len(known) == 4 and "pending" not in (hyp["H1"]["verdict"],
                                                              hyp["H2"]["verdict"])}


def interpretation(hyp, out) -> str:
    if out["A"]:
        return ("Label-free empirical transition statistics recover useful paths: V1 failed "
                "because of its specific branching-volume approximation, not because lightweight "
                "structural statistics are insufficient.")
    if out["B"]:
        return ("Label-free transition statistics are not enough here, but cheap supervised "
                "path propagation recovers useful paths: task awareness is required, expensive "
                "neural search is not.")
    if out["F"]:
        return ("Neither label-free transition statistics nor cheap supervised propagation beat "
                "the structural baseline: lightweight selection appears insufficient and path "
                "selection needs a richer approach.")
    return ("Mixed or incomplete evidence: see the hypothesis table; no single explanation is "
            "supported yet.")


# --------------------------------------------------------------------------- efficiency
def efficiency_rows(strong) -> list[dict]:
    rows = []
    for ds in plan.DATASETS:
        ti = _read(V4 / "selectors" / "transition_info" / f"{ds}.json")
        st = _read(V4 / "selectors" / "fastpath" / f"{ds}__structure.json")
        beam = _read(V4 / "selectors" / "ti_beam" / f"{ds}.json")
        fps = [_read(p) for p in sorted((V4 / "selectors" / "fastpath").glob(f"{ds}__v4__seed*.json"))]
        fps = [f for f in fps if f]
        n_cand = ti["candidate_count"] if ti else None
        gen = ti["timing"]["candidate_generation_s"] if ti else None
        prop = float(np.mean([f["timing"]["label_propagation_s"] for f in fps])) if fps else None
        sets = float(np.mean([f["timing"]["set_selection_s"] for f in fps])) if fps else None
        homo = float(np.mean([f["timing"]["homophily_hybrid_s"] for f in fps])) if fps else None
        structure = st["timing"]["structure_s"] if st else None
        primary = (_read(V4 / "decisions" / "ti_primary.json") or {}).get("primary", "TI_cov")
        spec = {
            "canonical": {"transition": 0.0, "prop": 0.0, "set": 0.0},
            "random": {"transition": 0.0, "prop": 0.0, "set": 0.0},
            "diverse": {"transition": 0.0, "prop": 0.0, "set": 0.0},
            "hybrid": {"transition": 0.0, "prop": homo, "set": 0.0},
            "ti": {"transition": ti["timing"]["transition_statistics_s"] if ti else None,
                   "prop": 0.0, "set": 0.0},
            "ti_set": {"transition": ti["timing"]["transition_statistics_s"] if ti else None,
                       "prop": 0.0, "set": ((ti["timing"]["ti_set_greedy_s"][primary]
                                             + ti["timing"]["fingerprint_cosine_s"]) if ti else None)},
            "fastpath": {"transition": structure, "prop": prop, "set": 0.0},
            "fastpath_set": {"transition": structure, "prop": prop, "set": sets},
            "ti_beam": {"transition": ((beam["timing"]["beam_s"] + beam["timing"]["exact_rescoring_s"])
                                       if beam else None), "prop": 0.0, "set": 0.0},
        }
        peak = {"ti": ti["timing"]["peak_rss_mib"] if ti else None,
                "ti_set": ti["timing"]["peak_rss_mib"] if ti else None,
                "fastpath": st["timing"]["peak_rss_mib"] if st else None,
                "fastpath_set": st["timing"]["peak_rss_mib"] if st else None,
                "ti_beam": beam["timing"]["peak_rss_mib"] if beam else None}
        for name, parts in spec.items():
            if name == "diverse" and ds != "acm":
                continue
            cond = "ti_beam_k5" if name == "ti_beam" else f"{name}_k5"
            runs = strong.get((ds, cond), {})
            total = None
            if all(v is not None for v in parts.values()) and gen is not None:
                total = gen + sum(parts.values())
            gpu_h = [r["fit"]["train_runtime_s"] / 3600 for r in runs.values()]
            m, sd = mean_sd([pct(r["test"]["micro_f1"]) for r in runs.values()])
            rows.append({
                "dataset": ds, "selector": name, "candidate_space": (
                    beam["schema_valid_candidates_le_lmax"] if name == "ti_beam" and beam
                    else n_cand),
                "candidates_evaluated": (beam["generated_prefixes"] if name == "ti_beam" and beam
                                         else (0 if name == "random" else n_cand)),
                "final_paths": 5, "candidate_generation_s": gen,
                "transition_statistics_s": parts["transition"],
                "label_propagation_s": parts["prop"], "set_selection_s": parts["set"],
                "selector_total_s": total,
                "selector_cpu_h": total / 3600 if total is not None else None,
                "selector_gpu_h": 0.0, "selector_peak_rss_mib": peak.get(name),
                "final_training_gpu_h_per_run": float(np.mean(gpu_h)) if gpu_h else None,
                "final_training_gpu_h_total": float(np.sum(gpu_h)) if gpu_h else None,
                "final_runs": len(runs), "test_micro_f1_mean": m, "test_micro_f1_sd": sd})
    return rows


# --------------------------------------------------------------------------- selections
def selected_rows(strong) -> list[dict]:
    rows = []
    for (ds, name), runs in sorted(strong.items()):
        for seed, r in sorted(runs.items()):
            if r["mode"] == "hgt":
                continue
            paths = (r.get("discovery") or {}).get("paths") or []
            rows.append({"dataset": ds, "condition": name, "seed": seed, "k": r["k"],
                         "paths": " | ".join(p.get("abbr") or _abbr(p["path"]) for p in paths),
                         "canonical_ranks": " ".join(str(p.get("canonical_rank")) for p in paths)})
    for ds in plan.DATASETS:
        ti = _read(V4 / "selectors" / "transition_info" / f"{ds}.json")
        if ti:
            names = [c["abbr"] for c in ti["candidates"]]
            for v in plan.TI["variants"]:
                for k in plan.KS:
                    rows.append({"dataset": ds, "condition": f"selector_file:{v}_top{k}",
                                 "seed": "all", "k": k,
                                 "paths": " | ".join(names[i] for i in ti["top_k"][v][str(k)]),
                                 "canonical_ranks": " ".join(str(i + 1) for i in
                                                             ti["top_k"][v][str(k)])})
                    idx = ti["ti_set"][v]["top_k"][str(k)]
                    rows.append({"dataset": ds, "condition": f"selector_file:{v}_set{k}",
                                 "seed": "all", "k": k,
                                 "paths": " | ".join(names[i] for i in idx),
                                 "canonical_ranks": " ".join(str(i + 1) for i in idx)})
    return rows


def _abbr(path):
    from .tasks import abbr
    return abbr(path)


# --------------------------------------------------------------------------- plots
def plots(strong, diag_rows, eff) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    made = []
    primary = (_read(V4 / "decisions" / "ti_primary.json") or {}).get("primary", "TI_cov")
    if diag_rows:
        util = [float(r["frozen_V2_single_path_accuracy"]) * 100 for r in diag_rows]
        panels = [("V1_structural_score", "structural score (log)", True),
                  ("homophily", "training-label homophily", False),
                  (primary, f"Transition-Info ({primary})", False),
                  ("FastPath_Q", "FastPath Q_FP", False)]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (col, label, logx) in zip(axes.ravel(), panels):
            xs = [float(r[col]) if r[col] not in (None, "") else np.nan for r in diag_rows]
            ax.scatter(xs, util, s=14)
            if logx:
                ax.set_xscale("log")
            from .stats import spearman
            rho = spearman([x if not np.isnan(x) else None for x in xs], util, 200, 0)["rho"]
            ax.set_title(f"{label}: Spearman {rho:.2f}" if rho is not None else label)
            ax.set_xlabel(label)
            ax.set_ylabel("V2 single-path test accuracy (%)")
            for r, x, y in zip(diag_rows, xs, util):
                if r["path"] in plan.DIAGNOSTIC["named_paths"]:
                    ax.annotate(r["path"], (x, y), fontsize=7)
        fig.suptitle("ACM: selector score vs frozen V2 single-path utility (88 paths)")
        fig.tight_layout()
        fig.savefig(PLOTS / "score_vs_utility.png", dpi=150)
        plt.close(fig)
        made.append("score_vs_utility.png")
    diag = _read(V4 / "diagnostics" / "acm_score_correlations.json")
    if diag:
        names = ["structural", "homophily", "TI_raw", "TI_cov", "TI_norm", "FastPath",
                 "FastPath_probe"]
        fig, ax = plt.subplots(figsize=(9, 4))
        for j, u in enumerate(("test_utility", "val_utility")):
            vals = [diag[u][n]["rho"] or 0.0 for n in names]
            ci = [diag[u][n]["ci95"] or [v, v] for n, v in zip(names, vals)]
            err = np.array([[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]])
            ax.bar(np.arange(len(names)) + 0.38 * j, vals, 0.38, yerr=err, capsize=3,
                   label=u.replace("_", " "))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(names)) + 0.19)
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel("Spearman rho with V2 utility")
        ax.legend()
        ax.set_title("ACM: rank agreement of selector scores with downstream path utility")
        fig.tight_layout()
        fig.savefig(PLOTS / "selector_rank_comparison.png", dpi=150)
        plt.close(fig)
        made.append("selector_rank_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    any_sel = False
    for ax, ds in zip(axes, plan.DATASETS):
        conds = [c["name"] for c in plan.conditions() if c["dataset"] == ds and c["k"] == 5
                 and c["token_control"] is None and (ds, c["name"]) in strong]
        sets = {}
        for name in conds:
            runs = strong[(ds, name)]
            rec = runs[min(runs)]
            sets[name] = [p.get("abbr") or _abbr(p["path"]) for p in rec["discovery"]["paths"]]
        paths = sorted({p for s in sets.values() for p in s})
        if not paths:
            ax.set_visible(False)
            continue
        any_sel = True
        mat = np.array([[1.0 if p in sets[c] else 0.0 for p in paths] for c in conds])
        ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=8)
        ax.set_xticks(range(len(paths)))
        ax.set_xticklabels(paths, rotation=90, fontsize=6)
        ax.set_title(f"{ds}: k=5 selections (first seed)")
    if any_sel:
        fig.tight_layout()
        fig.savefig(PLOTS / "selected_paths.png", dpi=150)
        made.append("selected_paths.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    any_eff = False
    for ax, ds in zip(axes, plan.DATASETS):
        rows = [r for r in eff if r["dataset"] == ds and r["test_micro_f1_mean"] is not None
                and r["selector_total_s"] is not None]
        for r in rows:
            any_eff = True
            x = max(r["selector_total_s"], 1e-2)
            ax.errorbar(x, r["test_micro_f1_mean"], yerr=r["test_micro_f1_sd"] or 0, fmt="o",
                        capsize=3)
            ax.annotate(r["selector"], (x, r["test_micro_f1_mean"]), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("selector search cost (CPU s; 0 GPU s)")
        ax.set_ylabel("test micro-F1 (%)")
        ax.set_title(f"{ds}: accuracy vs selector search cost (k=5)")
    if any_eff:
        fig.tight_layout()
        fig.savefig(PLOTS / "accuracy_vs_search_cost.png", dpi=150)
        made.append("accuracy_vs_search_cost.png")
    plt.close(fig)
    return made


# --------------------------------------------------------------------------- reports
def reports(main, stats_rows, hyp, out, eff, bridge_tbl, diag) -> None:
    primary = (_read(V4 / "decisions" / "ti_primary.json") or {}).get("primary")
    ext = _read(V4 / "decisions" / "freebase_extension.json")
    lines = ["# V4-Hedge final summary", "",
             f"Protocol hash `{plan.protocol_hash()}`. Generated automatically from V4 outputs "
             "and frozen V2 files only (no V3 file is read). Metric: test micro-F1 (%), mean ± "
             "sample SD over matched seeds; ACM is exploratory, Freebase is the clean "
             "confirmation dataset.", "",
             f"Primary Transition-Info variant (validation rule): **{primary}**.",
             f"Freebase seed extension: {ext['extend'] if ext else 'not decided'}"
             + (f" ({', '.join(ext['extended_conditions'])})" if ext and ext["extend"] else ""),
             ""]
    for ds in plan.DATASETS:
        pub = plan.PUBLISHED_HGT[ds]
        lines += [f"## {ds}", "", f"Published HGB HGT: micro {pub['micro_f1']}, macro "
                  f"{pub['macro_f1']}.", "",
                  "| condition | seeds | test micro-F1 | test macro-F1 | val micro-F1 |",
                  "|---|---|---|---|---|"]
        for r in main:
            if r["dataset"] == ds and r["seeds_completed"]:
                lines.append(f"| {r['condition']} | {r['seeds_completed']} | "
                             f"{fmt(r['test_micro_f1_mean'], r['test_micro_f1_sd'])} | "
                             f"{fmt(r['test_macro_f1_mean'], r['test_macro_f1_sd'])} | "
                             f"{fmt(r['val_micro_f1_mean'], r['val_micro_f1_sd'])} |")
        lines.append("")
    if bridge_tbl:
        lines += ["## ACM under the unchanged V2 backbone (V2 results reused)", "",
                  "| backbone | condition | seeds | test acc | val acc |", "|---|---|---|---|---|"]
        for r in bridge_tbl:
            lines.append(f"| {r['backbone']} | {r['condition']} | {r['seeds_completed']} | "
                         f"{fmt(r['test_micro_f1_mean'], r['test_micro_f1_sd'])} | "
                         f"{fmt(r['val_micro_f1_mean'], r['val_micro_f1_sd'])} |")
        lines.append("")
    if diag:
        lines += ["## ACM diagnostic (88 paths, frozen V2 single-path test utility)", "",
                  "| score | Spearman rho [95% CI] | p |", "|---|---|---|"]
        for name, v in diag["test_utility"].items():
            if v["rho"] is not None:
                ci = (f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" if v.get("ci95") else "")
                lines.append(f"| {name} | {v['rho']:.3f} {ci} | {v['p']:.2g} |")
        lines += ["", "Named paths (rank under each score, 1 = best):", ""]
        for p, ranks in diag["named_path_ranks"].items():
            lines.append(f"- {p}: " + ", ".join(f"{k} {v}" for k, v in ranks.items()))
        lines.append("")
    lines += ["## Hypotheses", "", "| hypothesis | verdict |", "|---|---|"]
    for key in plan.HYPOTHESES:
        lines.append(f"| {key}: {plan.HYPOTHESES[key]} | **{hyp[key]['verdict']}** |")
    lines += ["", "## Outcome", ""]
    lines += [f"- {k}: {v}" for k, v in out.items()]
    lines += ["", f"Interpretation: {interpretation(hyp, out)}", "",
              "## Key paired comparisons (strong backbone, k=5)", "",
              "| dataset | a - b | n | mean [95% CI] (pts) | p_perm | verdict |",
              "|---|---|---|---|---|---|"]
    for r in stats_rows:
        if r["scope"] == "strong" and r.get("k") == 5 and r["mean_diff_pts"] is not None:
            ci = (f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]" if r["ci95_low"] is not None
                  else "")
            lines.append(f"| {r['dataset']} | {r['a']} - {r['b']} | {r['n']} | "
                         f"{r['mean_diff_pts']:.2f} {ci} | "
                         f"{r['p_perm'] if r['p_perm'] is None else round(r['p_perm'], 3)} | "
                         f"{r['verdict']} |")
    lines += ["", "## Selector cost", "",
              "| dataset | selector | search CPU s | GPU s | final GPU h (total) |",
              "|---|---|---|---|---|"]
    for r in eff:
        lines.append(f"| {r['dataset']} | {r['selector']} | {fmt(r['selector_total_s'], None, 1)}"
                     f" | 0 | {fmt(r['final_training_gpu_h_total'], None, 2)} |")
    (SUMMARY / "final_summary.md").write_text("\n".join(lines) + "\n")
    claims = ["# V4-Hedge paper claims (rule-based, generated)", "",
              "Each claim below follows mechanically from the pre-registered decision rules in "
              "protocol/V4_HEDGE_PROTOCOL.md; verdicts marked pending need more completed runs.",
              ""]
    for key in plan.HYPOTHESES:
        claims.append(f"- **{key}** ({hyp[key]['verdict']}): {plan.HYPOTHESES[key]}")
    claims += ["", f"Outcome classes: {json.dumps(out)}", "",
               f"Interpretation: {interpretation(hyp, out)}", "",
               "Caveats: ACM is exploratory (its test set was seen in V1/V2; the diagnostic uses "
               "V2 test utilities post hoc); with n=5 seeds p-values are descriptive; all V4 "
               "selectors use zero GPU seconds by construction; V4 does not read V3, so outcomes "
               "D/E are decided only after both campaigns finish."]
    (SUMMARY / "paper_claims.md").write_text("\n".join(claims) + "\n")


def aggregate() -> dict:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    strong, bridge, v2 = load_strong(), load_bridge(), load_v2_frozen()
    main = main_rows(strong)
    _csv(SUMMARY / "main_results.csv", main)
    bridge_tbl = bridge_rows(bridge, v2)
    sel_rows = [dict(r, backbone="strong") for r in main if r["group"].startswith("selectors")]
    _csv(SUMMARY / "selector_results.csv", sel_rows + bridge_tbl)
    _csv(SUMMARY / "selected_paths.csv", selected_rows(strong))
    stats_rows = comparisons(strong, bridge, v2)
    _csv(SUMMARY / "statistics.csv", stats_rows)
    diag = _read(V4 / "diagnostics" / "acm_score_correlations.json")
    corr_rows = []
    if diag:
        for u in ("test_utility", "val_utility"):
            for name, v in diag[u].items():
                corr_rows.append({"dataset": "acm", "utility": u, "score": name, "n": v["n"],
                                  "rho": v["rho"], "p": v["p"],
                                  "ci95_low": (v["ci95"] or [None, None])[0],
                                  "ci95_high": (v["ci95"] or [None, None])[1]})
        for key, v in diag["differences"].items():
            u, pair = key.split(":")
            corr_rows.append({"dataset": "acm", "utility": u, "score": f"diff {pair}",
                              "n": v["n"], "rho": v["diff"],
                              "ci95_low": (v["ci95"] or [None, None])[0],
                              "ci95_high": (v["ci95"] or [None, None])[1]})
    fb = _read(V4 / "selectors" / "transition_info" / "freebase.json")
    fpf = _read(V4 / "selectors" / "fastpath" / "freebase__v4__seed0.json")
    if fb and fpf:
        from .stats import spearman
        struct = [c["structural_score"] for c in fb["candidates"]]
        for name, vals in [(v, [c[v] for c in fb["candidates"]]) for v in plan.TI["variants"]] + [
                ("FastPath_seed0", [c["Q_FP"] for c in fpf["candidates"]])]:
            s = spearman(vals, struct, plan.STATS["bootstrap"], plan.STATS["seed"])
            corr_rows.append({"dataset": "freebase", "utility": "structural score (descriptive)",
                              "score": name, "n": s["n"], "rho": s["rho"], "p": s["p"]})
    _csv(SUMMARY / "correlations.csv", corr_rows)
    eff = efficiency_rows(strong)
    _csv(SUMMARY / "efficiency.csv", eff)
    hyp = hypotheses(strong, stats_rows, diag)
    out = outcomes(hyp, stats_rows)
    diag_rows = []
    path = V4 / "diagnostics" / "acm_path_scores.csv"
    if path.exists():
        with path.open() as handle:
            diag_rows = list(csv.DictReader(handle))
    made = plots(strong, diag_rows, eff)
    reports(main, stats_rows, hyp, out, eff, bridge_tbl, diag)
    record = {"protocol_hash": plan.protocol_hash(), "hypotheses": hyp, "outcomes": out,
              "interpretation": interpretation(hyp, out), "plots": made,
              "completed_runs": sum(len(v) for v in strong.values()),
              "bridge_runs": sum(len(v) for v in bridge.values())}
    (SUMMARY / "v4_summary.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


if __name__ == "__main__":
    result = aggregate()
    print(json.dumps({"completed_runs": result["completed_runs"],
                      "bridge_runs": result["bridge_runs"],
                      "verdicts": {k: v["verdict"] for k, v in result["hypotheses"].items()}},
                     indent=2))
