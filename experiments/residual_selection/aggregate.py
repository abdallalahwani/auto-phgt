"""V3 aggregation (CPU only; safe to run at any time): tables, statistics, plots, reports.

    python -m experiments.residual_selection.aggregate
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

from experiments.residual_selection import published
from experiments.residual_selection.plan import (DATASETS, FREEBASE_EXTENSION, HYPOTHESES, MAG_CONDITIONS,
                                 REPRODUCTION_TARGET, SEEDS, V3, hgb_conditions)

SUMMARY, PAPER, PLOTS = V3 / "summary", V3 / "paper", V3 / "plots"
ABBR = {"paper": "P", "author": "A", "subject": "S", "term": "T", "venue": "V", "book": "B",
        "film": "F", "music": "M", "sports": "Sp", "people": "Pe", "location": "L",
        "organization": "O", "business": "Bu", "field_of_study": "F", "institution": "I"}
PRIMARY = {"acm": {"lmax": 4, "k": 5}, "dblp": {"lmax": 6, "k": 5}, "freebase": {"lmax": 4, "k": 5}}
V2_ACM_HGT = 86.44  # V2 10-seed HGT (artifacts/v3/provenance snapshot of artifacts/v2/summary)


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def abbr(path) -> str:
    out = ABBR.get(path[0], path[0])
    for i in range(1, len(path), 2):
        rel = path[i]
        out += (f"-{rel}-" if rel in ("cite", "ref", "cites") else "-") + ABBR.get(path[i + 1], path[i + 1])
    return out


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


# --------------------------------------------------------------------------- loading
def hgb_runs() -> dict:
    """{condition_id: {seed: record}} for completed HGB final runs."""
    out = {}
    for path in (V3 / "results").glob("*/*.json"):
        rec = _load(path)
        if rec and rec.get("status") == "completed" and rec.get("suite") == "autophgt_v3" \
                and rec.get("dataset") in DATASETS:
            out.setdefault(rec["condition"], {})[rec["seed"]] = rec
    return out


def mag_runs() -> dict:
    out = {}
    for path in (V3 / "results" / "ogbn-mag").glob("*.json"):
        rec = _load(path)
        if not (rec and rec.get("status") == "completed"):
            continue
        tag = rec.get("experiment_id", "")
        name = tag.split("__")[1] if "__" in tag else None
        cont = tag.endswith("__e250")
        rec["_continued"] = cont
        key = f"mag__{name}"
        if cont or rec["seed"] not in out.get(key, {}):
            out.setdefault(key, {})[rec["seed"]] = rec
    return out


def searches() -> list[dict]:
    return [r for r in (_load(p) for p in sorted((V3 / "searches").glob("*.json"))) if r]


def metric(rec, split="test", name="micro_f1"):
    if rec.get("dataset") == "ogbn-mag":
        block = rec["test"] if split == "test" else rec["validation"]
        return block["accuracy" if name == "micro_f1" else "macro_f1"]
    return rec["test" if split == "test" else "validation"][name]


# --------------------------------------------------------------------------- statistics
def paired(a: dict, b: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison over matched seeds: a - b in test micro-F1 (accuracy on MAG)."""
    seeds = sorted(s for s in a if s in b)
    diffs = [metric(a[s]) - metric(b[s]) for s in seeds]
    out = {"n": len(diffs), "seeds": seeds, "per_seed": diffs, "mean": None, "sd": None,
           "ci95": None, "p_perm": None, "p_t": None, "wins": sum(d > 0 for d in diffs)}
    if not diffs:
        return out
    out["mean"] = statistics.fmean(diffs)
    if len(diffs) < 2:
        return out
    out["sd"] = statistics.stdev(diffs)
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    out["ci95"] = [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]]
    observed = abs(out["mean"])
    if len(diffs) <= 14:
        signs = list(itertools.product((1, -1), repeat=len(diffs)))
        extreme = sum(abs(statistics.fmean(s * d for s, d in zip(sg, diffs))) >= observed - 1e-12
                      for sg in signs)
        out["p_perm"] = extreme / len(signs)
    if out["sd"] > 0:
        from scipy import stats
        out["p_t"] = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    return out


def verdict(comp) -> str:
    if not comp or comp["ci95"] is None:
        return "insufficient data"
    lo, hi = comp["ci95"]
    return "supported" if lo > 0 else ("contradicted" if hi < 0 else "inconclusive")


# --------------------------------------------------------------------------- tables
def condition_rows(runs: dict) -> list[dict]:
    rows = []
    for cond in hgb_conditions():
        seeds = SEEDS[cond["dataset"]] + (FREEBASE_EXTENSION["seeds"] if cond["dataset"] == "freebase" else [])
        members = runs.get(cond["id"], {})
        done = [members[s] for s in seeds if s in members]
        row = {"dataset": cond["dataset"], "condition": cond["name"], "group": cond["group"],
               "priority": f"P{cond['priority']}", "selector": cond["selector"], "k": cond["k"],
               "lmax": cond["lmax"], "token_control": cond["token_control"],
               "seeds_completed": len(done), "seeds": " ".join(str(r["seed"]) for r in done)}
        for label, split, name in (("test_micro_f1", "test", "micro_f1"),
                                   ("test_macro_f1", "test", "macro_f1"),
                                   ("val_micro_f1", "val", "micro_f1")):
            m, sd = mean_sd([100 * metric(r, split, name) for r in done])
            row[f"{label}_mean"], row[f"{label}_sd"] = m, sd
        for label, fn in (("best_epoch", lambda r: r["fit"]["best_epoch"]),
                          ("parameters", lambda r: r["parameters"]),
                          ("train_runtime_s", lambda r: r["fit"]["train_runtime_s"]),
                          ("peak_gpu_mib", lambda r: (r["fit"]["gpu_memory"] or {}).get("max_allocated_mib"))):
            row[f"{label}_mean"] = mean_sd([fn(r) for r in done])[0]
        rows.append(row)
    return rows


def mag_rows(runs: dict) -> list[dict]:
    rows = []
    for cond in MAG_CONDITIONS:
        members = runs.get(cond["id"], {})
        done = [members[s] for s in SEEDS["ogbn-mag"] if s in members]
        row = {"dataset": "ogbn-mag", "condition": cond["name"], "seeds_completed": len(done),
               "continued_to_250": sum(r["_continued"] for r in done)}
        for label, split, name in (("test_accuracy", "test", "micro_f1"),
                                   ("test_macro_f1", "test", "macro_f1"),
                                   ("val_accuracy", "val", "micro_f1")):
            row[f"{label}_mean"], row[f"{label}_sd"] = mean_sd([100 * metric(r, split, name) for r in done])
        row["best_epoch_mean"] = mean_sd([r["best_epoch"] for r in done])[0]
        row["epochs_mean"] = mean_sd([r["epochs_completed"] for r in done])[0]
        row["parameters"] = done[0]["parameters"] if done else None
        row["train_runtime_h_mean"] = mean_sd([r["training_runtime_s"] / 3600 for r in done])[0]
        rows.append(row)
    return rows


def comparisons(runs: dict, mag: dict) -> list[dict]:
    """Pre-registered paired comparisons (test micro-F1; accuracy on MAG)."""
    out = []

    def add(hyp, ds, a, b, primary=True):
        comp = paired(runs.get(a, {}) if ds != "ogbn-mag" else mag.get(a, {}),
                      runs.get(b, {}) if ds != "ogbn-mag" else mag.get(b, {}))
        out.append({"hypothesis": hyp, "dataset": ds, "a": a, "b": b, "primary": primary,
                    "verdict": verdict(comp), **{k: comp[k] for k in (
                        "n", "mean", "sd", "ci95", "p_perm", "p_t", "wins", "seeds")}})

    main = {"acm": "acm__rcms_k5", "dblp": "dblp__rcms_k5_L6", "freebase": "freebase__rcms_k5"}
    suffix = {"acm": "_k5", "dblp": "_k5_L6", "freebase": "_k5"}
    for ds, rc in main.items():
        add("H2", ds, rc, f"{ds}__canonical{suffix[ds]}")
        add("H3", ds, rc, f"{ds}__hybrid{suffix[ds]}", primary=ds != "acm")
        add("H4", ds, rc, f"{ds}__hgt_strong")
        add("H5", ds, rc, f"{ds}__rcms_indep{suffix[ds]}")
        add("H6", ds, rc, f"{ds}__rcms_nohgt{suffix[ds]}")
        add("random", ds, rc, f"{ds}__random{suffix[ds]}", primary=False)
        if ds in ("acm", "dblp"):
            add("capacity", ds, rc, f"{ds}__hgt_param_matched", primary=False)
    add("H7", "acm", "acm__rcms_k5", "acm__rcms_k5_dummy")
    add("H7", "acm", "acm__rcms_k5", "acm__rcms_k5_shuffle")
    add("H7", "dblp", "dblp__rcms_k5_L6", "dblp__rcms_k5_L6_dummy")
    for ds in ("acm", "dblp", "freebase"):
        add("k2", ds, f"{ds}__rcms_k2", f"{ds}__canonical_k2", primary=False)
        add("k2", ds, f"{ds}__rcms_k2", f"{ds}__hybrid_k2", primary=False)
    add("H8", "dblp", "dblp__rcms_k5_L6", "dblp__lmsps_paths_k30", primary=False)
    add("long_range", "acm", "acm__rcms_k5_L6", "acm__rcms_k5", primary=False)
    add("long_range", "freebase", "freebase__rcms_k5_L6", "freebase__rcms_k5", primary=False)
    for other in ("hgt", "canonical_k5", "random_k5", "dummy_k5"):
        add({"hgt": "H4", "canonical_k5": "H2", "random_k5": "random", "dummy_k5": "H7"}[other],
            "ogbn-mag", "mag__rcms_k5", f"mag__{other}", primary=False)
    return out


def selected_path_rows(runs) -> list[dict]:
    rows = []
    for cid, members in runs.items():
        for seed, rec in members.items():
            for p in (rec.get("discovery") or {}).get("paths", []) or []:
                rows.append({"dataset": rec["dataset"], "condition": rec["name"], "seed": seed,
                             "rank": p.get("rank"), "path": abbr(p["path"]),
                             "structural_score": p.get("structural_score", p.get("score")),
                             "canonical_rank": p.get("canonical_rank")})
    return rows


def proxy_rows(search_list) -> list[dict]:
    rows = []
    for s in search_list:
        for c in s["per_candidate"]:
            rows.append({"dataset": s["dataset"], "lmax": s["lmax"], "seed": s["seed"],
                         "path": abbr(c["path"]), "canonical_rank": c["canonical_rank"],
                         "structural_score": c["structural_score"], "coverage": c["coverage"],
                         "homophily": c.get("homophily"), "utility_step1": c["utility_step1"],
                         "utility_step1_nohgt": c["utility_step1_nohgt"]})
    return rows


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    from scipy import stats
    rho = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic
    return None if rho != rho else float(rho)


def stability_rows(search_list) -> list[dict]:
    rows = []
    groups = {}
    for s in search_list:
        groups.setdefault((s["dataset"], s["lmax"]), []).append(s)
    for (ds, lmax), members in sorted(groups.items()):
        for variant in ("rcms", "rcms_indep", "rcms_nohgt"):
            for k in ("2", "5"):
                sets = [frozenset(abbr(p) for p in m["selected"][variant].get(k, [])) for m in members]
                jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
                freq = {}
                for s_ in sets:
                    for p in s_:
                        freq[p] = freq.get(p, 0) + 1
                rows.append({"row": "set", "dataset": ds, "lmax": lmax, "variant": variant, "k": k,
                             "searches": len(members),
                             "mean_pairwise_jaccard": statistics.fmean(jac) if jac else None,
                             "selection_frequency": json.dumps(dict(sorted(freq.items(), key=lambda x: -x[1])))})
        util = [{abbr(c["path"]): c["utility_step1"] for c in m["per_candidate"]} for m in members]
        rhos = []
        for a, b in itertools.combinations(util, 2):
            keys = sorted(set(a) & set(b))
            r = spearman([a[k] for k in keys], [b[k] for k in keys])
            if r is not None:
                rhos.append(r)
        per_path = {}
        for u in util:
            for k, v in u.items():
                per_path.setdefault(k, []).append(v)
        rows.append({"row": "utility", "dataset": ds, "lmax": lmax, "searches": len(members),
                     "mean_pairwise_spearman_step1": statistics.fmean(rhos) if rhos else None,
                     "mean_utility_variance": statistics.fmean(
                         statistics.pvariance(v) for v in per_path.values() if len(v) > 1)
                     if any(len(v) > 1 for v in per_path.values()) else None})
    return rows


def efficiency_rows(runs, mag, search_list) -> list[dict]:
    rows = []
    for s in search_list:
        t = s["timing"]
        rows.append({"kind": "rcms_search", "dataset": s["dataset"], "lmax": s["lmax"], "seed": s["seed"],
                     "candidates_raw": s["candidates"]["raw"], "candidates_kept": s["candidates"]["kept"],
                     "candidate_generation_s": t.get("candidates_s", t.get("candidates_and_cache_s")),
                     "path_cache_s": t.get("path_cache_s"),
                     "backbone_s": t.get("hgt_crossfit_s", t.get("hgt_train_s")),
                     "search_s": t.get("search_full_s"), "total_s": t.get("total_s"),
                     "gpu_hours": (t.get("total_s") or 0) / 3600})
    for cid, members in runs.items():
        for seed, r in members.items():
            rows.append({"kind": "final", "dataset": r["dataset"], "condition": r["name"], "seed": seed,
                         "selector": r.get("selector"),
                         "selection_s": r.get("selection_runtime_s"),
                         "train_s": r["fit"]["train_runtime_s"],
                         "gpu_hours": r["fit"]["train_runtime_s"] / 3600,
                         "peak_gpu_mib": (r["fit"]["gpu_memory"] or {}).get("max_allocated_mib"),
                         "peak_cpu_rss_mib": r["environment"].get("peak_cpu_rss_mib"),
                         "parameters": r["parameters"],
                         "paths_selected": len(r["templates"]) if r.get("templates") else 0})
    for cid, members in mag.items():
        for seed, r in members.items():
            rows.append({"kind": "final", "dataset": "ogbn-mag", "condition": cid.split("__")[1], "seed": seed,
                         "train_s": r["training_runtime_s"], "gpu_hours": r["training_runtime_s"] / 3600,
                         "peak_gpu_mib": (r.get("gpu_memory") or {}).get("max_allocated_mib"),
                         "peak_cpu_rss_mib": r.get("peak_cpu_rss_mib"), "parameters": r["parameters"]})
    manifest = V3 / "manifest.jsonl"
    if manifest.exists():
        total = 0.0
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("runtime_s") and entry.get("gpu") not in (None, ""):
                total += entry["runtime_s"]
        rows.append({"kind": "campaign", "gpu_slot_hours": total / 3600})
    return rows


def published_rows(runs) -> list[dict]:
    rows = published.rows()
    for cid in ("dblp__lmsps_paths_k30", "dblp__rcms_k5_L6", "acm__hgt_strong", "dblp__hgt_strong",
                "freebase__hgt_strong"):
        members = runs.get(cid, {})
        if members:
            mi = mean_sd([100 * metric(r) for r in members.values()])
            ma = mean_sd([100 * metric(r, "test", "macro_f1") for r in members.values()])
            rows.append({"method": f"OUR RUN: {cid}", "dataset": cid.split("__")[0],
                         "micro_f1": mi[0], "micro_sd": mi[1], "macro_f1": ma[0], "macro_sd": ma[1],
                         "source": "V3", "label": "OUR RUN", "note": f"{len(members)} seeds"})
    return rows


# --------------------------------------------------------------------------- reports
def fmt(m, sd=None, digits=2):
    if m is None:
        return "–"
    return f"{m:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "")


def hypothesis_verdicts(comps, cond_rows) -> dict:
    by = {}
    for c in comps:
        by.setdefault(c["hypothesis"], []).append(c)
    out = {}
    hgt = {r["dataset"]: r for r in cond_rows if r["condition"] == "hgt_strong"}
    acm = hgt.get("acm", {}).get("test_micro_f1_mean")
    pub = 91.00
    if acm is None:
        out["H1"] = ("insufficient data", "strong HGT ACM runs missing")
    else:
        v2_gap, gap = pub - V2_ACM_HGT, pub - acm
        out["H1"] = ("supported" if gap <= 0.5 * v2_gap else "not supported",
                     f"ACM strong HGT {acm:.2f} vs published 91.00 (gap {gap:.2f}; V2 gap {v2_gap:.2f})")

    def rule(h, need, datasets=None, primary_only=True):
        items = [c for c in by.get(h, []) if (c["primary"] or not primary_only)
                 and (datasets is None or c["dataset"] in datasets)]
        verdicts = [c["verdict"] for c in items]
        sup, con = verdicts.count("supported"), verdicts.count("contradicted")
        detail = "; ".join(f"{c['dataset']} {c['b'].split('__')[1]}: {c['verdict']} "
                           f"({fmt(100 * c['mean'] if c['mean'] is not None else None)} pts)"
                           for c in items)
        if not items or all(v == "insufficient data" for v in verdicts):
            return ("insufficient data", detail)
        if con:
            return ("contradicted" if con >= need else "not supported", detail)
        return ("supported" if sup >= need else "inconclusive", detail)

    out["H2"] = rule("H2", 2)
    out["H3"] = rule("H3", 1, datasets=("dblp", "freebase"))
    out["H4"] = rule("H4", 2)
    out["H5"] = rule("H5", 1)
    out["H6"] = rule("H6", 1)
    out["H7"] = rule("H7", 2)
    out["H8"] = ("not testable", "LMSPS search was not reproduced in V3; see efficiency.csv for RCMS "
                                 "search cost and statistics.csv for RCMS vs LMSPS paths on DBLP")
    return out


def outcome(verdicts) -> str:
    h2, h3, h4 = (verdicts.get(h, ("",))[0] for h in ("H2", "H3", "H4"))
    if h2 == "supported" and h3 == "supported" and h4 == "supported":
        return "A: RCMS beats hybrid/random-level selectors and strong HGT on clean datasets"
    if h4 == "contradicted":
        return "D: strong HGT eliminates the Auto-PHGT gain"
    if h2 == "supported" and h4 != "supported":
        return "C: RCMS improves selector quality but does not beat strong HGT"
    return "mixed / not yet determined (see per-hypothesis verdicts)"


def write_reports(cond_rows, magr, comps, verdicts, stab, eff) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    lines = ["# V3 results summary", "", "Generated automatically by experiments/residual_selection/aggregate.py; "
             "means ± sample SD over completed seeds (percent).", ""]
    for ds in DATASETS:
        lines += [f"## {ds}", "", "| condition | seeds | test micro-F1 | test macro-F1 | val micro-F1 |",
                  "|---|---|---|---|---|"]
        for r in cond_rows:
            if r["dataset"] == ds:
                lines.append(f"| {r['condition']} | {r['seeds_completed']} | "
                             f"{fmt(r['test_micro_f1_mean'], r['test_micro_f1_sd'])} | "
                             f"{fmt(r['test_macro_f1_mean'], r['test_macro_f1_sd'])} | "
                             f"{fmt(r['val_micro_f1_mean'], r['val_micro_f1_sd'])} |")
        lines.append("")
    lines += ["## ogbn-mag", "", "| condition | seeds | test acc | val acc | epochs |", "|---|---|---|---|---|"]
    for r in magr:
        lines.append(f"| {r['condition']} | {r['seeds_completed']} | "
                     f"{fmt(r['test_accuracy_mean'], r['test_accuracy_sd'])} | "
                     f"{fmt(r['val_accuracy_mean'], r['val_accuracy_sd'])} | {fmt(r['epochs_mean'], None, 0)} |")
    lines += ["", "## Pre-registered comparisons (a − b, test micro-F1 points)", "",
              "| hyp | dataset | a | b | n | mean | 95% bootstrap CI | p (perm) | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        ci = c["ci95"]
        lines.append(f"| {c['hypothesis']} | {c['dataset']} | {c['a'].split('__')[1]} | {c['b'].split('__')[1]} "
                     f"| {c['n']} | {fmt(100 * c['mean'] if c['mean'] is not None else None)} | "
                     f"{'–' if ci is None else f'[{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]'} | "
                     f"{fmt(c['p_perm'], None, 3)} | {c['verdict']} |")
    (PAPER / "v3_results_summary.md").write_text("\n".join(lines) + "\n")
    claims = ["# Paper claims (automatically derived; pre-registered rules)", "",
              f"Outcome: **{outcome(verdicts)}**", "",
              "Rule: a paired comparison is *supported* if its 95% paired-bootstrap CI over seeds "
              "excludes 0 in the predicted direction, *contradicted* if it excludes 0 in the "
              "opposite direction, otherwise *inconclusive*. Published numbers are context only.", ""]
    for h, text in HYPOTHESES.items():
        status, detail = verdicts.get(h, ("insufficient data", ""))
        claims += [f"## {h}: {status}", "", text, "", f"Evidence: {detail}", ""]
    (PAPER / "paper_claims.md").write_text("\n".join(claims) + "\n")


def latex(cond_rows, magr, comps) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    keys = ["hgt_strong", "hgt_param_matched", "canonical", "random", "hybrid", "rcms"]
    table = {}
    for r in cond_rows:
        prim = PRIMARY[r["dataset"]]
        name = r["condition"]
        base = name.replace(f"_k{prim['k']}", "").replace("_L6", "") if r["group"] == "selectors_k5" else name
        if r["group"] == "selectors_k5" and not (r["lmax"] == prim["lmax"] and r["k"] == prim["k"]):
            continue
        table[(r["dataset"], base)] = fmt(r["test_micro_f1_mean"], r["test_micro_f1_sd"])
    mag = {r["condition"].replace("_k5", ""): fmt(r["test_accuracy_mean"], r["test_accuracy_sd"]) for r in magr}
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"Method & ACM & DBLP & Freebase & ogbn-mag \\", r"\midrule"]
    for k in keys:
        rows.append(f"{k.replace('_', ' ')} & " + " & ".join(
            [table.get((ds, k), "–") for ds in DATASETS] + [mag.get(k.replace("_strong", ""), "–")]) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "main_table.tex").write_text("\n".join(rows) + "\n")
    rows = [r"\begin{tabular}{llccc}", r"\toprule",
            r"Dataset & Setting & Independent & Without HGT & Full RCMS \\", r"\midrule"]
    for ds, prim in PRIMARY.items():
        cells = []
        for sel in ("rcms_indep", "rcms_nohgt", "rcms"):
            name = f"{sel}_k{prim['k']}" + ("_L6" if prim["lmax"] == 6 else "")
            cells.append(next((fmt(r["test_micro_f1_mean"], r["test_micro_f1_sd"]) for r in cond_rows
                               if r["dataset"] == ds and r["condition"] == name), "–"))
        rows.append(f"{ds} & k={prim['k']}, L$\\le${prim['lmax']} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "selector_ablation.tex").write_text("\n".join(rows) + "\n")


def efficiency_latex(eff) -> None:
    groups = {}
    for r in eff:
        if r.get("kind") == "rcms_search":
            groups.setdefault((r["dataset"], r["lmax"]), []).append(r)
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r"Dataset & L & Candidates & Backbone (s) & Search (s) & Total (s) \\", r"\midrule"]
    for (ds, lmax), members in sorted(groups.items()):
        m = lambda key: fmt(mean_sd([x.get(key) for x in members])[0], None, 0)
        rows.append(f"{ds} & {lmax} & {members[0]['candidates_kept']} & {m('backbone_s')} & "
                    f"{m('search_s')} & {m('total_s')} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "efficiency_table.tex").write_text("\n".join(rows) + "\n")


def selected_latex(sel_rows) -> None:
    rows = [r"\begin{tabular}{lll}", r"\toprule", r"Dataset & Selector & Paths (seed 0) \\", r"\midrule"]
    seen = {}
    for r in sel_rows:
        if r["seed"] == 0 and r["condition"].endswith(("k5", "k5_L6", "k2")):
            seen.setdefault((r["dataset"], r["condition"]), []).append(r["path"])
    for (ds, cond), paths in sorted(seen.items()):
        rows.append(f"{ds} & {cond.replace('_', ' ')} & {', '.join(paths)} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "selected_paths_table.tex").write_text("\n".join(rows) + "\n")


def plots(proxy, search_list, cond_rows, eff) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(V3, ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    for xkey, fname in (("structural_score", "structural_vs_utility.png"),
                        ("homophily", "homophily_vs_utility.png")):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        for ax, ds in zip(axes, list(DATASETS) + ["ogbn-mag"]):
            pts = [(r[xkey], r["utility_step1"]) for r in proxy
                   if r["dataset"] == ds and r["lmax"] == 4 and r[xkey] is not None and r["utility_step1"] is not None]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=8, alpha=0.6)
                if xkey == "structural_score":
                    ax.set_xscale("log")
            ax.set_title(ds)
            ax.set_xlabel(xkey)
            ax.set_ylabel("RCMS step-1 utility (nats)")
        fig.tight_layout()
        fig.savefig(PLOTS / fname, dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for ds in list(DATASETS) + ["ogbn-mag"]:
        curves = [[s_["utility"] for s_ in s["steps_full"]] for s in search_list
                  if s["dataset"] == ds and s["lmax"] == PRIMARY.get(ds, {"lmax": 4})["lmax"]]
        if curves:
            n = max(len(c) for c in curves)
            mean = [statistics.fmean(c[i] for c in curves if len(c) > i) for i in range(n)]
            ax.plot(range(1, n + 1), mean, marker="o", label=ds)
    ax.set_xlabel("greedy step")
    ax.set_ylabel("marginal utility (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "greedy_utility_curve.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["canonical", "hybrid", "random", "rcms_k5", "rcms_indep_k5", "rcms_nohgt_k5"]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = []
        for ds in DATASETS:
            v = [s["residual_cka"].get(name, {}).get("mean_pairwise_cka") for s in search_list
                 if s["dataset"] == ds and "residual_cka" in s and s["lmax"] == PRIMARY[ds]["lmax"]]
            vals.append(mean_sd(v)[0] or 0)
        ax.bar([j + i * width for j in range(len(DATASETS))], vals, width, label=name)
    ax.set_xticks([j + 0.4 for j in range(len(DATASETS))])
    ax.set_xticklabels(list(DATASETS))
    ax.set_ylabel("mean pairwise residual CKA")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "residual_cka.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in cond_rows:
        if r["test_micro_f1_mean"] is not None and r["train_runtime_s_mean"] is not None:
            ax.scatter(r["train_runtime_s_mean"], r["test_micro_f1_mean"], s=12)
            ax.annotate(f"{r['dataset'][:3]}:{r['condition']}", (r["train_runtime_s_mean"], r["test_micro_f1_mean"]),
                        fontsize=5)
    ax.set_xscale("log")
    ax.set_xlabel("training time per run (s)")
    ax.set_ylabel("test micro-F1 (%)")
    fig.tight_layout()
    fig.savefig(PLOTS / "accuracy_vs_cost.png", dpi=120)
    plt.close(fig)
    sel = {}
    for s in search_list:
        for p in s["selected"]["rcms"].get("5", []):
            key = (s["dataset"], s["lmax"])
            sel.setdefault(key, {}).setdefault(abbr(p), 0)
            sel[key][abbr(p)] += 1
    if sel:
        fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4))
        axes = axes if hasattr(axes, "__len__") else [axes]
        for ax, ((ds, lmax), counts) in zip(axes, sorted(sel.items())):
            items = sorted(counts.items(), key=lambda x: -x[1])[:12]
            ax.barh([i for i, _ in items], [c for _, c in items])
            ax.set_title(f"{ds} L<={lmax}: RCMS k=5 selection count")
            ax.invert_yaxis()
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(PLOTS / "selected_path_frequency.png", dpi=120)
        plt.close(fig)


def reproduction_report(cond_rows) -> None:
    lines = ["# HGT reproduction audit", "",
             "Reference: official HGB code (THUDM/HGB, NC/benchmark/methods/HGT, commit in "
             "`hgb_reference/COMMIT`), scripts `run_acm.sh`, `run_dblp.sh`, `run_freebash.sh`.", "",
             "## HGB reference configuration (from the official scripts and train_hgt.py)", "",
             "| dataset | feats-type | layers | heads | hidden | use_norm | weight decay | schedule |",
             "|---|---|---|---|---|---|---|---|",
             "| ACM | 0 (all given features; featureless types one-hot) | 2 | 8 | 64 | False | 1e-4 | OneCycle max 1e-3, 300 steps |",
             "| DBLP | 2 (target features, others one-hot) | 3 | 8 | 64 | True | 1e-4 | OneCycle max 1e-3, 300 steps |",
             "| Freebase | 2 (all one-hot: no features) | 3 | 8 | 64 | True | 0 | OneCycle max 1e-3, 100 steps |", "",
             "Other HGB facts: AdamW; `--lr` and `--dropout` are parsed but unused (HGTLayer "
             "dropout is fixed at 0.2; OneCycle max_lr is 1e-3); early stopping on validation "
             "loss with patience 30, at most 300 epochs, restoring the best checkpoint; 20% of "
             "the training labels are held out for validation; micro/macro-F1 via sklearn.", "",
             "## Our V1/V2 configuration (ACM)", "",
             "PyG `HGTConv` (softmax over all incoming edges across relations; GELU before the "
             "output projection; per-type LayerNorm after every layer), d=64, 4 heads, 2 layers, "
             "dropout 0.3, AdamW lr 2e-3 constant, weight decay 1e-3, early stopping on "
             "validation accuracy (patience 20, at most 100 epochs), V1 imputed term features.", "",
             "## Differences that remain in V3", "",
             "- Implementation: PyG HGTConv vs DGL HGT-DGL (joint vs per-relation softmax with mean "
             "cross-relation reduction; GELU; input projection without tanh; LayerNorm always on).",
             "- Dropout, learning rate, weight decay, feature type and depth are tuned on validation "
             "(HGB fixed them); heads (8), hidden size (64), OneCycle schedule, val-loss early stopping, "
             "patience 30 and 300 epochs follow HGB.", "",
             "## Validation tuning and chosen configurations", ""]
    for ds in DATASETS:
        rec = _load(V3 / "baseline_audit" / f"chosen_{ds}.json")
        if not rec:
            lines.append(f"- {ds}: tuning not finished")
            continue
        ref = rec.get("hgb_reference_row") or {}
        lines += [f"### {ds}", "", f"- grid: {rec['grid_configs']} configurations, {rec['tuning_runs']} runs "
                  f"(validation only)", f"- chosen: `{json.dumps(rec['chosen'])}` "
                  f"(mean val micro-F1 {100 * rec['chosen_row']['val_micro_f1']:.2f}, "
                  f"{rec['chosen_row']['parameters']} parameters)",
                  f"- HGB reference configuration in our implementation: val micro-F1 "
                  f"{100 * ref['val_micro_f1']:.2f}" if ref else "- HGB reference configuration: not in grid", ""]
        pm = _load(V3 / "baseline_audit" / f"param_match_{ds}.json")
        if pm:
            lines.append(f"- parameter-matched HGT width {pm['hgt_width']} ({pm['hgt_params']} params) "
                         f"vs Auto-PHGT {pm['auto_phgt_reference_params']}")
            lines.append("")
    lines += ["## Final strong-HGT test results (V3 seeds)", "",
              "| dataset | ours micro-F1 | ours macro-F1 | HGB published micro / macro |", "|---|---|---|---|"]
    pub = {"acm": "91.00 / 91.12", "dblp": "93.49 / 93.01", "freebase": "60.51 / 29.28"}
    warn = False
    for r in cond_rows:
        if r["condition"] == "hgt_strong":
            lines.append(f"| {r['dataset']} | {fmt(r['test_micro_f1_mean'], r['test_micro_f1_sd'])} | "
                         f"{fmt(r['test_macro_f1_mean'], r['test_macro_f1_sd'])} | {pub[r['dataset']]} |")
            if r["dataset"] == "acm" and r["test_micro_f1_mean"] is not None \
                    and r["test_micro_f1_mean"] < REPRODUCTION_TARGET["warning_below"]:
                warn = True
    if warn:
        lines += ["", "**BACKBONE_REPRODUCTION_WARNING**: the validation-selected strong HGT is below "
                      f"{REPRODUCTION_TARGET['warning_below']} micro-F1 on ACM; results use it anyway."]
    (V3 / "baseline_audit").mkdir(parents=True, exist_ok=True)
    (V3 / "baseline_audit" / "HGT_REPRODUCTION.md").write_text("\n".join(lines) + "\n")


def aggregate() -> None:
    runs, mag, search_list = hgb_runs(), mag_runs(), searches()
    cond_rows = condition_rows(runs)
    magr = mag_rows(mag)
    comps = comparisons(runs, mag)
    main = cond_rows + magr
    _csv(SUMMARY / "main_results.csv", main)
    _csv(SUMMARY / "selector_results.csv", [r for r in cond_rows if r["group"].startswith("selectors")])
    _csv(SUMMARY / "selector_ablation.csv", [r for r in cond_rows if r["selector"] in ("rcms", "rcms_indep", "rcms_nohgt")
                                             and not r["token_control"]])
    _csv(SUMMARY / "capacity_controls.csv", [r for r in cond_rows if r["group"] in ("backbone", "capacity")
                                             or r["condition"] in ("rcms_k5", "rcms_k5_L6")] + magr)
    sel_rows = selected_path_rows(runs)
    _csv(SUMMARY / "selected_paths.csv", sel_rows)
    proxy = proxy_rows(search_list)
    _csv(SUMMARY / "single_path_or_proxy_scores.csv", proxy)
    stab = stability_rows(search_list)
    _csv(SUMMARY / "search_stability.csv", stab)
    eff = efficiency_rows(runs, mag, search_list)
    _csv(SUMMARY / "efficiency.csv", eff)
    _csv(SUMMARY / "external_baselines.csv", published_rows(runs))
    _csv(V3 / "baselines" / "published_hgb.csv", [r for r in published.rows() if r["source"] == published.HGB])
    _csv(SUMMARY / "statistics.csv", [{**c, "ci95": json.dumps(c["ci95"]), "seeds": json.dumps(c["seeds"])} for c in comps])
    verdicts = hypothesis_verdicts(comps, cond_rows)
    write_reports(cond_rows, magr, comps, verdicts, stab, eff)
    latex(cond_rows, magr, comps)
    efficiency_latex(eff)
    selected_latex(sel_rows)
    reproduction_report(cond_rows)
    try:
        plots(proxy, search_list, cond_rows, eff)
    except Exception as error:  # plots must never block the tables
        (V3 / "logs").mkdir(parents=True, exist_ok=True)
        (V3 / "logs" / "plot_error.txt").write_text(repr(error))
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "hypotheses.json").write_text(json.dumps({"verdicts": verdicts, "outcome": outcome(verdicts)},
                                                        indent=2) + "\n")


if __name__ == "__main__":
    aggregate()
