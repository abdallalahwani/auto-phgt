"""Analysis of the v2 follow-up suite (reads artifacts/v2; safe to run at any time).

    python -m experiments.selector_controls_summary

Writes artifacts/v2/summary/: per-condition tables, paired tests for the pre-registered
hypotheses (Holm-corrected over H2-H4), the random path-set distribution with the canonical
percentile, single-path utility, and correlations of path(-set) properties with accuracy.
Means and sample standard deviations use completed seeds only; missing seeds are listed.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats

from .selector_controls import CONFIG_PATH, V2

SUMMARY = V2 / "summary"
METRICS = ("test_accuracy", "test_macro_f1", "val_accuracy", "best_epoch", "training_runtime_s",
           "parameters")


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _abbr(path) -> str:
    names = {"paper": "P", "author": "A", "subject": "S", "term": "T", "field_of_study": "F",
             "institution": "I"}
    out = [names.get(path[0], path[0])]
    for i in range(1, len(path), 2):
        rel, node = path[i], path[i + 1]
        out.append(f"-{rel}-" if rel in ("cite", "ref", "cites") else "-")
        out.append(names.get(node, node))
    return "".join(out)


def load_runs():
    config = json.loads(CONFIG_PATH.read_text())
    runs = []
    for spec in config["experiments"]:
        result = _load(spec["result"])
        done = bool(result and result.get("status") == "completed"
                    and result.get("experiment_id") == spec["id"])
        row = dict(spec, done=done)
        if done:
            row.update(test_accuracy=result["test"]["accuracy"],
                       test_macro_f1=result["test"]["macro_f1"],
                       val_accuracy=result["validation"]["accuracy"],
                       best_epoch=result["best_epoch"],
                       epochs_completed=result["epochs_completed"],
                       training_runtime_s=result["training_runtime_s"],
                       parameters=result["parameters"], result=result)
        runs.append(row)
    return config, runs


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def condition_row(label, members, seeds):
    done = [m for m in members if m and m["done"]]
    row = {"condition": label, "seeds_planned": len(seeds), "seeds_completed": len(done),
           "missing_seeds": " ".join(str(s) for s, m in zip(seeds, members)
                                     if not (m and m["done"]))}
    for metric in METRICS:
        row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std([m.get(metric) for m in done])
    return row


def paired(a: dict, b: dict, metric="test_accuracy"):
    """Paired difference a - b over seeds where both completed, with a 95% t-interval."""
    seeds = sorted(s for s in a if s in b and a[s]["done"] and b[s]["done"])
    diffs = [a[s][metric] - b[s][metric] for s in seeds]
    out = {"n": len(diffs), "seeds": seeds, "per_seed": diffs, "mean": None, "ci95": None,
           "p": None, "wins": sum(d > 0 for d in diffs)}
    if len(diffs) >= 2:
        mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
        half = stats.t.ppf(0.975, len(diffs) - 1) * sd / math.sqrt(len(diffs))
        out.update(mean=mean, ci95=[mean - half, mean + half],
                   p=float(stats.ttest_1samp(diffs, 0.0).pvalue) if sd > 0 else 0.0)
    elif diffs:
        out["mean"] = diffs[0]
    return out


def holm(pvalues: dict) -> dict:
    items = sorted((p, k) for k, p in pvalues.items() if p is not None)
    adjusted, running = {}, 0.0
    for i, (p, key) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        adjusted[key] = running
    return adjusted


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None, "p": None}
    rho, p = stats.spearmanr([x for x, _ in pairs], [y for _, y in pairs])
    return {"n": len(pairs), "rho": float(rho), "p": float(p)}


def by_seed(runs, predicate):
    return {r["seed"]: r for r in runs if predicate(r)}


def acm_tables(runs):
    acm = [r for r in runs if r["part"] == "acm"]
    ids = {r["id"]: r for r in acm}
    seeds10 = list(range(10))

    def cond(template):
        return [ids.get(template.format(seed=s)) for s in seeds10]

    selectors = {}
    for k in (5, 2):
        rows = []
        names = ["discovered", "random", "diverse", "homophily", "hybrid"] + (
            ["manual"] if k == 2 else [])
        for name in names:
            rows.append(condition_row(name, cond(f"v2_acm_auto_phgt_seed{{seed}}_{name}_k{k}"),
                                      seeds10))
        selectors[f"k{k}"] = rows
    controls = []
    for label, template in (
            ("HGT", "v2_acm_hgt_seed{seed}_discovered_k5"),
            ("HGT d=68 (param-matched)", "v2_acm_hgt_d68_seed{seed}_discovered_k5"),
            ("Tokens", "v2_acm_tokens_seed{seed}_discovered_k5"),
            ("Auto-PHGT (canonical paths)", "v2_acm_auto_phgt_seed{seed}_discovered_k5"),
            ("Auto-PHGT dummy tokens", "v2_acm_auto_phgt_dummy_seed{seed}_discovered_k5"),
            ("Auto-PHGT shuffled tokens (canonical)",
             "v2_acm_auto_phgt_shuffle_seed{seed}_discovered_k5"),
            ("Auto-PHGT (random paths)", "v2_acm_auto_phgt_seed{seed}_random_k5"),
            ("Auto-PHGT shuffled tokens (random)", "v2_acm_auto_phgt_shuffle_seed{seed}_random_k5")):
        controls.append(condition_row(label, cond(template), seeds10))

    def s(template):
        return {seed: ids[template.format(seed=seed)] for seed in seeds10
                if template.format(seed=seed) in ids}

    disc = s("v2_acm_auto_phgt_seed{seed}_discovered_k5")
    comparisons = {
        "H2 diverse - discovered (k5)": paired(s("v2_acm_auto_phgt_seed{seed}_diverse_k5"), disc),
        "H3a homophily - discovered (k5)": paired(
            s("v2_acm_auto_phgt_seed{seed}_homophily_k5"), disc),
        "H3b homophily - random (k5)": paired(s("v2_acm_auto_phgt_seed{seed}_homophily_k5"),
                                              s("v2_acm_auto_phgt_seed{seed}_random_k5")),
        "H4 discovered - dummy tokens": paired(
            disc, s("v2_acm_auto_phgt_dummy_seed{seed}_discovered_k5")),
        "random - discovered (k5)": paired(s("v2_acm_auto_phgt_seed{seed}_random_k5"), disc),
        "hybrid - discovered (k5)": paired(s("v2_acm_auto_phgt_seed{seed}_hybrid_k5"), disc),
        "diverse - random (k5)": paired(s("v2_acm_auto_phgt_seed{seed}_diverse_k5"),
                                        s("v2_acm_auto_phgt_seed{seed}_random_k5")),
        "discovered - HGT": paired(disc, s("v2_acm_hgt_seed{seed}_discovered_k5")),
        "discovered - HGT d68": paired(disc, s("v2_acm_hgt_d68_seed{seed}_discovered_k5")),
        "discovered - shuffled (canonical)": paired(
            disc, s("v2_acm_auto_phgt_shuffle_seed{seed}_discovered_k5")),
        "random - shuffled (random)": paired(
            s("v2_acm_auto_phgt_seed{seed}_random_k5"),
            s("v2_acm_auto_phgt_shuffle_seed{seed}_random_k5")),
        "random paths - HGT": paired(s("v2_acm_auto_phgt_seed{seed}_random_k5"),
                                     s("v2_acm_hgt_seed{seed}_discovered_k5")),
        "manual - discovered (k2)": paired(s("v2_acm_auto_phgt_seed{seed}_manual_k2"),
                                           s("v2_acm_auto_phgt_seed{seed}_discovered_k2")),
        "manual - random (k2)": paired(s("v2_acm_auto_phgt_seed{seed}_manual_k2"),
                                       s("v2_acm_auto_phgt_seed{seed}_random_k2")),
    }
    adjusted = holm({key: comparisons[key]["p"] for key in (
        "H2 diverse - discovered (k5)", "H3a homophily - discovered (k5)",
        "H4 discovered - dummy tokens")})
    for key, value in adjusted.items():
        comparisons[key]["p_holm"] = value
    return selectors, controls, comparisons


def run_properties(result: dict) -> dict:
    """Path-set properties from one run's own records (homophily is split-specific)."""
    discovery = result["discovery"]
    paths = [p["path"] for p in discovery["paths"]]
    table = {tuple(r["path"]): r for r in (discovery.get("homophily") or {}).get("paths", [])}
    families = [set(p["families"]) for p in discovery["paths"]]
    union = set().union(*families) if families else set()
    pairs = [(i, j) for i in range(len(families)) for j in range(i + 1, len(families))]
    jaccard = [len(families[i] & families[j]) / len(families[i] | families[j]) for i, j in pairs]
    stats_rows = result.get("path_statistics") or []
    homophily = [table.get(tuple(p), {}).get("homophily") for p in paths]
    return {
        "n_author_paths": sum("author" in p for p in paths),
        "n_subject_paths": sum("subject" in p for p in paths),
        "n_term_paths": sum("term" in p for p in paths),
        "n_citation_paths": sum(any(r in ("cite", "ref") for r in p[1::2]) for p in paths),
        "family_coverage": len(union),
        "redundancy": statistics.fmean(jaccard) if jaccard else 0.0,
        "mean_log_structural_score": statistics.fmean(
            math.log(p["score"]) for p in discovery["paths"]),
        "mean_canonical_rank": statistics.fmean(p["canonical_rank"] for p in discovery["paths"]),
        "mean_homophily": (statistics.fmean(h for h in homophily if h is not None)
                           if any(h is not None for h in homophily) else None),
        "mean_completion": (statistics.fmean(r["complete_rate"] for r in stats_rows)
                            if stats_rows else None),
        "paths": " | ".join(_abbr(p) for p in paths),
    }


def random_sets(runs, canonical_rows):
    sets = {}
    for r in runs:
        if r["group"] == "acm_random_sets" and r["done"]:
            sets.setdefault(r["path_seed"], []).append(r)
    rows = []
    for path_seed, members in sorted(sets.items()):
        props = run_properties(members[0]["result"])
        props.pop("mean_homophily")
        homophily = [run_properties(m["result"])["mean_homophily"] for m in members]
        completion = [run_properties(m["result"])["mean_completion"] for m in members]
        rows.append({"random_set": path_seed - 1000, "path_seed": path_seed,
                     "seeds_completed": len(members),
                     "test_accuracy_mean": statistics.fmean(m["test_accuracy"] for m in members),
                     "val_accuracy_mean": statistics.fmean(m["val_accuracy"] for m in members),
                     **props,
                     "mean_homophily": mean_std(homophily)[0],
                     "mean_completion": mean_std(completion)[0]})
    complete = [r for r in rows if r["seeds_completed"] == 3]
    canonical = [c for c in canonical_rows if c["done"] and c["seed"] in (0, 1, 2)]
    summary = {"random_sets_complete": len(complete), "random_sets_partial": len(rows)}
    if complete and len(canonical) == 3:
        for split in ("test", "val"):
            value = statistics.fmean(c[f"{split}_accuracy"] for c in canonical)
            others = [r[f"{split}_accuracy_mean"] for r in complete]
            below = sum(o < value for o in others) + 0.5 * sum(o == value for o in others)
            summary[split] = {"canonical_mean_seeds_0_2": value,
                              "canonical_percentile": 100.0 * below / len(others),
                              "random_min": min(others), "random_median": statistics.median(others),
                              "random_max": max(others)}
        props = ("n_author_paths", "n_subject_paths", "n_term_paths", "n_citation_paths",
                 "family_coverage", "redundancy", "mean_log_structural_score",
                 "mean_canonical_rank", "mean_homophily", "mean_completion")
        summary["spearman_vs_test_accuracy"] = {
            p: spearman([r[p] for r in complete], [r["test_accuracy_mean"] for r in complete])
            for p in props}
    return rows, summary


def single_paths(runs):
    per_rank = {}
    for r in runs:
        if r["group"] == "acm_single_path" and r["done"]:
            per_rank.setdefault(r["ranks"][0], []).append(r)
    rows = []
    for rank, members in sorted(per_rank.items()):
        props = [run_properties(m["result"]) for m in members]
        path = members[0]["result"]["discovery"]["paths"][0]
        rows.append({"canonical_rank": rank, "path": _abbr(path["path"]),
                     "structural_score": path["score"], "seeds_completed": len(members),
                     "test_accuracy_mean": statistics.fmean(m["test_accuracy"] for m in members),
                     "val_accuracy_mean": statistics.fmean(m["val_accuracy"] for m in members),
                     "homophily": mean_std([p["mean_homophily"] for p in props])[0],
                     "completion": mean_std([p["mean_completion"] for p in props])[0]})
    complete = [r for r in rows if r["seeds_completed"] == 3]
    summary = {"paths_complete": len(complete), "paths_partial": len(rows)}
    if len(complete) >= 3:
        summary["spearman_vs_test_accuracy"] = {
            p: spearman([r[p] for r in complete], [r["test_accuracy_mean"] for r in complete])
            for p in ("structural_score", "homophily", "completion")}
    return rows, summary


def mag_tables(runs):
    mag = {r["id"]: r for r in runs if r["part"] == "mag"}
    seeds = [0, 1, 2]

    def s(template):
        return {seed: mag[template.format(seed=seed)] for seed in seeds}

    groups = {
        "HGT (300-epoch cap)": s("v2_mag_hgt_seed{seed}_discovered_k5_e300"),
        "Auto-PHGT canonical (300)": s("v2_mag_auto_phgt_seed{seed}_discovered_k5_e300"),
        "Auto-PHGT random (300)": s("v2_mag_auto_phgt_seed{seed}_random_k5_e300"),
        "Auto-PHGT dummy tokens (300)": s("v2_mag_auto_phgt_dummy_seed{seed}_discovered_k5_e300"),
    }
    rows = [condition_row(label, [members[seed] for seed in seeds], seeds)
            for label, members in groups.items()]
    for row, members in zip(rows, groups.values()):
        row["epochs_completed"] = " ".join(str(m.get("epochs_completed", "-"))
                                           for m in members.values())
    comparisons = {
        "H5 canonical - random": paired(groups["Auto-PHGT canonical (300)"],
                                        groups["Auto-PHGT random (300)"]),
        "canonical - HGT": paired(groups["Auto-PHGT canonical (300)"],
                                  groups["HGT (300-epoch cap)"]),
        "canonical - dummy": paired(groups["Auto-PHGT canonical (300)"],
                                    groups["Auto-PHGT dummy tokens (300)"]),
    }
    return rows, comparisons


def aggregate() -> dict:
    config, runs = load_runs()
    selectors, controls, comparisons = acm_tables(runs)
    canonical = [r for r in runs if r["group"] == "acm_selectors_k5"
                 and r["path_selection"] == "discovered"]
    set_rows, set_summary = random_sets(runs, canonical)
    path_rows, path_summary = single_paths(runs)
    mag_rows, mag_comparisons = mag_tables(runs)
    for name, rows in (("acm_selectors_k5", selectors["k5"]), ("acm_selectors_k2", selectors["k2"]),
                       ("acm_controls", controls), ("acm_random_sets", set_rows),
                       ("acm_single_paths", path_rows), ("mag", mag_rows)):
        _write_csv(SUMMARY / f"{name}.csv", rows)
    progress = {}
    for r in runs:
        entry = progress.setdefault(r["group"], {"done": 0, "planned": 0})
        entry["planned"] += 1
        entry["done"] += r["done"]
    summary = {"suite": config["suite"], "frozen_at": config["frozen_at"],
               "progress": progress, "acm_selectors": selectors, "acm_controls": controls,
               "acm_paired": comparisons, "acm_random_sets": set_summary,
               "acm_single_paths": path_summary, "mag": mag_rows, "mag_paired": mag_comparisons}
    (SUMMARY / "v2_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


if __name__ == "__main__":
    aggregate()
