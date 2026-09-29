"""Completion-aware result tables; reused baselines retain their original provenance."""

from __future__ import annotations

import csv
import fcntl
import statistics
from collections import Counter
from pathlib import Path

from auto_phgt.runtime import write_json
from experiments.lightweight_selection.stats import paired

from . import plan


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def aggregate() -> dict:
    lock = plan.check_lock()
    folder = plan.OUT / "summary"
    folder.mkdir(parents=True, exist_ok=True)
    with (plan.OUT / "aggregate.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        runs, missing, grouped = [], [], {}
        for spec in plan.tasks():
            record = plan.final_record(spec, lock)
            if record is None:
                missing.append(spec["id"])
                continue
            source = record.get("reused_from") or str(plan.result_path(spec).relative_to(plan.ROOT))
            runs.append({**spec, "source": source, "reused": bool(record.get("reused_from")),
                         "micro_f1": record["test"]["micro_f1"],
                         "macro_f1": record["test"]["macro_f1"],
                         "gpu": record["environment"].get("gpu_name", "cpu"),
                         "parameters": record["parameters"],
                         "gpu_peak_mib": record["fit"].get("gpu_memory", {}).get("max_allocated_mib"),
                         "train_runtime_s": record["fit"]["train_runtime_s"]})
            grouped.setdefault((spec["dataset"], spec["method"], spec["k"]), {})[spec["seed"]] = record
        rows = []
        for (ds, method, k), by_seed in grouped.items():
            micro = [100 * record["test"]["micro_f1"] for record in by_seed.values()]
            macro = [100 * record["test"]["macro_f1"] for record in by_seed.values()]
            complete = set(by_seed) == set(plan.SEEDS)
            rows.append({"dataset": ds, "method": method, "k": k, "lmax": plan.lmax(ds, k),
                         "n": len(by_seed), "complete": complete,
                         "seeds": " ".join(map(str, sorted(by_seed))),
                         "micro_f1_mean": statistics.mean(micro),
                         "micro_f1_sd": statistics.stdev(micro) if len(micro) > 1 else None,
                         "macro_f1_mean": statistics.mean(macro),
                         "macro_f1_sd": statistics.stdev(macro) if len(macro) > 1 else None})
        comparisons = []
        for ds in plan.DATASETS:
            for k in plan.KS:
                a = grouped.get((ds, "edgeoverlap", k), {})
                for baseline in ("hgt", "random", "canonical", "hybrid", "fastpath"):
                    b = grouped.get((ds, baseline, None if baseline == "hgt" else k), {})
                    for metric in ("micro_f1", "macro_f1"):
                        stats = paired({s: r["test"][metric] for s, r in a.items()},
                                       {s: r["test"][metric] for s, r in b.items()})
                        comparisons.append({"dataset": ds, "method": "edgeoverlap", "baseline": baseline,
                                            "k": k, "metric": metric, "n": stats["n"],
                                            "complete": stats["n"] == len(plan.SEEDS),
                                            "mean_difference_points": stats["mean"],
                                            "ci_low": stats["ci95"][0] if stats["ci95"] else None,
                                            "ci_high": stats["ci95"][1] if stats["ci95"] else None,
                                            "permutation_p": stats["p_perm"],
                                            "wins": stats["wins"], "losses": stats["losses"]})
        secondary = []
        for k in plan.KS:
            actual = grouped.get(("acm", "edgeoverlap", k), {})
            for baseline in ("ti", "ti_set"):
                old = {}
                for seed in plan.SEEDS:
                    reference = lock["secondary_acm_references"][f"acm__{baseline}_k{k}__seed{seed}"]
                    path = plan.ROOT / reference["path"]
                    if plan.digest(path) != reference["sha256"]:
                        raise ValueError(f"secondary reference changed: {path}")
                    old[seed] = plan.read_json(path)
                result = paired({s: r["test"]["micro_f1"] for s, r in actual.items()},
                                {s: r["test"]["micro_f1"] for s, r in old.items()})
                secondary.append({"dataset": "acm", "k": k, "baseline": baseline,
                                  "complete": result["n"] == 5, **result})
        write_csv(folder / "runs.csv", runs)
        write_csv(folder / "main_results.csv", rows)
        write_csv(folder / "paired_statistics.csv", comparisons)
        efficiency = []
        for path in sorted((plan.OUT / "selectors").glob("*/*.json")):
            cached = plan.read_json(path)
            if cached.get("protocol_hash") != lock["protocol_hash"]:
                raise ValueError(f"selector cache belongs to another protocol: {path}")
            efficiency.append({"dataset": cached["dataset"],
                               "cache": str(path.relative_to(plan.ROOT)),
                               "selectors": " ".join(cached["selections"]),
                               "lmax": cached["lmax"], "seed": cached.get("seed"),
                               "preparation_runtime_s": cached["runtime_s"],
                               "gpu_s": cached["gpu_s"]})
        write_csv(folder / "selector_efficiency.csv", efficiency)
        report = {"protocol_hash": lock["protocol_hash"], "complete": not missing,
                  "completed": len(runs), "reused": sum(row["reused"] for row in runs),
                  "missing": missing, "by_dataset": dict(Counter(row["dataset"] for row in runs)),
                  "main": rows, "paired": comparisons, "secondary_acm": secondary}
        write_json(folder / "summary.json", report)
        lines = ["# Final ACM / DBLP / IMDB comparison", "",
                 f"Protocol `{lock['protocol_hash']}`. Complete: **{not missing}**.",
                 f"Completed cells: {len(runs)}/165; reused: {report['reused']}; missing: {len(missing)}.",
                 "", "Five fixed seeds. HGT is shared between both k tables, not trained twice.",
                 "IMDB is multi-label: micro-F1 is not accuracy; threshold is fixed at sigmoid >= 0.5.",
                 "DBLP k=2 uses L<=4; k=5 uses L<=6, preserving existing results.",
                 "Incomplete rows and intervals are descriptive only, never confirmation claims.",
                 "ACM and DBLP include reused, previously inspected results; do not present them as untouched test sets.",
                 "Runtime comparisons mix GPU hardware and must retain the per-run device provenance.", ""]
        for ds in plan.DATASETS:
            for k in plan.KS:
                lines += [f"## {ds.upper()}, k={k}", "",
                          "| Method | n | Micro-F1 (%) | Macro-F1 (%) | Complete |",
                          "|---|---:|---:|---:|---|"]
                for method in plan.METHODS:
                    row = next((item for item in rows if item["dataset"] == ds and
                                item["method"] == method and item["k"] == (None if method == "hgt" else k)), None)
                    if row is None:
                        lines.append(f"| {method} | 0 | pending | pending | False |")
                    else:
                        micro_sd = "n/a" if row["micro_f1_sd"] is None else f"{row['micro_f1_sd']:.2f}"
                        macro_sd = "n/a" if row["macro_f1_sd"] is None else f"{row['macro_f1_sd']:.2f}"
                        lines.append(f"| {method} | {row['n']} | {row['micro_f1_mean']:.2f} +/- {micro_sd} | "
                                     f"{row['macro_f1_mean']:.2f} +/- {macro_sd} | {row['complete']} |")
                lines.append("")
        lines += ["## EdgeOverlap-specific interpretation", "",
                  "Existing ACM TI_cov and TI-Set results are reused as zero-training-cost secondary controls.",
                  "No new TI-Set runs are scheduled on DBLP/IMDB; those datasets cannot directly establish",
                  "EdgeOverlap superiority to fingerprint novelty without that comparator.", ""]
        temp = folder / "final_summary.tmp"
        temp.write_text("\n".join(lines))
        temp.replace(folder / "final_summary.md")
    return report


if __name__ == "__main__":
    result = aggregate()
    print(f"V5 aggregate: {result['completed']}/165 complete; reused={result['reused']}")
