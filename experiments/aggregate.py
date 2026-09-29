"""CPU-only aggregation of the final suite (standard library only; no torch).

Reads the frozen inventory, the final result JSONs, per-experiment status files and the
checkpoint progress sidecars, and writes condition tables (mean ± sample std across
completed seeds, with explicit completed/failed/unfinished seed lists) plus raw CSVs for
learning curves, path analyses and MAG batch statistics.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ARTIFACTS = Path("artifacts")

METRICS = ("test_accuracy", "test_macro_f1", "val_accuracy", "val_macro_f1", "best_epoch",
           "epochs_completed", "training_runtime_s", "total_runtime_s", "parameters",
           "peak_gpu_allocated_mib")

TABLES = {
    "acm_main": [("HGT", "acm", "hgt", "discovered", 5, range(5)),
                 ("Tokens", "acm", "tokens", "discovered", 5, range(5)),
                 ("Auto-PHGT", "acm", "auto_phgt", "discovered", 5, range(5))],
    "acm_discovery_ablation": [
        ("Auto-PHGT discovered", "acm", "auto_phgt", "discovered", 5, range(5)),
        ("Auto-PHGT random", "acm", "auto_phgt", "random", 5, range(5))],
    "acm_k_sensitivity": [
        ("k=1", "acm", "auto_phgt", "discovered", 1, range(3)),
        ("k=3", "acm", "auto_phgt", "discovered", 3, range(3)),
        ("k=5 (seeds 0-2, matched)", "acm", "auto_phgt", "discovered", 5, range(3)),
        ("k=5 (seeds 0-4)", "acm", "auto_phgt", "discovered", 5, range(5))],
    "mag_main": [("HGT", "ogbn-mag", "hgt", "discovered", 5, range(3)),
                 ("Auto-PHGT", "ogbn-mag", "auto_phgt", "discovered", 5, range(3))],
}

PAIRED = {  # (table, minuend label, subtrahend label)
    "acm_main": [("Auto-PHGT", "HGT"), ("Auto-PHGT", "Tokens"), ("Tokens", "HGT")],
    "acm_discovery_ablation": [("Auto-PHGT discovered", "Auto-PHGT random")],
    "mag_main": [("Auto-PHGT", "HGT")],
}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
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


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def run_rows(config: dict, artifacts: Path) -> list[dict]:
    rows = []
    for spec in config["experiments"]:
        result = _load(Path(spec["result"]))
        if result and not (result.get("status") == "completed"
                           and result.get("experiment_id") == spec["id"]):
            result = None
        status = _load(artifacts / "status" / f"{spec['id']}.json") or {}
        checkpoint = Path(spec["checkpoint"])
        progress = _load(checkpoint.with_name(checkpoint.stem + ".progress.json")) or {}
        if result:
            state = "completed"
        else:
            state = status.get("status") or ("interrupted" if progress else "pending")
        row = {key: spec[key] for key in ("id", "group", "priority", "dataset", "mode", "seed",
                                          "path_selection", "k")}
        row.update(state=state, attempts=len(status.get("attempts", [])),
                   last_epoch=progress.get("epoch"),
                   best_val_accuracy_so_far=progress.get("best_val_accuracy"))
        if result:
            env = result.get("environment", {})
            row.update(test_accuracy=result["test"]["accuracy"],
                       test_macro_f1=result["test"]["macro_f1"],
                       val_accuracy=result["validation"]["accuracy"],
                       val_macro_f1=result["validation"]["macro_f1"],
                       best_epoch=result["best_epoch"],
                       epochs_completed=result["epochs_completed"],
                       early_stopped=result["early_stopped"],
                       training_runtime_s=result["training_runtime_s"],
                       total_runtime_s=result.get("total_runtime_s"),
                       parameters=result["parameters"],
                       peak_gpu_allocated_mib=result.get("gpu_memory", {}).get(
                           "max_allocated_mib"),
                       peak_cpu_rss_mib=result.get("peak_cpu_rss_mib"),
                       host=env.get("host"), gpu_name=env.get("gpu_name"),
                       gpu_uuid=env.get("gpu_uuid"),
                       slurm_gpu_id=(result.get("assigned_gpu") or {}).get("slurm_gpu_id"),
                       git_commit=(env.get("git") or {}).get("commit"),
                       resumed_from_epoch=result.get("resumed_from_epoch"))
        elif status.get("attempts"):
            row["last_error"] = (status["attempts"][-1].get("error_tail") or "")[-500:]
        rows.append(row)
    return rows


def condition_tables(rows: list[dict]) -> dict:
    by_key = {(r["dataset"], r["mode"], r["path_selection"], r["k"], r["seed"]): r
              for r in rows}
    tables = {}
    for name, conditions in TABLES.items():
        table = []
        for label, dataset, mode, selection, k, seeds in conditions:
            members = [by_key.get((dataset, mode, selection, k, seed)) for seed in seeds]
            done = [m for m in members if m and m["state"] == "completed"]
            entry = {"condition": label, "dataset": dataset, "mode": mode,
                     "path_selection": selection, "k": k,
                     "seeds_planned": len(members), "seeds_completed": len(done),
                     "completed_seeds": " ".join(str(m["seed"]) for m in done),
                     "failed_seeds": " ".join(str(m["seed"]) for m in members
                                              if m and m["state"] == "failed"),
                     "unfinished_seeds": " ".join(
                         f"{m['seed']}({m['state']}"
                         + (f", epoch {m['last_epoch']}" if m.get("last_epoch") else "") + ")"
                         for m in members if m and m["state"] not in ("completed", "failed"))}
            for metric in METRICS:
                mean, std = _mean_std([m.get(metric) for m in done])
                entry[f"{metric}_mean"], entry[f"{metric}_std"] = mean, std
            table.append(entry)
        tables[name] = table
    return tables


def paired_differences(rows: list[dict]) -> dict:
    """Per-seed differences for seeds where both conditions completed."""
    by_key = {(r["dataset"], r["mode"], r["path_selection"], r["k"], r["seed"]): r
              for r in rows}
    out = {}
    for table, pairs in PAIRED.items():
        conditions = {c[0]: c for c in TABLES[table]}
        for left, right in pairs:
            _, dataset, mode_a, sel_a, k_a, seeds = conditions[left]
            _, _, mode_b, sel_b, k_b, _ = conditions[right]
            diffs = {}
            for metric in ("test_accuracy", "test_macro_f1"):
                values = []
                for seed in seeds:
                    a = by_key.get((dataset, mode_a, sel_a, k_a, seed))
                    b = by_key.get((dataset, mode_b, sel_b, k_b, seed))
                    if a and b and a["state"] == b["state"] == "completed":
                        values.append((seed, a[metric] - b[metric]))
                mean, std = _mean_std([v for _, v in values])
                diffs[metric] = {"seeds": [s for s, _ in values],
                                 "per_seed": [v for _, v in values], "mean": mean, "std": std}
            out[f"{table}: {left} - {right}"] = diffs
    return out


def raw_tables(config: dict) -> dict:
    histories, paths, rankings, completion, batches = [], [], [], [], []
    seen_rankings = set()
    for spec in config["experiments"]:
        result = _load(Path(spec["result"]))
        if not result or result.get("experiment_id") != spec["id"]:
            continue
        base = {key: spec[key] for key in ("id", "dataset", "mode", "seed", "path_selection",
                                           "k")}
        for row in result.get("history", []):
            histories.append(base | row)
        discovery = result.get("discovery", {})
        for row in discovery.get("paths", []):
            paths.append(base | {"rank": row["rank"], "path": " -> ".join(row["path"]),
                                 "canonical_score": row["score"],
                                 "canonical_rank": row.get("canonical_rank"),
                                 "candidate_count": discovery.get("candidate_count")})
        if spec["dataset"] not in seen_rankings and discovery.get("canonical_ranking"):
            seen_rankings.add(spec["dataset"])
            for row in discovery["canonical_ranking"]:
                rankings.append({"dataset": spec["dataset"], "canonical_rank": row["canonical_rank"],
                                 "path": " -> ".join(row["path"]), "score": row["score"]})
        for row in result.get("path_statistics") or []:
            completion.append(base | row)
        for row in result.get("sampled_batch_statistics") or []:
            flat = {"batch": row["batch"], "seeds": row["seeds"], "sample_s": row["sample_s"]}
            flat |= {f"nodes_{t}": n for t, n in row["nodes"].items()}
            flat |= {f"edges_{t}": n for t, n in row["edges"].items()}
            batches.append(base | flat)
    return {"histories": histories, "selected_paths": paths, "canonical_rankings": rankings,
            "path_statistics": completion, "mag_batch_statistics": batches}


def aggregate(artifacts: Path = ARTIFACTS, out_dir: Path | None = None) -> dict:
    config = json.loads((artifacts / "final_suite_config.json").read_text())
    out_dir = out_dir or artifacts / "summary"
    rows = run_rows(config, artifacts)
    tables = condition_tables(rows)
    raw = raw_tables(config)
    for name, table in tables.items():
        _write_csv(out_dir / f"{name}.csv", table)
    _write_csv(out_dir / "runs.csv", rows)
    for name, table in raw.items():
        if table:
            _write_csv(out_dir / "raw" / f"{name}.csv", table)
    states = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    summary = {"suite": config["suite"], "git": config.get("git"),
               "experiments_planned": len(rows), "states": states,
               "tables": tables, "paired_differences": paired_differences(rows),
               "runs": rows,
               "random_paths": [r for r in raw["selected_paths"]
                                if r["path_selection"] == "random"],
               "canonical_rankings": raw["canonical_rankings"],
               "note": ("mean/std over completed seeds only; std is the sample standard "
                        "deviation (None for one seed); unfinished and failed seeds are "
                        "listed per condition and never silently dropped")}
    path = out_dir / "final_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


if __name__ == "__main__":
    aggregate()
