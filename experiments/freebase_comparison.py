"""Frozen Freebase-only EdgeOverlap extension, matched to completed V4 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from auto_phgt.runtime import device_info, to_device, write_json
from experiments.residual_selection import data, train
from experiments.lightweight_selection import tasks as v4
from experiments.method_comparison import edge_overlap, plan as v5
from experiments.lightweight_selection.stats import paired

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/freebase_edgeoverlap"
LOCK = OUT / "protocol_lock.json"
SELECTED = OUT / "selection.json"
SEEDS = range(5)
KS = (2, 5)
METHODS = ("hgt", "canonical", "random", "hybrid", "fastpath", "ti", "ti_set")
SOURCE = ROOT / "artifacts/v4_hedge"
DATASET = ROOT / "data/hgb/freebase/freebase/processed/data.pt"
TI = SOURCE / "selectors/transition_info/freebase.json"
PYTHON = ROOT / ".venv/bin/python"


def hash_file(path: Path) -> str:
    return v5.digest(path)


def result_path(k: int, seed: int) -> Path:
    return OUT / "results" / f"edgeoverlap_k{k}_seed{seed}.json"


def existing_path(method: str, k: int, seed: int) -> Path:
    suffix = "hgt" if method == "hgt" else f"{method}_k{k}"
    return SOURCE / "results/freebase" / f"freebase__{suffix}__seed{seed}.json"


def code_hashes() -> dict:
    paths = [
        Path(__file__), ROOT / "experiments/method_comparison/edge_overlap.py",
        ROOT / "experiments/method_comparison/plan.py", ROOT / "experiments/residual_selection/data.py",
        ROOT / "experiments/residual_selection/train.py", ROOT / "experiments/residual_selection/plan.py",
        ROOT / "experiments/lightweight_selection/tasks.py", ROOT / "experiments/lightweight_selection/plan.py",
        ROOT / "experiments/lightweight_selection/stats.py",
    ]
    paths += sorted((ROOT / "auto_phgt").glob("*.py"))
    return {str(path.relative_to(ROOT)): hash_file(path) for path in paths}


def freeze() -> dict:
    if LOCK.exists():
        raise FileExistsError(f"{LOCK} already exists: refusing to change the frozen protocol")
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    parent = SOURCE / "protocol/protocol_lock.json"
    parent_lock = json.loads(parent.read_text())
    for relative, expected in parent_lock["code"].items():
        if hash_file(ROOT / relative)[:len(expected)] != expected:
            raise ValueError(f"V4 dependency changed: {relative}")
    graph = data.load("freebase")
    paths, _ = v4.candidates(graph, "book")
    source_ti = json.loads(TI.read_text())
    if [row["path"] for row in source_ti["candidates"]] != paths or len(paths) != 14:
        raise ValueError("V4 TI_cov candidate space does not match current Freebase graph")
    config = v4.full_cfg(v4.chosen_config("freebase"))
    references = {}
    for method in METHODS:
        for k in ((5,) if method == "hgt" else KS):
            for seed in SEEDS:
                path = existing_path(method, k, seed)
                record = json.loads(path.read_text())
                if (record["status"] != "completed" or record["config"] != config
                    or record["seed"] != seed or record["dataset"] != "freebase"
                    or record["k"] != (None if method == "hgt" else k)
                    or record["lmax"] != 4
                    or record["selector"] != (None if method == "hgt" else method)):
                    raise ValueError(f"incompatible V4 comparison: {path}")
                references[str(path.relative_to(ROOT))] = hash_file(path)
    values = {"dataset": "freebase", "source": "V4-Hedge", "parent_lock_sha256": hash_file(parent),
              "processed_sha256": hash_file(DATASET), "ti_sha256": hash_file(TI),
              "config": config, "references": references, "code": code_hashes(),
              "seeds": list(SEEDS), "ks": list(KS), "max_hops": 4,
              "source_nodes": 1024, "instances_per_path": 32, "sample_seed": 0,
              "edge_overlap_rule": "mean per-source typed directed edge Jaccard on complete sampled walks; empty union=0",
              "greedy_rule": "TI_cov(p)*(1-max_{q selected} O(p,q)); stable canonical rank",
              "research_note": "Freebase test results from V4 were already inspected. This is a post-hoc extension, not clean confirmation.",
              "validation_rule": "V4 validation-tuned configuration reused without any test-driven adjustment"}
    values["protocol_hash"] = hashlib.sha256(
        json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]
    write_json(LOCK, values)
    return values


def check_lock(*, verify_sources: bool = False) -> dict:
    if not LOCK.exists():
        raise FileNotFoundError(f"freeze protocol first: {LOCK}")
    record = json.loads(LOCK.read_text())
    if record["code"] != code_hashes():
        raise ValueError("Freebase extension code changed after freeze")
    if verify_sources:
        if record["processed_sha256"] != hash_file(DATASET):
            raise ValueError("Freebase dataset changed")
        if record["parent_lock_sha256"] != hash_file(SOURCE / "protocol/protocol_lock.json"):
            raise ValueError("V4 parent protocol changed")
        if record["ti_sha256"] != hash_file(TI):
            raise ValueError("Freebase TI_cov score file changed")
        for name, digest in record["references"].items():
            if hash_file(ROOT / name) != digest:
                raise ValueError(f"V4 comparison changed: {name}")
    return record


def selection(lock: dict) -> dict:
    existing = json.loads(SELECTED.read_text()) if SELECTED.exists() else None
    if existing is not None:
        if existing.get("protocol_hash") != lock["protocol_hash"]:
            raise ValueError("selection belongs to another protocol")
        return existing
    graph = data.load("freebase")
    paths, _ = v4.candidates(graph, "book")
    source = json.loads(TI.read_text())
    if [row["path"] for row in source["candidates"]] != paths:
        raise ValueError("candidate ranking differs from frozen TI scores")
    scores = [row["TI_cov"] for row in source["candidates"]]
    started = time.perf_counter()
    matrix, metadata = edge_overlap.edge_overlap_matrix(
        graph, paths, max_sources=lock["source_nodes"],
        instances_per_path=lock["instances_per_path"], seed=lock["sample_seed"])
    indices, steps = edge_overlap.greedy_edge_overlap(scores, matrix, max(KS))
    result = {"protocol_hash": lock["protocol_hash"], "paths": paths,
              "ti_cov": scores, "overlap": matrix.tolist(), "sampling": metadata,
              "indices": indices, "steps": steps, "selected": {
                  str(k): [paths[i] for i in indices[:k]] for k in KS},
              "selection_runtime_s": time.perf_counter() - started, "gpu_s": 0}
    write_json(SELECTED, result)
    return result


def one(k: int, seed: int, *, pilot: bool = False) -> dict:
    lock = check_lock()
    output = result_path(k, seed)
    if not pilot and output.exists():
        record = json.loads(output.read_text())
        if (record.get("protocol_hash") != lock["protocol_hash"]
            or record.get("status") != "completed"
            or record.get("k") != k or record.get("seed") != seed
            or record.get("selected_sha256") != hash_file(SELECTED)
            or not {"micro_f1", "macro_f1"} <= record.get("test", {}).keys()):
            raise ValueError(f"existing result is incompatible: {output}")
        return record
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU training requires an active Slurm allocation with CUDA")
    chosen = selection(lock)
    graph = data.load("freebase")
    paths = chosen["selected"][str(k)]
    config = dict(lock["config"])
    if pilot:
        config.update(max_epochs=3, patience=3)
    train_ids, val_ids, test_ids = data.split(graph, "book", seed)
    device = torch.device("cuda")
    features = to_device(data.features(graph, "book", config["feat"]), device)
    extractor = train.extractor_for(graph, paths, seed)
    model = train.build_model(graph, paths, int(graph["book"].y.max()) + 1, features,
                              config, "auto_phgt").to(device)
    started = time.perf_counter()
    fitted = train.fit(model, graph, features, extractor, train_ids, val_ids, config, seed)
    result = {"protocol_hash": lock["protocol_hash"], "status": "pilot_completed" if pilot else "completed",
              "dataset": "freebase", "condition": f"edgeoverlap_k{k}", "seed": seed,
              "k": k, "lmax": 4, "templates": paths, "config": config,
              "selected_sha256": hash_file(SELECTED),
              "fit": {key: val for key, val in fitted.items() if key != "history"},
              "validation": fitted["val"], "environment": device_info(device),
              "train_runtime_s": time.perf_counter() - started}
    if pilot:
        write_json(OUT / "pilot.json", result)
        return result
    events = OUT / "test_evaluations.jsonl"
    with events.open("a") as handle:
        handle.write(json.dumps({"k": k, "seed": seed, "protocol_hash": lock["protocol_hash"]}) + "\n")
    result["test"] = train.evaluate_test(model, graph, features, extractor, test_ids)
    write_json(output, result)
    return result


def aggregate() -> dict:
    lock = check_lock(verify_sources=True)
    rows, comparisons = [], []
    for k in KS:
        ours = {}
        for seed in SEEDS:
            path = result_path(k, seed)
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record.get("protocol_hash") != lock["protocol_hash"] or record.get("status") != "completed":
                raise ValueError(f"invalid result: {path}")
            ours[seed] = record
        for method in ("edgeoverlap", *METHODS):
            values = (ours if method == "edgeoverlap" else {
                seed: json.loads(existing_path(method, k, seed).read_text()) for seed in SEEDS})
            for metric in ("micro_f1", "macro_f1"):
                numbers = [100 * item["test"][metric] for item in values.values()]
                import statistics
                rows.append({"method": method, "k": k, "metric": metric, "n": len(numbers),
                             "mean": statistics.mean(numbers) if numbers else None,
                             "sample_sd": statistics.stdev(numbers) if len(numbers) > 1 else None,
                             "complete": len(numbers) == 5})
                if method != "edgeoverlap":
                    result = paired({s: r["test"][metric] for s, r in ours.items()},
                                    {s: r["test"][metric] for s, r in values.items()})
                    comparisons.append({"k": k, "metric": metric, "baseline": method,
                                        "n": result["n"], "mean_diff_pts": result["mean"],
                                        "ci95": result["ci95"], "p_perm": result["p_perm"],
                                        "wins": result["wins"], "losses": result["losses"]})
    report = {"protocol_hash": lock["protocol_hash"], "complete": all(row["complete"] for row in rows),
              "results": rows, "comparisons": comparisons}
    write_json(OUT / "summary.json", report)
    for name, table in (("results.csv", rows), ("paired.csv", comparisons)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=table[0])
            writer.writeheader()
            writer.writerows(table)
    return report


def run_all(gpus: list[str]) -> None:
    lock = check_lock(verify_sources=True)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("need at least one distinct allocated GPU")
    selection(lock)
    # A full-size validation-only GPU pilot, with no test access.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([str(PYTHON), "-m", "experiments.freebase_comparison",
                    "--pilot"], cwd=ROOT, env=env, check=True)
    for k in KS:
        for seed in SEEDS:
            if result_path(k, seed).exists():
                record = json.loads(result_path(k, seed).read_text())
                if (record.get("protocol_hash") != lock["protocol_hash"]
                    or record.get("status") != "completed"
                    or record.get("k") != k or record.get("seed") != seed
                    or record.get("selected_sha256") != hash_file(SELECTED)
                    or not {"micro_f1", "macro_f1"} <= record.get("test", {}).keys()):
                    raise ValueError(f"incompatible existing final: {result_path(k, seed)}")
    jobs = [(k, seed) for k in KS for seed in SEEDS if not result_path(k, seed).exists()]
    active: dict[str, tuple[subprocess.Popen, object, object, int, int]] = {}
    try:
        while jobs or active:
            for gpu in gpus:
                if gpu in active or not jobs:
                    continue
                k, seed = jobs.pop(0)
                log = OUT / "logs" / f"k{k}_seed{seed}"
                log.parent.mkdir(parents=True, exist_ok=True)
                stdout, stderr = log.with_suffix(".out").open("a"), log.with_suffix(".err").open("a")
                proc = subprocess.Popen([str(PYTHON), "-m", "experiments.freebase_comparison",
                                         "--task", str(k), str(seed)], cwd=ROOT,
                                        env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                        stdout=stdout, stderr=stderr, start_new_session=True)
                active[gpu] = (proc, stdout, stderr, k, seed)
                print(f"start k={k} seed={seed} gpu={gpu} pid={proc.pid}", flush=True)
            time.sleep(2)
            for gpu, (proc, stdout, stderr, k, seed) in list(active.items()):
                status = proc.poll()
                if status is None:
                    continue
                stdout.close()
                stderr.close()
                del active[gpu]
                print(f"finished k={k} seed={seed} gpu={gpu} rc={status}", flush=True)
                if status != 0 or not result_path(k, seed).exists():
                    raise RuntimeError(f"Freebase EdgeOverlap k={k}, seed={seed} failed; check {OUT / 'logs'}")
    except BaseException:
        for proc, stdout, stderr, _, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
            stdout.close()
            stderr.close()
        raise
    report = aggregate()
    if not report["complete"]:
        raise RuntimeError("Freebase EdgeOverlap results are incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze", action="store_true")
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--pilot", action="store_true")
    choice.add_argument("--task", nargs=2, type=int, metavar=("K", "SEED"))
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    if args.freeze:
        print("frozen:", freeze()["protocol_hash"])
    elif args.prepare:
        selection(check_lock(verify_sources=True))
    elif args.pilot:
        one(5, 0, pilot=True)
    elif args.task:
        k, seed = args.task
        if k not in KS or seed not in SEEDS:
            parser.error("task must use k=2/5 and seed=0..4")
        one(k, seed)
    elif args.aggregate:
        print("complete:", aggregate()["complete"])
    elif args.run:
        visible = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                   if gpu.strip()]
        run_all(visible)


if __name__ == "__main__":
    main()
