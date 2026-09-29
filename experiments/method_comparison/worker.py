"""CPU preparation, validation-only GPU pilots, and one atomic final result per run."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import torch

from auto_phgt.runtime import device_info, peak_cpu_rss_mib, to_device, write_json
from auto_phgt.training import set_seed
from experiments.residual_selection import data, train

from . import plan
from .multilabel import evaluate_multilabel, fit_multilabel
from .selectors import candidate_space, prepare_dataset, selected_paths


def cuda_device() -> torch.device:
    if not os.environ.get("SLURM_JOB_ID") and socket.gethostname().split(".")[0] != "c-004":
        raise RuntimeError("GPU work requires Slurm or the explicitly approved c-004 host")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a silent CPU training fallback")
    return torch.device("cuda")


def run(spec: dict, *, pilot: bool = False) -> dict:
    lock = plan.check_lock(spec["dataset"])
    if not pilot:
        existing = plan.final_record(spec, lock)
        if existing is not None:
            print(f"skip completed: {spec['id']}", flush=True)
            return existing
    started = time.perf_counter()
    device = cuda_device()
    ds = spec["dataset"]
    graph = plan.load_graph(ds)
    target = plan.DATASETS[ds]["target"]
    train_ids, val_ids, test_ids = data.split(graph, target, spec["seed"])
    cfg = dict(lock["configs"][ds])
    multilabel = plan.DATASETS[ds]["multilabel"]
    if pilot:
        cfg.update(max_epochs=3, patience=3)
        templates = candidate_space(graph, target, plan.lmax(ds, 5))[0][:5]
        if len(templates) != 5:
            raise ValueError("the full-size pilot requires five feasible paths")
        discovery = {"selector": "canonical", "pilot_only": True}
    else:
        templates, discovery = selected_paths(spec, lock, graph)
    if multilabel:
        set_seed(spec["seed"])
    features = to_device(data.features(graph, target, cfg["feat"]), device)
    classes = graph[target].y.shape[1] if multilabel else int(graph[target].y.max()) + 1
    extractor = None if spec["mode"] == "hgt" else train.extractor_for(
        graph, templates, spec["seed"])
    model = train.build_model(graph, templates, classes, features, cfg, spec["mode"]).to(device)
    print(f"training: {spec['id']} device={device} pilot={pilot}", flush=True)
    fit_fn = fit_multilabel if multilabel else train.fit
    fitted = fit_fn(model, graph, features, extractor, train_ids, val_ids, cfg, spec["seed"])
    environment = {**device_info(device), "peak_cpu_rss_mib": peak_cpu_rss_mib()}
    record = {
        "task": spec, "experiment_id": spec["id"], "suite": "autophgt_v5_final",
        "protocol_hash": lock["protocol_hash"], "dataset": ds, "seed": spec["seed"],
        "method": spec["method"], "k": spec["k"], "lmax": spec["lmax"], "config": cfg,
        "templates": None if spec["mode"] == "hgt" else templates, "discovery": discovery,
        "fit": {key: value for key, value in fitted.items() if key != "history"},
        "history": fitted["history"], "validation": fitted["val"],
        "parameters": fitted["parameters"], "parameter_breakdown": model.num_parameters(),
        "split": {"train": train_ids.numel(), "val": val_ids.numel(), "test": test_ids.numel()},
        "environment": environment, "time": datetime.now(timezone.utc).isoformat(),
    }
    if pilot:
        record.update(status="pilot_completed", pilot=True,
                      runtime_s=time.perf_counter() - started)
        write_json(plan.OUT / "pilots" / f"{ds}.json", record)
        print(f"validation-only pilot completed: {ds}", flush=True)
        return record
    event = {"task_id": spec["id"], "host": socket.gethostname(), "seed": spec["seed"],
             "dataset": ds, "time": datetime.now(timezone.utc).isoformat()}
    events = plan.OUT / "test_evaluations.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a") as handle:
        handle.write(json.dumps(event) + "\n")
    evaluate = evaluate_multilabel if multilabel else train.evaluate_test
    metrics = evaluate(model, graph, features, extractor, test_ids)
    record.update(status="completed", test=metrics, runtime_s=time.perf_counter() - started)
    write_json(plan.result_path(spec), record)
    print(f"completed: {spec['id']}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", choices=plan.DATASETS)
    group.add_argument("--pilot", choices=plan.DATASETS)
    group.add_argument("--task")
    group.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    os.chdir(plan.ROOT)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(1)
    if args.freeze:
        lock = plan.freeze()
        print(f"Frozen {lock['protocol_hash']}: {len(lock['tasks'])} cells, "
              f"{len(lock['reuse'])} reused, {len(lock['tasks']) - len(lock['reuse'])} new")
    elif args.prepare:
        print(json.dumps(prepare_dataset(args.prepare)))
    elif args.pilot:
        spec = next(item for item in plan.tasks(args.pilot)
                    if item["method"] == "canonical" and item["k"] == 5 and item["seed"] == 0)
        run(spec, pilot=True)
    else:
        matches = [item for item in plan.tasks() if item["id"] == args.task]
        if len(matches) != 1:
            parser.error(f"unknown task {args.task!r}")
        run(matches[0])


if __name__ == "__main__":
    main()
