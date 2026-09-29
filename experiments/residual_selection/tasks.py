"""V3 task implementations. Each task reads its inputs from files written by its DAG
dependencies and writes its own outputs under artifacts/v3; see experiments/residual_campaign.py."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from auto_phgt.discovery import StatisticalDiscoveryModule
from auto_phgt.runtime import device_info, git_state, peak_cpu_rss_mib, to_device, write_json

from . import data, rcms
from .plan import (DATASETS, FREEBASE_EXTENSION, GRID, HGB_DATASETS, LMSPS_DBLP, LMSPS_LETTERS,
                   MAG, MAG_CONDITIONS, MANUAL_PATHS, RCMS, TUNE_SEEDS, V3, hgb_conditions,
                   protocol_hash)
from .train import backbone, build_model, evaluate_test, extractor_for, fit, predict

HEURISTIC = {"canonical": "discovered", "random": "random", "diverse": "diverse",
             "homophily": "homophily", "hybrid": "hybrid"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_threads() -> None:
    threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def log_test_event(task_id: str, **fields) -> None:
    path = V3 / "logs" / "test_evaluations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"time": now(), "task_id": task_id, "host": socket.gethostname(),
                                 **fields}) + "\n")


def environment(dev) -> dict:
    return {**device_info(dev), "git": git_state(), "protocol_hash": protocol_hash(),
            "peak_cpu_rss_mib": peak_cpu_rss_mib()}


def chosen_config(ds: str) -> dict:
    return json.loads((V3 / "baseline_audit" / f"chosen_{ds}.json").read_text())["chosen"]


def search_path(ds: str, seed: int, lmax: int) -> Path:
    return V3 / "searches" / f"{ds}_L{lmax}_seed{seed}.json"


def result_path(cond_id: str, seed: int) -> Path:
    return V3 / "results" / cond_id.split("__")[0] / f"{cond_id}__seed{seed}.json"


# --------------------------------------------------------------------------- tuning
def tune(ds: str, feat: str, layers: int) -> dict:
    """Validation-only HGT grid for one (feature regime, depth). Test is never evaluated."""
    dev = device()
    graph, t = data.load(ds), data.target(ds)
    classes = int(graph[t].y.max()) + 1
    template = rcms.candidate_space(graph, t, 4)[0][:1]
    x = to_device(data.features(graph, t, feat), dev)
    runs = []
    for lr in GRID["lr"]:
        for wd in GRID["wd"][ds]:
            for dropout in GRID["dropout"]:
                cfg = {"feat": feat, "layers": layers, "lr": lr, "wd": wd, "dropout": dropout}
                for seed in TUNE_SEEDS:
                    train, val, _ = data.split(graph, t, seed)
                    model = build_model(graph, template, classes, x, cfg, "hgt").to(dev)
                    res = fit(model, graph, x, None, train, val, cfg, seed)
                    runs.append({"config": cfg, "seed": seed, "val": res["val"],
                                 "best_val_loss": res["best_val_loss"],
                                 "best_epoch": res["best_epoch"],
                                 "epochs_completed": res["epochs_completed"],
                                 "runtime_s": res["train_runtime_s"],
                                 "parameters": res["parameters"]})
                    del model
    out = V3 / "tuning" / f"{ds}_{feat}_L{layers}.json"
    write_json(out, {"dataset": ds, "feat": feat, "layers": layers, "runs": runs,
                     "environment": environment(dev), "time": now()})
    return {"runs": len(runs)}


def select_config(ds: str) -> dict:
    """Pick the configuration with the highest mean validation micro-F1 over tuning seeds
    (ties: lower mean validation loss)."""
    runs = []
    for path in sorted((V3 / "tuning").glob(f"{ds}_*.json")):
        runs.extend(json.loads(path.read_text())["runs"])
    groups = {}
    for run in runs:
        key = json.dumps(run["config"], sort_keys=True)
        groups.setdefault(key, []).append(run)
    table = []
    for key, members in groups.items():
        table.append({"config": json.loads(key), "seeds": len(members),
                      "val_micro_f1": sum(m["val"]["micro_f1"] for m in members) / len(members),
                      "val_macro_f1": sum(m["val"]["macro_f1"] for m in members) / len(members),
                      "val_loss": sum(m["best_val_loss"] for m in members) / len(members),
                      "best_epoch": sum(m["best_epoch"] for m in members) / len(members),
                      "parameters": members[0]["parameters"]})
    table.sort(key=lambda r: (-r["val_micro_f1"], r["val_loss"]))
    ref = DATASETS[ds]["hgb_reference"]
    ref_key = {k: ref[k] for k in ("feat", "layers", "lr", "wd", "dropout")}
    ref_row = next((r for r in table if r["config"] == ref_key), None)
    record = {"dataset": ds, "chosen": table[0]["config"], "chosen_row": table[0],
              "hgb_reference_config": ref, "hgb_reference_row": ref_row,
              "grid_configs": len(table), "tuning_runs": len(runs), "ranking": table,
              "selection_rule": "max mean validation micro-F1 over tuning seeds; ties by val loss",
              "time": now()}
    write_json(V3 / "baseline_audit" / f"chosen_{ds}.json", record)
    return {"chosen": table[0]["config"], "val_micro_f1": table[0]["val_micro_f1"]}


def param_match(ds: str) -> dict:
    """HGT width whose parameter count is closest to the main Auto-PHGT (k=5) model."""
    cfg = chosen_config(ds)
    graph, t = data.load(ds), data.target(ds)
    classes = int(graph[t].y.max()) + 1
    lmax = 6 if ds == "dblp" else 4
    templates = rcms.candidate_space(graph, t, lmax)[0][:5]
    x = data.features(graph, t, cfg["feat"])
    target_params = sum(p.numel() for p in build_model(graph, templates, classes, x, cfg,
                                                       "auto_phgt").parameters())
    default = sum(p.numel() for p in build_model(graph, templates[:1], classes, x, cfg,
                                                 "hgt").parameters())
    best = None
    for width in range(8, 513, 8):
        n = sum(p.numel() for p in build_model(graph, templates[:1], classes, x, cfg, "hgt",
                                               width=width).parameters())
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (width, n)
        if n > 2 * target_params:
            break
    record = {"dataset": ds, "auto_phgt_reference_params": target_params,
              "auto_phgt_reference_templates": templates, "hgt_default_params": default,
              "hgt_width": best[0], "hgt_params": best[1],
              "relative_gap": (best[1] - target_params) / target_params, "time": now()}
    write_json(V3 / "baseline_audit" / f"param_match_{ds}.json", record)
    return record


# --------------------------------------------------------------------------- RCMS search
def search(ds: str, seed: int, lmax: int) -> dict:
    dev, timing = device(), {}
    t0 = time.perf_counter()
    cfg = chosen_config(ds)
    graph, t = data.load(ds), data.target(ds)
    labels = graph[t].y
    classes = int(labels.max()) + 1
    train, val, _ = data.split(graph, t, seed)
    offset = seed + RCMS["search_seed_offset"]

    t1 = time.perf_counter()
    ranked, scores = rcms.candidate_space(graph, t, lmax)
    probe = train[torch.randperm(train.numel(), generator=torch.Generator().manual_seed(offset))]
    kept, dropped = rcms.screen(graph, ranked, probe[:RCMS["coverage_probe_nodes"]], offset)
    schemas = [s for s, _ in kept]
    kept_rank = [ranked.index(s) for s in schemas]
    timing["candidates_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    x_cpu = data.features(graph, t, cfg["feat"])
    node_repr = data.label_free_repr(graph, x_cpu, RCMS["d_z"], seed=0)
    z = rcms.standardize(rcms.path_reprs(graph, schemas, node_repr, train, offset), dim=1)
    timing["path_cache_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    y = labels[train]
    folds = rcms.stratified_folds(y, RCMS["folds"], offset)
    x = to_device(x_cpu, dev)
    h = torch.zeros(train.numel(), classes)
    fold_hgt = []
    for j in range(RCMS["folds"]):
        fit_ids, oof_ids = train[folds != j], train[folds == j]
        model = build_model(graph, [ranked[0]], classes, x, cfg, "hgt").to(dev)
        res = fit(model, graph, x, None, fit_ids, val, cfg, seed * 10 + j)
        logits = predict(model, graph, x, None, oof_ids).cpu()
        h[folds == j] = F.log_softmax(logits, -1)
        fold_hgt.append({"fold": j, "fit_nodes": fit_ids.numel(), "oof_nodes": oof_ids.numel(),
                         "val": res["val"], "best_epoch": res["best_epoch"],
                         "oof_accuracy": float((logits.argmax(-1) == labels[oof_ids]).float().mean()),
                         "runtime_s": res["train_runtime_s"]})
        del model
    hs = rcms.standardize(h, dim=0)
    timing["hgt_crossfit_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    kmax = min(5, len(schemas))
    full_idx, steps_full, first_full = rcms.greedy(hs, z, y, folds, classes, kmax, dev)
    timing["search_full_s"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    nohgt_idx, steps_nohgt, first_nohgt = rcms.greedy(None, z, y, folds, classes, kmax, dev,
                                                      use_h=False)
    timing["search_nohgt_s"] = time.perf_counter() - t1
    sel = rcms.selections(steps_full, first_full, steps_nohgt, [2, 5], len(schemas))

    t1 = time.perf_counter()
    table, sets_full = rcms.comparison_sets(graph, t, ranked, scores, train, labels, seed, 5, lmax)
    position = {r: i for i, r in enumerate(kept_rank)}
    cka_sets = {name: [position[i] for i in idx if i in position]
                for name, idx in sets_full.items()}
    cka_sets.update({f"{variant}_k5": sel[variant]["5"] for variant in sel})
    cka = rcms.residual_cka(hs, z, cka_sets)
    timing["analysis_s"] = time.perf_counter() - t1
    homophily = {tuple(r["path"]): r for r in table["paths"]}
    per_candidate = [{"index": i, "path": s, "canonical_rank": kept_rank[i] + 1,
                      "structural_score": scores[tuple(s)], "coverage": c,
                      "homophily": homophily[tuple(s)]["homophily"],
                      "homophily_support": homophily[tuple(s)]["support"],
                      "utility_step1": first_full.get(i), "utility_step1_nohgt": first_nohgt.get(i)}
                     for i, (s, c) in enumerate(kept)]
    selected = {variant: {k: [schemas[i] for i in idx] for k, idx in ks.items()}
                for variant, ks in sel.items()}
    timing["total_s"] = time.perf_counter() - t0
    record = {"dataset": ds, "seed": seed, "lmax": lmax, "config": backbone(cfg),
              "candidates": {"raw": len(ranked), "kept": len(schemas), "dropped": dropped},
              "per_candidate": per_candidate, "steps_full": steps_full,
              "steps_nohgt": steps_nohgt, "selected": selected, "selected_index": sel,
              "fold_hgt": fold_hgt, "residual_cka": cka, "timing": timing,
              "search_nodes": train.numel(), "probe": {k: RCMS[k] for k in (
                  "d_z", "search_instances", "probe_l2", "probe_lr", "probe_steps", "folds")},
              "environment": environment(dev), "time": now()}
    write_json(search_path(ds, seed, lmax), record)
    return {"kept": len(schemas), "rcms_k5": selected["rcms"].get("5")}


# --------------------------------------------------------------------------- final runs
def template_record(graph, t, templates, lmax):
    ranked, scores = rcms.candidate_space(graph, t, lmax)
    engine = StatisticalDiscoveryModule(graph, t, lmax)
    rank = {tuple(p): i + 1 for i, p in enumerate(ranked)}
    own = engine.calculate_path_frequencies([list(p) for p in templates])
    return {"lmax": lmax, "candidate_count": len(ranked),
            "paths": [{"rank": i + 1, "path": list(p), "structural_score": own[tuple(p)],
                       "canonical_rank": rank.get(tuple(p))} for i, p in enumerate(templates)]}


def lmsps_templates() -> list:
    out = []
    for word in LMSPS_DBLP:
        nodes = [LMSPS_LETTERS[c] for c in word]
        schema = [nodes[0]]
        for node in nodes[1:]:
            schema += ["to", node]
        out.append(schema)
    return out


def resolve_templates(cond, graph, t, seed, train):
    """Templates for a condition and a record of how they were selected."""
    ds, sel, k, lmax = cond["dataset"], cond["selector"], cond["k"], cond["lmax"]
    if cond["mode"] == "hgt":
        return rcms.candidate_space(graph, t, 4)[0][:1], None
    if sel in HEURISTIC:
        from experiments.common import select_templates
        templates, record = select_templates(
            graph, dataset=ds, selection=HEURISTIC[sel], k=k, max_hops=lmax, seed=seed,
            target_node_type=t, template_dir=str(V3 / "discovery"), train_ids=train,
            labels=graph[t].y)
        return [list(x.schema) for x in templates], {"selector": sel, **record}
    if sel == "all":
        paths = rcms.candidate_space(graph, t, lmax)[0]
    elif sel.startswith("rcms"):
        search_record = json.loads(search_path(ds, seed, lmax).read_text())
        paths = search_record["selected"][sel][str(k)]
    elif sel == "manual":
        paths = MANUAL_PATHS[ds]
    elif sel == "lmsps":
        paths = lmsps_templates()
    else:
        raise ValueError(f"unknown selector {sel!r}")
    return [list(p) for p in paths], {"selector": sel, **template_record(graph, t, paths, lmax)}


def final(cond_id: str, seed: int) -> dict:
    cond = next(c for c in hgb_conditions() if c["id"] == cond_id)
    exp_id = f"{cond_id}__seed{seed}"
    dev = device()
    ds = cond["dataset"]
    cfg = chosen_config(ds)
    graph, t = data.load(ds), data.target(ds)
    classes = int(graph[t].y.max()) + 1
    train, val, test = data.split(graph, t, seed)
    t0 = time.perf_counter()
    templates, discovery = resolve_templates(cond, graph, t, seed, train)
    select_s = time.perf_counter() - t0
    width = None
    if cond["width"] == "pm":
        width = json.loads((V3 / "baseline_audit" / f"param_match_{ds}.json").read_text())["hgt_width"]
    x = to_device(data.features(graph, t, cfg["feat"]), dev)
    extractor = extractor_for(graph, templates, seed) if cond["mode"] != "hgt" else None
    model = build_model(graph, templates, classes, x, cfg, cond["mode"],
                        token_control=cond["token_control"], width=width).to(dev)
    res = fit(model, graph, x, extractor, train, val, cfg, seed)
    log_test_event(exp_id, dataset=ds, condition=cond_id, seed=seed, n_test=int(test.numel()))
    test_scores = evaluate_test(model, graph, x, extractor, test)
    paths = None if cond["mode"] == "hgt" else templates
    stats = None
    if extractor is not None:
        from experiments.common import path_statistics
        stats = path_statistics(extractor, templates, train)
    record = {"experiment_id": exp_id, "status": "completed", "suite": "autophgt_v3",
              "dataset": ds, "condition": cond_id, **{k: cond[k] for k in (
                  "name", "mode", "selector", "k", "lmax", "token_control", "group",
                  "priority")},
              "width": width, "seed": seed, "config": backbone(cfg), "templates": paths,
              "discovery": discovery, "path_statistics": stats,
              "split": {"train": train.numel(), "val": val.numel(), "test": test.numel()},
              "fit": {k: v for k, v in res.items() if k != "history"}, "history": res["history"],
              "validation": res["val"], "test": test_scores,
              "parameters": res["parameters"], "parameter_breakdown": model.num_parameters(),
              "selection_runtime_s": select_s, "environment": environment(dev), "time": now()}
    write_json(result_path(cond_id, seed), record)
    return {"test_micro_f1": test_scores["micro_f1"], "val_micro_f1": res["val"]["micro_f1"]}


def freebase_decision() -> dict:
    """Seed extension from VALIDATION variability only (no test result is read)."""
    sds = {}
    for cond in hgb_conditions():
        if cond["dataset"] != "freebase" or cond["priority"] != 0:
            continue
        vals = []
        for seed in range(5):
            path = result_path(cond["id"], seed)
            if path.exists():
                vals.append(json.loads(path.read_text())["validation"]["micro_f1"])
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            sds[cond["id"]] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    mean_sd = sum(sds.values()) / len(sds) if sds else 0.0
    record = {"extend": mean_sd >= FREEBASE_EXTENSION["sd_threshold"], "mean_val_sd": mean_sd,
              "per_condition_val_sd": sds, **FREEBASE_EXTENSION, "time": now()}
    write_json(V3 / "decisions" / "freebase_extension.json", record)
    return record


def freebase_extension_enabled() -> bool:
    path = V3 / "decisions" / "freebase_extension.json"
    return path.exists() and json.loads(path.read_text())["extend"]


# --------------------------------------------------------------------------- MAG
def mag_search(seed: int = 0) -> dict:
    """RCMS on ogbn-mag with a pre-declared internal holdout (no 3-fold HGT cross-fitting):
    an HGT (V1 configuration, at most MAG['search_epochs'] epochs) is trained on 80% of the
    official training papers; utilities are estimated on a stratified subset of the other
    20% with 3-fold probe cross-validation. One search serves all MAG seeds."""
    from auto_phgt.data import load_mag
    from auto_phgt.model import AutoPHGT
    from auto_phgt.sampling import HeteroNeighborSampler
    from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features
    from auto_phgt.training import fit_sampled_graph, split_ids
    from experiments.protocol import hyperparameters

    dev, timing, t0 = device(), {}, time.perf_counter()
    hp = hyperparameters("ogbn-mag")
    graph = load_mag("data/ogb_mag")
    labels = graph["paper"].y
    classes = int(labels.max()) + 1
    train, val, _ = split_ids(graph, seed=seed)
    offset = seed + RCMS["search_seed_offset"]
    fold = rcms.stratified_folds(labels[train], 5, offset)
    holdout, fit_ids = train[fold == 0], train[fold != 0]
    sub_fold = rcms.stratified_folds(labels[holdout], max(1, holdout.numel() // MAG["search_subset"]),
                                     offset + 1)
    subset = holdout[sub_fold == 0][:MAG["search_subset"]]

    t1 = time.perf_counter()
    ranked, scores = rcms.candidate_space(graph, "paper", 4)
    kept, dropped = rcms.screen(graph, ranked, subset[:RCMS["coverage_probe_nodes"]], offset)
    schemas = [s for s, _ in kept]
    features = impute_missing_features(graph)
    node_repr = data.label_free_repr(graph, features, RCMS["d_z"], seed=0)
    z = rcms.standardize(rcms.path_reprs(graph, schemas, node_repr, subset, offset), dim=1)
    timing["candidates_and_cache_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    model = AutoPHGT.build(graph, [ranked[0]], classes, x_dict=features, d_model=hp["d_model"],
                           pooling=hp["pooling"], token_dropout=hp["token_dropout"],
                           hgt_layers=hp["hgt_layers"], hgt_heads=hp["hgt_heads"],
                           fusion_layers=hp["fusion_layers"], fusion_heads=hp["fusion_heads"],
                           ffn_mult=hp["ffn_mult"], dropout=hp["dropout"], mode="hgt").to(dev)
    extractor = MetaPathInstanceExtractor(graph, [ranked[0]], instances_per_path=1, seed=seed)
    train_sampler = HeteroNeighborSampler(graph, hp["num_neighbors"], seed=seed)
    eval_sampler = HeteroNeighborSampler(graph, hp["num_neighbors"], seed=seed + 1)
    res = fit_sampled_graph(model, graph, features, extractor, train_sampler, eval_sampler,
                            fit_ids, val, epochs=MAG["search_epochs"], patience=hp["patience"],
                            batch_size=hp["batch_size"], lr=hp["lr"],
                            weight_decay=hp["weight_decay"], seed=seed, progress=True)
    timing["hgt_train_s"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    model.eval()
    eval_sampler.generator.manual_seed(1)
    parts = []
    with torch.no_grad():
        for start in range(0, subset.numel(), hp["batch_size"]):
            seeds = subset[start:start + hp["batch_size"]]
            edges, n_id, count = eval_sampler.sample("paper", seeds)
            parts.append(model(features, edges, target_ids=n_id["paper"][:count], n_id_dict=n_id,
                               target_local=torch.arange(count)).cpu())
    h = F.log_softmax(torch.cat(parts), -1)
    timing["hgt_inference_s"] = time.perf_counter() - t1
    hs = rcms.standardize(h, dim=0)
    y = labels[subset]
    folds = rcms.stratified_folds(y, RCMS["folds"], offset + 2)
    t1 = time.perf_counter()
    _, steps_full, first_full = rcms.greedy(hs, z, y, folds, classes, 5, dev)
    timing["search_full_s"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    _, steps_nohgt, first_nohgt = rcms.greedy(None, z, y, folds, classes, 5, dev, use_h=False)
    timing["search_nohgt_s"] = time.perf_counter() - t1
    sel = rcms.selections(steps_full, first_full, steps_nohgt, [2, 5], len(schemas))
    selected = {variant: {k: [schemas[i] for i in idx] for k, idx in ks.items()}
                for variant, ks in sel.items()}
    per_candidate = [{"index": i, "path": s, "canonical_rank": ranked.index(s) + 1,
                      "structural_score": scores[tuple(s)], "coverage": c,
                      "utility_step1": first_full.get(i), "utility_step1_nohgt": first_nohgt.get(i)}
                     for i, (s, c) in enumerate(kept)]
    timing["total_s"] = time.perf_counter() - t0
    record = {"dataset": "ogbn-mag", "seed": seed, "lmax": 4, "config": hp,
              "search_protocol": "80/20 stratified internal holdout of the official training set",
              "search_hgt": {"epochs": res["epochs_completed"], "best_epoch": res["best_epoch"],
                             "best_val_accuracy": res["best_val_accuracy"]},
              "search_nodes": subset.numel(), "fit_nodes": fit_ids.numel(),
              "candidates": {"raw": len(ranked), "kept": len(schemas), "dropped": dropped},
              "per_candidate": per_candidate, "steps_full": steps_full,
              "steps_nohgt": steps_nohgt, "selected": selected, "selected_index": sel,
              "timing": timing, "environment": environment(dev), "time": now()}
    write_json(search_path("ogbn-mag", seed, 4), record)
    return {"rcms_k5": selected["rcms"]["5"]}


def mag_paths(name: str, seed: int, epochs: int):
    tag = f"mag__{name}__seed{seed}" + ("" if epochs == MAG["epochs"] else f"__e{epochs}")
    return (V3 / "results" / "ogbn-mag" / f"{tag}.json", V3 / "checkpoints" / f"{tag}.pt", tag)


def mag_final(name: str, seed: int, epochs: int = MAG["epochs"]) -> dict:
    from experiments.run_mag import run as run_mag
    cond = next(c for c in MAG_CONDITIONS if c["name"] == name)
    output, checkpoint, tag = mag_paths(name, seed, epochs)
    kwargs = {}
    if cond["selection"] == "rcms":
        search_record = json.loads(search_path("ogbn-mag", 0, 4).read_text())
        kwargs = {"path_selection": "fixed", "paths": search_record["selected"]["rcms"]["5"]}
    else:
        kwargs = {"path_selection": cond["selection"]}
    result = run_mag(mode=cond["mode"], seed=seed, epochs=epochs, k=5, device="cuda",
                     output=output, checkpoint=checkpoint, token_control=cond["token_control"],
                     template_dir=str(V3 / "discovery"),
                     extra={"experiment_id": tag, "suite": "autophgt_v3", "condition": cond["id"],
                            "protocol_hash": protocol_hash()}, **kwargs)
    log_test_event(tag, dataset="ogbn-mag", condition=cond["id"], seed=seed,
                   n_test=result["split"]["test"], note="evaluated once at the end of run_mag.run")
    return {"test_accuracy": result["test"]["accuracy"], "epochs": result["epochs_completed"]}


def mag_continue(name: str, seed: int, seconds_left: float) -> dict:
    """Pre-declared continuation from 150 to 250 epochs, decided on validation only."""
    output, checkpoint, _ = mag_paths(name, seed, MAG["epochs"])
    record = json.loads(output.read_text())
    history = record["history"]
    decision = {"name": name, "seed": seed, "early_stopped": record["early_stopped"]}
    if record["early_stopped"] or len(history) < MAG["epochs"]:
        return {**decision, "continued": False, "reason": "early-stopped before the cap"}
    window = MAG["continue_window"]
    earlier = history[:-window] if len(history) > window else history[:1]
    gain = max(r["val_accuracy"] for r in history) - max(r["val_accuracy"] for r in earlier)
    need = (MAG["continue_to"] - MAG["epochs"]) * sum(r["epoch_runtime_s"] for r in history[-10:]) / 10
    decision.update(val_gain_last_window=gain, estimated_seconds=need, seconds_left=seconds_left)
    if gain < MAG["continue_min_gain"]:
        return {**decision, "continued": False, "reason": "validation no longer improving"}
    if need > seconds_left:
        return {**decision, "continued": False, "reason": "insufficient walltime"}
    new_output, new_checkpoint, _ = mag_paths(name, seed, MAG["continue_to"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["config"]["hyperparameters"]["epochs"] = MAG["continue_to"]
    state["stopped"] = False
    torch.save(state, new_checkpoint)
    result = mag_final(name, seed, epochs=MAG["continue_to"])
    return {**decision, "continued": True, **result}


# --------------------------------------------------------------------------- provenance
def provenance() -> dict:
    dev = device()
    out = V3 / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for ds in HGB_DATASETS:
        records[ds] = data.record(ds, data.load(ds))
    write_json(out / "datasets.json", records)
    v2 = Path("artifacts/v2/summary")
    if v2.exists():
        snap = out / "v2_acm_summary_snapshot"
        snap.mkdir(exist_ok=True)
        for name in ("acm_selectors_k5.csv", "acm_selectors_k2.csv", "acm_controls.csv",
                     "acm_random_sets.csv", "acm_single_paths.csv", "v2_summary.json"):
            if (v2 / name).exists():
                shutil.copy2(v2 / name, snap / name)
                os.chmod(snap / name, 0o444)
    write_json(out / "environment.json", environment(dev))
    return {"datasets": list(records)}
