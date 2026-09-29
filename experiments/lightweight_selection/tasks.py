"""V4-Hedge task implementations. Each task reads its inputs from files written by its DAG
dependencies and writes its own outputs under artifacts/v4_hedge (see
experiments/lightweight_campaign.py). V3 modules are imported as code only; no V3 file is read."""

from __future__ import annotations

import csv
import json
import os
import resource
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from auto_phgt.discovery import StatisticalDiscoveryModule
from auto_phgt.runtime import device_info, git_state, peak_cpu_rss_mib, to_device, write_json
from experiments.residual_selection import data
from experiments.residual_selection.train import build_model, evaluate_test, extractor_for, fit

from . import plan
from . import selectors as sel
from . import transitions as tr
from .plan import V4

HEURISTIC = {"canonical": "discovered", "random": "random", "diverse": "diverse",
             "hybrid": "hybrid"}
TI_EXPLICIT = {"ti_raw": "TI_raw", "ti_cov": "TI_cov", "ti_norm": "TI_norm"}
ABBR = {"paper": "P", "author": "A", "subject": "S", "term": "T", "book": "B", "film": "F",
        "music": "M", "sports": "Sp", "people": "Pe", "location": "L", "organization": "O",
        "business": "Bu"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_threads() -> None:
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def log_test_event(task_id: str, **fields) -> None:
    path = V4 / "logs" / "test_evaluations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"time": now(), "task_id": task_id,
                                 "host": socket.gethostname(), **fields}) + "\n")


def environment(dev) -> dict:
    return {**device_info(dev), "git": git_state(), "protocol_hash": plan.protocol_hash(),
            "peak_cpu_rss_mib": peak_cpu_rss_mib()}


def read(path) -> dict:
    return json.loads(Path(path).read_text())


def abbr(path) -> str:
    """V2's path abbreviation (experiments/selector_controls_summary._abbr), extended to Freebase types."""
    out = [ABBR.get(path[0], path[0])]
    for i in range(1, len(path), 2):
        rel, node = path[i], path[i + 1]
        out.append(f"-{rel}-" if rel in ("cite", "ref", "cites") else "-")
        out.append(ABBR.get(node, node))
    return "".join(out)


def candidates(graph, t: str, lmax: int = plan.LMAX):
    """Canonical candidate space in structural-score order (V2/V3 ordering, stable ties)."""
    engine = StatisticalDiscoveryModule(graph, t, lmax)
    cands = engine.discover_candidate_paths()
    scores = engine.calculate_path_frequencies(cands)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [list(p) for p, _ in ranked], {tuple(p): s for p, s in ranked}


def full_cfg(cfg: dict) -> dict:
    return {**plan.BACKBONE_FIXED, **cfg}


def chosen_config(ds: str) -> dict:
    return read(V4 / "baseline" / f"chosen_{ds}.json")["chosen"]


def result_path(cond_id: str, seed: int) -> Path:
    return V4 / "results" / cond_id.split("__")[0] / f"{cond_id}__seed{seed}.json"


def bridge_path(cond_id: str, seed: int) -> Path:
    return V4 / "results" / "acm_v2backbone" / f"{cond_id}__seed{seed}.json"


def ti_path(ds: str) -> Path:
    return V4 / "selectors" / "transition_info" / f"{ds}.json"


def fastpath_path(ds: str, split: str, seed: int) -> Path:
    return V4 / "selectors" / "fastpath" / f"{ds}__{split}__seed{seed}.json"


def beam_path(ds: str) -> Path:
    return V4 / "selectors" / "ti_beam" / f"{ds}.json"


def primary_variant() -> str:
    return read(V4 / "decisions" / "ti_primary.json")["primary"]


def v4_splits(ds: str) -> list[int]:
    seeds = list(plan.SEEDS[ds])
    if ds == "freebase":
        seeds += plan.FREEBASE_EXTENSION["seeds"]
    return seeds


# --------------------------------------------------------------------------- provenance
EXPECTED = {"acm": {"target_nodes": 3025, "classes": 3, "official_train": 907,
                    "official_test": 2118, "edge_types": 8},
            "freebase": {"target_nodes": 40402, "classes": 7, "official_train": 2386,
                         "official_test": 5568, "edge_types": 36}}


def dataset_checks(ds: str, graph) -> dict:
    t = data.target(ds)
    node = graph[t]
    train, test = node.train_mask, node.test_mask
    labels = node.y
    edges = {}
    for et in graph.edge_types:
        ei = graph[et].edge_index
        src, _, dst = et
        edges["__".join(et)] = {
            "edges": int(ei.size(1)),
            "in_range": bool(ei.numel() == 0 or (int(ei[0].max()) < graph[src].num_nodes
                                                 and int(ei[1].max()) < graph[dst].num_nodes)),
            "self_loops": int((ei[0] == ei[1]).sum()) if src == dst else 0,
            "duplicates": int(ei.size(1) - torch.unique(ei, dim=1).size(1))}
    observed = {"target_nodes": node.num_nodes, "classes": int(labels[train | test].max()) + 1,
                "official_train": int(train.sum()), "official_test": int(test.sum()),
                "edge_types": len(graph.edge_types)}
    split_sizes = {}
    for seed in v4_splits(ds):
        tr_ids, va, te = data.split(graph, t, seed)
        split_sizes[seed] = {"train": tr_ids.numel(), "val": va.numel(), "test": te.numel()}
    return {"record": data.record(ds, graph), "observed": observed, "expected": EXPECTED[ds],
            "matches_expected": observed == EXPECTED[ds],
            "train_test_disjoint": not bool((train & test).any()),
            "edges": edges, "v4_split_sizes": split_sizes,
            "features_present": {nt: ("x" in graph[nt]) for nt in graph.node_types}}


def provenance() -> dict:
    dev = device()
    out = {"environment": environment(dev), "versions": versions(), "time": now(),
           "datasets": {ds: dataset_checks(ds, data.load(ds)) for ds in plan.DATASETS}}
    write_json(V4 / "logs" / "provenance_runtime.json", out)
    bad = [ds for ds, rec in out["datasets"].items()
           if not (rec["matches_expected"] and rec["train_test_disjoint"])]
    if bad:
        raise RuntimeError(f"dataset verification failed for {bad}")
    return {"verified": list(plan.DATASETS)}


def versions() -> dict:
    import platform

    import scipy
    import sklearn
    import torch_geometric
    try:
        import dgl  # noqa: F401
        dgl_version = dgl.__version__
    except ImportError:
        dgl_version = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "pyg": torch_geometric.__version__, "dgl": dgl_version or "not installed (unused)",
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


# --------------------------------------------------------------------------- backbone tuning
def tune(ds: str, layers: int, lr: float) -> dict:
    """Validation-only HGT grid slice (all feature regimes, wd, dropout, tuning seeds)."""
    dev = device()
    graph, t = data.load(ds), data.target(ds)
    classes = int(graph[t].y.max()) + 1
    template = candidates(graph, t)[0][:1]
    grid = plan.GRID[ds]
    runs = []
    for feat in grid["feat"]:
        x = to_device(data.features(graph, t, feat), dev)
        for wd in grid["wd"]:
            for dropout in grid["dropout"]:
                cfg = {"feat": feat, "layers": layers, "lr": lr, "wd": wd, "dropout": dropout}
                for seed in plan.TUNE_SEEDS:
                    train, val, _ = data.split(graph, t, seed)
                    model = build_model(graph, template, classes, x, full_cfg(cfg), "hgt").to(dev)
                    res = fit(model, graph, x, None, train, val, full_cfg(cfg), seed)
                    runs.append({"config": cfg, "seed": seed, "val": res["val"],
                                 "best_val_loss": res["best_val_loss"],
                                 "best_epoch": res["best_epoch"],
                                 "epochs_completed": res["epochs_completed"],
                                 "runtime_s": res["train_runtime_s"],
                                 "parameters": res["parameters"]})
                    del model
    write_json(V4 / "tuning" / f"{ds}_L{layers}_lr{lr:g}.json",
               {"dataset": ds, "layers": layers, "lr": lr, "runs": runs,
                "environment": environment(dev), "time": now()})
    return {"runs": len(runs)}


def select_config(ds: str) -> dict:
    """Highest mean validation micro-F1 over the tuning seeds; ties by lower validation loss."""
    runs = []
    for path in sorted((V4 / "tuning").glob(f"{ds}_*.json")):
        runs.extend(read(path)["runs"])
    groups = {}
    for run in runs:
        groups.setdefault(json.dumps(run["config"], sort_keys=True), []).append(run)
    table = []
    for key, members in groups.items():
        table.append({"config": json.loads(key), "seeds": len(members),
                      "val_micro_f1": float(np.mean([m["val"]["micro_f1"] for m in members])),
                      "val_macro_f1": float(np.mean([m["val"]["macro_f1"] for m in members])),
                      "val_loss": float(np.mean([m["best_val_loss"] for m in members])),
                      "best_epoch": float(np.mean([m["best_epoch"] for m in members])),
                      "runtime_s": float(np.mean([m["runtime_s"] for m in members])),
                      "parameters": members[0]["parameters"]})
    table.sort(key=lambda r: (-r["val_micro_f1"], r["val_loss"]))
    ref = plan.HGB_REFERENCE[ds]
    ref_row = next((r for r in table if r["config"] == ref), None)
    write_json(V4 / "baseline" / f"chosen_{ds}.json",
               {"dataset": ds, "chosen": table[0]["config"], "chosen_row": table[0],
                "hgb_reference_config": ref, "hgb_reference_row": ref_row,
                "grid_configs": len(table), "tuning_runs": len(runs), "ranking": table,
                "selection_rule": plan.SELECTION_RULE, "time": now()})
    return {"chosen": table[0]["config"], "val_micro_f1": table[0]["val_micro_f1"]}


# --------------------------------------------------------------------------- Transition-Info
def ti(ds: str) -> dict:
    """Transition-Info statistics of every <=4-hop candidate, rankings, TI set-aware greedy."""
    t_all, c_all = time.perf_counter(), cpu_seconds()
    graph, t = data.load(ds), data.target(ds)
    t0 = time.perf_counter()
    ranked, scores = candidates(graph, t)
    candidate_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ops = tr.relation_operators(graph)
    operator_s = time.perf_counter() - t0
    n = graph[t].num_nodes
    proj = tr.gaussian_projection(n, plan.TI_SET["fingerprint_dim"], plan.TI_SET["fingerprint_seed"])
    rows, fps = [], []
    t0 = time.perf_counter()
    for i, path in enumerate(ranked):
        stats, fp = tr.transition_info(ops, path, n, chunk=plan.TI["chunk_rows"],
                                       dense_fraction=plan.TI["dense_fraction"],
                                       projection=proj, eps=plan.TI["eps"])
        rows.append({"index": i, "canonical_rank": i + 1, "path": path, "abbr": abbr(path),
                     "hops": len(path) // 2, "structural_score": scores[tuple(path)], **stats})
        fps.append(fp)
    stats_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    cos = sel.cosine_matrix(np.stack(fps))
    cosine_s = time.perf_counter() - t0
    rankings, top_k, ti_set, set_s = {}, {}, {}, {}
    for variant in plan.TI["variants"]:
        values = [r[variant] for r in rows]
        rankings[variant] = sel.rank_by(values)
        top_k[variant] = {str(k): rankings[variant][:k] for k in plan.KS}
        s0 = time.perf_counter()
        chosen, steps = sel.ti_set_greedy(values, cos, max(plan.KS))
        set_s[variant] = time.perf_counter() - s0
        ti_set[variant] = {"selected": chosen, "steps": steps,
                           "top_k": {str(k): chosen[:k] for k in plan.KS}}
    out_dir = ti_path(ds).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{ds}_fingerprint_cosine.npy", cos)
    timing = {"candidate_generation_s": candidate_s, "operators_s": operator_s,
              "transition_statistics_s": stats_s, "fingerprint_cosine_s": cosine_s,
              "ti_set_greedy_s": set_s, "total_s": time.perf_counter() - t_all,
              "cpu_s": cpu_seconds() - c_all, "gpu_s": 0.0, "peak_rss_mib": peak_cpu_rss_mib()}
    write_json(ti_path(ds), {"dataset": ds, "target": t, "lmax": plan.LMAX,
                             "candidate_count": len(ranked), "candidates": rows,
                             "rankings": rankings, "top_k": top_k, "ti_set": ti_set,
                             "settings": {"ti": plan.TI, "ti_set": plan.TI_SET},
                             "timing": timing, "host": socket.gethostname(), "time": now()})
    return {"candidates": len(ranked), "stats_s": round(stats_s, 1)}


def ti_decision() -> dict:
    """Primary TI variant from the ACM diagnostic's VALIDATION-utility correlations."""
    path = V4 / "diagnostics" / "acm_score_correlations.json"
    order = plan.TI["tie_order"]
    record = {"rule": plan.TI["primary_rule"], "tie_order": order, "time": now()}
    if path.exists():
        corr = read(path)["val_utility"]
        rho = {v: corr[v]["rho"] for v in plan.TI["variants"]}
        value = {v: (-np.inf if rho[v] is None else rho[v]) for v in rho}
        best = max(value.values())
        primary = next(v for v in order if value[v] == best)
        record.update(primary=primary, val_rho=rho, source=str(path))
    else:
        record.update(primary=order[0], val_rho=None,
                      source="fallback: ACM diagnostic unavailable")
    write_json(V4 / "decisions" / "ti_primary.json", record)
    return {"primary": record["primary"]}


def ti_beam(ds: str) -> dict:
    """TI-Beam discovery up to BEAM['lmax'] hops; exact TI re-scoring of the best prefixes."""
    t_all, c_all = time.perf_counter(), cpu_seconds()
    graph, t = data.load(ds), data.target(ds)
    variant = primary_variant()
    ops = tr.relation_operators(graph)
    res = tr.beam_search(graph, ops, t, variant=variant, eps=plan.TI["eps"],
                         dense_fraction=plan.TI["dense_fraction"], **plan.BEAM)
    found = sorted(res["discovered"], key=lambda r: (-r["score"], "-".join(r["path"])))
    t0 = time.perf_counter()
    n = graph[t].num_nodes
    exact = []
    for row in found[:plan.BEAM["rescore_top"]]:
        stats, _ = tr.transition_info(ops, row["path"], n, chunk=plan.TI["chunk_rows"],
                                      dense_fraction=plan.TI["dense_fraction"],
                                      eps=plan.TI["eps"])
        exact.append({"path": row["path"], "abbr": abbr(row["path"]), "hops": len(row["path"]) // 2,
                      "sampled_score": row["score"], **stats})
    rescore_s = time.perf_counter() - t0
    exact.sort(key=lambda r: (-r[variant], "-".join(r["path"])))
    space = StatisticalDiscoveryModule(graph, t, plan.BEAM["lmax"]).discover_candidate_paths()
    selected = {str(k): [r["path"] for r in exact[:k]] for k in plan.KS}
    write_json(beam_path(ds), {
        "dataset": ds, "variant": variant, "settings": plan.BEAM,
        "schema_valid_candidates_le_lmax": len(space), "generated_prefixes": res["generated"],
        "discovered": len(found), "sampled_sources": res["sources"], "exact": exact,
        "selected": selected, "log": res["log"],
        "timing": {"beam_s": res["runtime_s"], "exact_rescoring_s": rescore_s,
                   "total_s": time.perf_counter() - t_all, "cpu_s": cpu_seconds() - c_all,
                   "gpu_s": 0.0, "peak_rss_mib": peak_cpu_rss_mib()},
        "time": now()})
    return {"discovered": len(found), "selected_k5": [abbr(p) for p in selected["5"]]}


# --------------------------------------------------------------------------- FastPath
def split_train(graph, ds: str, t: str, kind: str, seed: int):
    if kind == "v4":
        return data.split(graph, t, seed)[0]
    from auto_phgt.training import split_ids
    return split_ids(graph, target_node_type=t, seed=seed, acm_validation_size=180)[0]


def fastpath(ds: str) -> dict:
    """FastPath and FastPath-Set for every split used by V4 (training labels only)."""
    from auto_phgt.selection import homophily_table, hybrid_top_k
    t_all, c_all = time.perf_counter(), cpu_seconds()
    graph, t = data.load(ds), data.target(ds)
    ranked, scores = candidates(graph, t)
    labels = graph[t].y
    classes = int(labels.max()) + 1
    ops = tr.relation_operators(graph)
    official = np.sort(graph[t].train_mask.nonzero().view(-1).numpy())
    t0 = time.perf_counter()
    blocks, completion = [], []
    for path in ranked:
        b, c = tr.train_block(ops, path, official, chunk=plan.TI["chunk_rows"],
                              dense_fraction=plan.TI["dense_fraction"])
        blocks.append(b)
        completion.append(c)
    structure_s = time.perf_counter() - t0
    splits = [("v4", s) for s in v4_splits(ds)]
    if ds == "acm":
        splits += [("v2", s) for s in sorted(set(plan.V2_BRIDGE["seeds"])
                                             | set(plan.DIAGNOSTIC["v2_seeds"]))]
    cfg = plan.FASTPATH
    summary = {}
    for kind, seed in splits:
        train = np.sort(split_train(graph, ds, t, kind, seed).numpy())
        pos = np.searchsorted(official, train)
        assert np.array_equal(official[pos], train), "split must lie inside the official train"
        y = labels[torch.as_tensor(train)].numpy()
        folds = sel.stratified_folds(y, cfg["folds"], seed + cfg["fold_seed_offset"])
        t0 = time.perf_counter()
        rows, feats = [], []
        for i, path in enumerate(ranked):
            sub = blocks[i][np.ix_(pos, pos)]
            L, W, priors = sel.propagate(sub, y, folds, classes)
            rows.append({"index": i, "canonical_rank": i + 1, "abbr": abbr(path),
                         **sel.fastpath_scores(L, W, priors, y, folds, cfg["smoothing_eps"]),
                         "completion": float(completion[i][pos].mean())})
            feats.append(sel.fold_features(L, W))
        prop_s = time.perf_counter() - t0
        ranking = sel.rank_by([r["Q_FP"] for r in rows])
        t0 = time.perf_counter()
        chosen, steps, single, empty = sel.fastpath_set_greedy(
            np.stack(feats), y, folds, classes, max(plan.KS),
            C=cfg["set_classifier"]["C"], max_iter=cfg["set_classifier"]["max_iter"])
        set_s = time.perf_counter() - t0
        for i, r in enumerate(rows):
            r["probe_CE_single"] = single.get(i)
        t0 = time.perf_counter()
        homo = homophily_table(graph, ranked, torch.as_tensor(train), labels,
                               target_node_type=t, seed=seed)
        hybrid = {str(k): [ranked.index(list(p)) for p in
                           hybrid_top_k(ranked, scores, homo, k)] for k in plan.KS}
        hybrid_s = time.perf_counter() - t0
        by_path = {tuple(r["path"]): r for r in homo["paths"]}
        for i, path in enumerate(ranked):
            rows[i]["homophily"] = by_path[tuple(path)]["homophily"]
        record = {
            "dataset": ds, "split": kind, "seed": seed, "n_train": int(train.size),
            "train_ids_sha": _sha(train), "folds": cfg["folds"],
            "fold_seed": seed + cfg["fold_seed_offset"], "candidates": rows,
            "fastpath": {"ranking": ranking,
                         "top_k": {str(k): ranking[:k] for k in plan.KS}},
            "fastpath_set": {"selected": chosen, "steps": steps, "ce_empty": empty,
                             "top_k": {str(k): chosen[:k] for k in plan.KS}},
            "hybrid_reference": hybrid,
            "timing": {"label_propagation_s": prop_s, "set_selection_s": set_s,
                       "homophily_hybrid_s": hybrid_s},
            "settings": cfg, "time": now()}
        write_json(fastpath_path(ds, kind, seed), record)
        write_json(V4 / "selectors" / "fastpath_set" / f"{ds}__{kind}__seed{seed}.json",
                   {k: record[k] for k in ("dataset", "split", "seed", "fastpath_set",
                                           "settings", "time")})
        summary[f"{kind}{seed}"] = {"fp5": [abbr(ranked[i]) for i in ranking[:5]],
                                    "fps5": [abbr(ranked[i]) for i in chosen[:5]]}
    write_json(V4 / "selectors" / "fastpath" / f"{ds}__structure.json",
               {"dataset": ds, "candidate_count": len(ranked), "official_train": int(official.size),
                "timing": {"structure_s": structure_s, "total_s": time.perf_counter() - t_all,
                           "cpu_s": cpu_seconds() - c_all, "gpu_s": 0.0,
                           "peak_rss_mib": peak_cpu_rss_mib()},
                "splits": [f"{k}:{s}" for k, s in splits], "time": now()})
    return {"splits": len(splits), "structure_s": round(structure_s, 1)}


def _sha(ids) -> str:
    import hashlib
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- ACM diagnostic
def _read_v2_single_paths() -> dict:
    with plan.V2_SINGLE_PATHS.open() as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def diagnostic() -> dict:
    """ACM: every selector score for all 88 candidates against the frozen V2 utilities."""
    from .stats import rank_of, spearman, spearman_diff_ci
    ti_rec = read(ti_path("acm"))
    v2 = _read_v2_single_paths()
    fp = [read(fastpath_path("acm", "v2", s)) for s in plan.DIAGNOSTIC["v2_seeds"]]
    rows = []
    for c in ti_rec["candidates"]:
        name = c["abbr"]
        ref = v2[name]
        assert int(ref["canonical_rank"]) == c["canonical_rank"], name
        per = [f["candidates"][c["index"]] for f in fp]
        homo = [p["homophily"] for p in per if p["homophily"] is not None]
        rows.append({
            "path": name, "canonical_rank": c["canonical_rank"], "path_length": c["hops"],
            "relation_types": "|".join(c["path"][1::2]),
            "node_types": "-".join(c["path"][::2]),
            "V1_structural_score": c["structural_score"],
            "V2_structural_score": float(ref["structural_score"]),
            "homophily": float(ref["homophily"]) if ref["homophily"] else None,
            "homophily_recomputed": float(np.mean(homo)) if homo else None,
            "TI_raw": c["TI_raw"], "TI_cov": c["TI_cov"], "TI_norm": c["TI_norm"],
            "C": c["C"], "C_any": c["C_any"], "I": c["I"], "H_q": c["H_q"],
            "I_incl_self": c["I_incl_self"], "self_return_share": c["self_return_share"],
            "FastPath_Q": float(np.mean([p["Q_FP"] for p in per])),
            "FastPath_CE": float(np.mean([p["CE"] for p in per])),
            "FastPath_accuracy": float(np.mean([p["accuracy"] for p in per])),
            "FastPath_macro_f1": float(np.mean([p["macro_f1"] for p in per])),
            "FastPath_coverage": float(np.mean([p["coverage"] for p in per])),
            "FastPath_probe_CE": float(np.mean([p["probe_CE_single"] for p in per])),
            "frozen_V2_single_path_accuracy": float(ref["test_accuracy_mean"]),
            "frozen_V2_single_path_val_accuracy": float(ref["val_accuracy_mean"]),
            "V2_seeds": int(ref["seeds_completed"])})
    scores = {"structural": [r["V1_structural_score"] for r in rows],
              "homophily": [r["homophily"] for r in rows],
              "TI_raw": [r["TI_raw"] for r in rows], "TI_cov": [r["TI_cov"] for r in rows],
              "TI_norm": [r["TI_norm"] for r in rows],
              "FastPath": [r["FastPath_Q"] for r in rows],
              "FastPath_accuracy": [r["FastPath_accuracy"] for r in rows],
              "FastPath_probe": [-r["FastPath_probe_CE"] for r in rows],
              "I_incl_self": [r["I_incl_self"] for r in rows]}
    utility = {"test_utility": [r["frozen_V2_single_path_accuracy"] for r in rows],
               "val_utility": [r["frozen_V2_single_path_val_accuracy"] for r in rows]}
    corr = {u: {name: spearman(values, util) for name, values in scores.items()}
            for u, util in utility.items()}
    diffs = {}
    for u, util in utility.items():
        for a, b in [("TI_raw", "structural"), ("TI_cov", "structural"),
                     ("TI_norm", "structural"), ("FastPath", "homophily"),
                     ("FastPath", "structural"), ("TI_cov", "homophily")]:
            diffs[f"{u}:{a}-{b}"] = spearman_diff_ci(scores[a], scores[b], util,
                                                    plan.STATS["bootstrap"], plan.STATS["seed"])
    ranks = {name: rank_of(values) for name, values in scores.items()}
    ranks["utility"] = rank_of(utility["test_utility"])
    named = {p: {name: ranks[name][[r["path"] for r in rows].index(p)] for name in ranks}
             for p in plan.DIAGNOSTIC["named_paths"]}
    out = V4 / "diagnostics"
    _csv(out / "acm_path_scores.csv", rows)
    corr_rows = [{"score": name, "utility": u, **corr[u][name]} for u in corr for name in corr[u]]
    corr_rows += [{"score": key.split(":")[1], "utility": key.split(":")[0], **val}
                  for key, val in diffs.items()]
    _csv(out / "acm_score_correlations.csv", corr_rows)
    _csv(out / "acm_rank_table.csv",
         [{"path": r["path"], **{f"rank_{n}": ranks[n][i] for n in ranks}}
          for i, r in enumerate(rows)])
    tb = []
    for name, values in scores.items():
        order = sorted(range(len(rows)), key=lambda i: (-(values[i] if values[i] is not None
                                                            else -np.inf), i))
        for pos, i in enumerate(order[:10] + order[-10:]):
            tb.append({"score": name, "position": "top" if pos < 10 else "bottom",
                       "rank": ranks[name][i], "path": rows[i]["path"], "value": values[i],
                       "V2_utility": rows[i]["frozen_V2_single_path_accuracy"]})
    _csv(out / "acm_top_bottom.csv", tb)
    _csv(out / "acm_named_paths.csv", [{"path": p, **v} for p, v in named.items()])
    homo_check = [abs(r["homophily"] - r["homophily_recomputed"]) for r in rows
                  if r["homophily"] is not None and r["homophily_recomputed"] is not None]
    record = {"test_utility": corr["test_utility"], "val_utility": corr["val_utility"],
              "differences": diffs, "named_path_ranks": named,
              "v2_benchmarks": plan.DIAGNOSTIC["v2_benchmarks"],
              "homophily_recompute_max_abs_diff": max(homo_check) if homo_check else None,
              "structural_matches_v2": all(abs(r["V1_structural_score"] - r["V2_structural_score"])
                                           <= 1e-6 * r["V2_structural_score"] for r in rows),
              "n_paths": len(rows), "time": now()}
    write_json(out / "acm_score_correlations.json", record)
    return {k: round(v["rho"], 3) for k, v in corr["test_utility"].items() if v["rho"] is not None}


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- final runs
def template_record(graph, t, templates, lmax, extra=None):
    ranked, scores = candidates(graph, t, lmax)
    rank = {tuple(p): i + 1 for i, p in enumerate(ranked)}
    own = StatisticalDiscoveryModule(graph, t, lmax).calculate_path_frequencies(
        [list(p) for p in templates])
    return {"lmax": lmax, "candidate_count": len(ranked), **(extra or {}),
            "paths": [{"rank": i + 1, "path": list(p), "abbr": abbr(p),
                       "structural_score": own[tuple(p)], "canonical_rank": rank.get(tuple(p))}
                      for i, p in enumerate(templates)]}


def v4_selection(ds: str, selector: str, k: int, seed: int, split: str = "v4"):
    """Paths chosen by a V4 selector (or an explicit TI variant) from the selector files."""
    graph, t = data.load(ds), data.target(ds)
    ranked, _ = candidates(graph, t)
    if selector in ("ti", "ti_set") or selector in TI_EXPLICIT:
        rec = read(ti_path(ds))
        variant = TI_EXPLICIT.get(selector) or primary_variant()
        idx = (rec["top_k"][variant][str(k)] if selector != "ti_set"
               else rec["ti_set"][variant]["top_k"][str(k)])
        return [ranked[i] for i in idx], {"variant": variant, "indices": idx,
                                          "file": str(ti_path(ds))}
    if selector in ("fastpath", "fastpath_set"):
        rec = read(fastpath_path(ds, split, seed))
        idx = rec[selector]["top_k"][str(k)]
        return [ranked[i] for i in idx], {"indices": idx, "file": str(fastpath_path(ds, split, seed))}
    if selector == "ti_beam":
        rec = read(beam_path(ds))
        return [list(p) for p in rec["selected"][str(k)]], {"variant": rec["variant"],
                                                            "file": str(beam_path(ds))}
    raise ValueError(f"unknown V4 selector {selector!r}")


def resolve_templates(cond, graph, t, seed, train):
    ds, selector, k, lmax = cond["dataset"], cond["selector"], cond["k"], cond["lmax"]
    if cond["mode"] == "hgt":
        return candidates(graph, t)[0][:1], None
    if selector in HEURISTIC:
        from experiments.common import select_templates
        templates, record = select_templates(
            graph, dataset=ds, selection=HEURISTIC[selector], k=k, max_hops=lmax, seed=seed,
            target_node_type=t, template_dir=str(V4 / "discovery"), train_ids=train,
            labels=graph[t].y)
        return [list(x.schema) for x in templates], {"selector": selector, **record}
    paths, info = v4_selection(ds, selector, k, seed)
    return paths, {"selector": selector, **template_record(graph, t, paths, lmax, info)}


def conditions_by_id() -> dict:
    return {c["id"]: c for c in plan.conditions() + plan.bridge_conditions()}


def final(cond_id: str, seed: int) -> dict:
    cond = conditions_by_id()[cond_id]
    exp_id = f"{cond_id}__seed{seed}"
    dev = device()
    ds = cond["dataset"]
    cfg = full_cfg(chosen_config(ds))
    graph, t = data.load(ds), data.target(ds)
    classes = int(graph[t].y.max()) + 1
    train, val, test = data.split(graph, t, seed)
    t0 = time.perf_counter()
    templates, discovery = resolve_templates(cond, graph, t, seed, train)
    select_s = time.perf_counter() - t0
    x = to_device(data.features(graph, t, cfg["feat"]), dev)
    extractor = extractor_for(graph, templates, seed) if cond["mode"] != "hgt" else None
    model = build_model(graph, templates, classes, x, cfg, cond["mode"],
                        token_control=cond["token_control"]).to(dev)
    res = fit(model, graph, x, extractor, train, val, cfg, seed)
    log_test_event(exp_id, dataset=ds, condition=cond_id, seed=seed, n_test=int(test.numel()))
    test_scores = evaluate_test(model, graph, x, extractor, test)
    stats = None
    if extractor is not None:
        from experiments.common import path_statistics
        stats = path_statistics(extractor, templates, train)
    record = {"experiment_id": exp_id, "status": "completed", "suite": "autophgt_v4_hedge",
              "backbone": "strong", "dataset": ds, "condition": cond_id,
              **{k: cond[k] for k in ("name", "mode", "selector", "k", "lmax", "token_control",
                                      "group", "priority")},
              "seed": seed, "config": cfg,
              "templates": None if cond["mode"] == "hgt" else templates,
              "discovery": discovery, "path_statistics": stats,
              "split": {"train": train.numel(), "val": val.numel(), "test": test.numel()},
              "fit": {k: v for k, v in res.items() if k != "history"}, "history": res["history"],
              "validation": res["val"], "test": test_scores, "parameters": res["parameters"],
              "parameter_breakdown": model.num_parameters(), "selection_runtime_s": select_s,
              "environment": environment(dev), "time": now()}
    write_json(result_path(cond_id, seed), record)
    return {"test_micro_f1": test_scores["micro_f1"], "val_micro_f1": res["val"]["micro_f1"]}


def bridge(cond_id: str) -> dict:
    """V2-backbone bridge: a V4 selector under the unchanged V1/V2 ACM protocol, all seeds."""
    from experiments.run_acm import run
    cond = conditions_by_id()[cond_id]
    smoke = os.environ.get("V4_SMOKE") == "1"
    done = []
    for seed in plan.V2_BRIDGE["seeds"]:
        out = bridge_path(cond_id, seed)
        if out.exists() and read(out).get("status") == "completed":
            done.append(seed)
            continue
        paths, info = v4_selection("acm", cond["selector"], cond["k"], seed, split="v2")
        log_test_event(f"{cond_id}__seed{seed}", dataset="acm", condition=cond_id, seed=seed,
                       protocol="v2_backbone")
        run(mode="auto_phgt", seed=seed, k=cond["k"], path_selection="fixed", paths=paths,
            output=str(out), no_checkpoint=True, template_dir=str(V4 / "discovery"),
            epochs=2 if smoke else None, patience=1 if smoke else None,
            extra={"suite": "autophgt_v4_hedge", "backbone": "v2", "condition": cond_id,
                   "selector": cond["selector"], "v4_selection": info,
                   "protocol_hash": plan.protocol_hash()})
        done.append(seed)
    return {"seeds": done}


# --------------------------------------------------------------------------- decisions
def freebase_decision() -> dict:
    """Seed extension from VALIDATION variability only (no test result is read)."""
    rule = plan.FREEBASE_EXTENSION
    sds, means = {}, {}
    for cond in plan.conditions():
        if cond["dataset"] != "freebase" or cond["priority"] != 0:
            continue
        vals = [read(result_path(cond["id"], s))["validation"]["micro_f1"]
                for s in plan.SEEDS["freebase"] if result_path(cond["id"], s).exists()]
        if len(vals) >= 2:
            sds[cond["name"]] = float(np.std(vals, ddof=1))
            means[cond["name"]] = float(np.mean(vals))
    mean_sd = float(np.mean(list(sds.values()))) if sds else 0.0
    pool = sorted((n for n in rule["v4_pool"] if n in means), key=lambda n: (-means[n], n))
    chosen = pool[:rule["top_v4"]]
    record = {"extend": bool(sds) and mean_sd >= rule["sd_threshold"], "mean_val_sd": mean_sd,
              "per_condition_val_sd": sds, "val_means": means, "v4_ranked_by_val": pool,
              "extended_conditions": chosen + list(rule["comparators"]), **rule, "time": now()}
    write_json(V4 / "decisions" / "freebase_extension.json", record)
    return {"extend": record["extend"], "conditions": record["extended_conditions"]}


def gate_open(gate: str, args: dict) -> bool:
    if gate == "freebase_extension":
        path = V4 / "decisions" / "freebase_extension.json"
        if not path.exists():
            return False
        rec = read(path)
        name = args["cond_id"].split("__", 1)[1]
        return bool(rec["extend"]) and name in rec["extended_conditions"]
    if gate == "not_primary_variant":
        path = V4 / "decisions" / "ti_primary.json"
        if not path.exists():
            return False
        name = args["cond_id"].split("__", 1)[1].rsplit("_k", 1)[0]
        return TI_EXPLICIT[name] != read(path)["primary"]
    raise ValueError(gate)
