"""Cached exact TI_cov, training-only FastPath/Hybrid, and sampled EdgeOverlap."""

from __future__ import annotations

import time

import numpy as np
import torch

from auto_phgt.discovery import StatisticalDiscoveryModule, random_valid_paths
from auto_phgt.runtime import write_json
from auto_phgt.selection import MIN_HOMOPHILY_SUPPORT, homophily_table, hybrid_top_k
from auto_phgt.tokenization import MetaPathInstanceExtractor
from experiments.residual_selection import data
from experiments.lightweight_selection import selectors as v4sel
from experiments.lightweight_selection import transitions as tr

from . import plan
from .edge_overlap import edge_overlap_matrix, greedy_edge_overlap
from .multilabel import multilabel_fastpath_scores


def candidate_space(graph, target: str, hops: int) -> tuple[list, dict]:
    engine = StatisticalDiscoveryModule(graph, target, hops)
    candidates = engine.discover_candidate_paths()
    scores = engine.calculate_path_frequencies(candidates)
    ranked = sorted(scores, key=lambda path: -scores[path])
    return [list(path) for path in ranked], scores


def read_cache(path, lock: dict) -> dict | None:
    if not path.exists():
        return None
    value = plan.read_json(path)
    if value.get("protocol_hash") != lock["protocol_hash"]:
        raise ValueError(f"{path}: selector cache belongs to another protocol")
    return value


def put_cache(path, lock: dict, value: dict) -> None:
    write_json(path, {**value, "protocol_hash": lock["protocol_hash"]})
    print(f"ready: {path.relative_to(plan.ROOT)}", flush=True)


def multilabel_homophily(graph, paths, train_ids, *, target: str, seed: int) -> dict:
    """V2 Hybrid's positive lift, replacing exact class equality with label-set Jaccard."""
    y = graph[target].y[train_ids].cpu().numpy()
    if y.ndim != 2 or not np.isin(y, (0, 1)).all() or len(y) == 0:
        raise ValueError("Hybrid requires a nonempty binary training-label matrix")
    patterns, counts = np.unique(y, axis=0, return_counts=True)
    patterns = patterns.astype(np.float64)
    intersection = patterns @ patterns.T
    union = patterns.sum(1)[:, None] + patterns.sum(1)[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    shares = counts / counts.sum()
    chance = float(shares @ similarity @ shares)
    known = torch.zeros(graph[target].num_nodes, dtype=torch.bool)
    known[train_ids] = True
    labels = torch.zeros((known.numel(), y.shape[1]), dtype=torch.bool)
    labels[train_ids] = torch.as_tensor(y, dtype=torch.bool)
    extractor = MetaPathInstanceExtractor(graph, paths, target_node_type=target,
                                          instances_per_path=16, seed=seed)
    instances = extractor.sample(train_ids, seed=seed)
    rows = []
    for index, template in enumerate(extractor.templates):
        endpoints = instances.node_ids[:, index, :, template.length - 1]
        usable = (endpoints >= 0) & known[endpoints.clamp(min=0)]
        usable &= endpoints != train_ids[:, None]
        support = int(usable.sum())
        if support >= MIN_HOMOPHILY_SUPPORT:
            src = labels[train_ids][:, None, :]
            dst = labels[endpoints.clamp(min=0)]
            inter = (src & dst).sum(-1).float()
            total = (src | dst).sum(-1).float()
            jaccard = torch.where(total > 0, inter / total.clamp(min=1), 0.0)
            rate = float(jaccard[usable].mean())
            lift = max(rate - chance, 0.0)
        else:
            rate, lift = None, 0.0
        rows.append({"path": list(template.schema), "homophily": rate, "lift": lift,
                     "support": support})
    return {"chance": chance, "paths": rows, "label_similarity": "positive-label-set Jaccard",
            "samples_per_node": 16, "min_support": MIN_HOMOPHILY_SUPPORT, "seed": seed}


def baseline_selections(graph, target: str, dataset: str, paths, scores, train, seed, ks, hops):
    if graph[target].y.ndim == 2:
        table = multilabel_homophily(graph, paths, train, target=target, seed=seed)
    else:
        table = homophily_table(graph, paths, train, graph[target].y,
                                target_node_type=target, seed=seed)
    selected = {"canonical": {}, "random": {}, "hybrid": {}}
    for k in ks:
        if k > len(paths):
            raise ValueError(f"{dataset}: k={k} exceeds {len(paths)} candidates at L={hops}")
        selected["canonical"][str(k)] = paths[:k]
        selected["random"][str(k)] = random_valid_paths(
            graph, target_node_type=target, max_hops=hops, k=k, seed=seed)[0]
        selected["hybrid"][str(k)] = hybrid_top_k(paths, scores, table, k)
    return {"selections": selected, "homophily": table}


def prepare_dataset(dataset: str) -> dict:
    lock = plan.check_lock(dataset, full_data_check=True)
    pending = [spec for spec in plan.tasks(dataset) if plan.final_record(spec, lock) is None]
    graph = plan.load_graph(dataset)
    target = plan.DATASETS[dataset]["target"]
    started = time.perf_counter()
    ops = None
    known_ti = {}
    known_fp = {seed: {} for seed in plan.SEEDS}
    for hops in sorted({spec["lmax"] for spec in pending if spec["k"] is not None}):
        specs = [spec for spec in pending if spec["lmax"] == hops and spec["k"] is not None]
        paths, scores = candidate_space(graph, target, hops)
        if any(spec["k"] > len(paths) for spec in specs):
            raise ValueError(f"{dataset} L={hops}: requested k is infeasible")
        baseline_specs = [spec for spec in specs if spec["method"] in ("canonical", "random", "hybrid")]
        for seed in sorted({spec["seed"] for spec in baseline_specs}):
            cache = plan.selector_path(dataset, hops, "baselines", seed)
            if read_cache(cache, lock) is not None:
                continue
            train, _, _ = data.split(graph, target, seed)
            ks = sorted({spec["k"] for spec in baseline_specs if spec["seed"] == seed})
            begin = time.perf_counter()
            record = baseline_selections(graph, target, dataset, paths, scores, train, seed, ks, hops)
            put_cache(cache, lock, {"dataset": dataset, "lmax": hops, "seed": seed,
                                   "runtime_s": time.perf_counter() - begin, "gpu_s": 0.0, **record})

        for seed in sorted({spec["seed"] for spec in specs if spec["method"] == "fastpath"}):
            cache = plan.selector_path(dataset, hops, "fastpath", seed)
            saved = read_cache(cache, lock)
            if saved is not None:
                known_fp[seed].update({tuple(row["path"]): row for row in saved["candidates"]})
                continue
            if ops is None:
                ops = tr.relation_operators(graph)
            train, _, _ = data.split(graph, target, seed)
            y = graph[target].y[train].cpu().numpy()
            fold_seed = seed + plan.FASTPATH["fold_seed_offset"]
            rows = []
            begin = time.perf_counter()
            for path in paths:
                if tuple(path) in known_fp[seed]:
                    rows.append(known_fp[seed][tuple(path)])
                    continue
                block, _ = tr.train_block(ops, path, train.numpy())
                if y.ndim == 2:
                    score = multilabel_fastpath_scores(
                        block, y, folds=plan.FASTPATH["folds"], seed=fold_seed,
                        smoothing_eps=plan.FASTPATH["smoothing_eps"])
                else:
                    folds = v4sel.stratified_folds(y, plan.FASTPATH["folds"], fold_seed)
                    probabilities, mass, priors = v4sel.propagate(block, y, folds, int(y.max()) + 1)
                    score = v4sel.fastpath_scores(probabilities, mass, priors, y, folds,
                                                   plan.FASTPATH["smoothing_eps"])
                if not np.isfinite(score["Q_FP"]):
                    raise ValueError(f"nonfinite FastPath score for {dataset}, seed {seed}, {path}")
                row = {"path": path, **score}
                rows.append(row)
                known_fp[seed][tuple(path)] = row
            order = v4sel.rank_by([row["Q_FP"] for row in rows])
            ks = {spec["k"] for spec in specs if spec["method"] == "fastpath" and spec["seed"] == seed}
            selected = {str(k): [paths[index] for index in order[:k]] for k in sorted(ks)}
            put_cache(cache, lock, {"dataset": dataset, "lmax": hops, "seed": seed,
                                   "selections": {"fastpath": selected}, "candidates": rows,
                                   "runtime_s": time.perf_counter() - begin, "gpu_s": 0.0})

        edge_specs = [spec for spec in specs if spec["method"] == "edgeoverlap"]
        if not edge_specs:
            continue
        cache = plan.selector_path(dataset, hops, "edgeoverlap")
        saved = read_cache(cache, lock)
        if saved is not None:
            known_ti.update({tuple(row["path"]): row for row in saved["candidates"]})
            continue
        begin = time.perf_counter()
        ti_source = None
        if dataset == "acm" and hops == 4:
            reference = lock["ti_acm_source"]
            source = plan.ROOT / reference["path"]
            if plan.digest(source) != reference["sha256"]:
                raise ValueError("reused ACM TI_cov scores changed")
            original = plan.read_json(source)["candidates"]
            if [row["path"] for row in original] != paths:
                raise ValueError("reused ACM TI_cov candidate order differs")
            known_ti.update({tuple(row["path"]): row for row in original})
            ti_source = reference["path"]
        rows = []
        for path in paths:
            if tuple(path) not in known_ti:
                if ops is None:
                    ops = tr.relation_operators(graph)
                stats, _ = tr.transition_info(ops, path, graph[target].num_nodes)
                known_ti[tuple(path)] = {"path": path, **stats}
            rows.append(known_ti[tuple(path)])
        settings = {key: plan.EDGE_OVERLAP[key] for key in (
            "max_sources", "instances_per_path", "seed")}
        overlap, metadata = edge_overlap_matrix(graph, paths, **settings)
        ks = sorted({spec["k"] for spec in edge_specs})
        selected, steps = greedy_edge_overlap([row["TI_cov"] for row in rows], overlap, max(ks))
        put_cache(cache, lock, {
            "dataset": dataset, "lmax": hops, "candidates": rows, "ti_reused_from": ti_source,
            "overlap": overlap.tolist(), "sampling": metadata, "steps": steps,
            "selections": {"edgeoverlap": {str(k): [paths[index] for index in selected[:k]] for k in ks}},
            "runtime_s": time.perf_counter() - begin, "gpu_s": 0.0,
        })
    return {"dataset": dataset, "runtime_s": time.perf_counter() - started, "status": "prepared"}


def selected_paths(spec: dict, lock: dict, graph) -> tuple[list, dict | None]:
    if spec["method"] == "hgt":
        paths, _ = candidate_space(graph, plan.DATASETS[spec["dataset"]]["target"], 4)
        return paths[:1], None
    path = plan.required_selection(spec)
    record = read_cache(path, lock)
    if record is None:
        raise FileNotFoundError(f"selector is not ready: {path}")
    selected = record["selections"][spec["method"]][str(spec["k"])]
    if len(selected) != spec["k"] or len({tuple(item) for item in selected}) != spec["k"]:
        raise ValueError(f"{spec['id']}: selector did not produce exactly k distinct paths")
    return selected, {"cache": str(path.relative_to(plan.ROOT)), "sha256": plan.digest(path),
                      "selector": spec["method"], "lmax": spec["lmax"], "label_free": spec["method"] in (
                          "canonical", "random", "edgeoverlap")}
