"""CLI options, result records, and file locations shared by the experiment scripts."""

from __future__ import annotations

import argparse
import hashlib
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from auto_phgt.discovery import (SCORING_METHOD, StatisticalDiscoveryModule,
                                 default_template_path, random_valid_paths, save_templates)
from auto_phgt.model import AutoPHGT
from auto_phgt.runtime import (DEVICE_CHOICES, cuda_memory, device_info, git_state,
                               peak_cpu_rss_mib, write_json)
from auto_phgt.selection import (diverse_top_k, homophily_table, homophily_top_k,
                                 hybrid_top_k, path_families)
from auto_phgt.tokenization import (MetaPathInstanceExtractor, discover_or_load_templates,
                                    instance_statistics, load_metapath_templates)
from auto_phgt.training import set_seed

MODES = ("hgt", "tokens", "auto_phgt")
PATH_SELECTIONS = ("discovered", "random", "diverse", "homophily", "hybrid", "fixed", "rank")


def add_common_args(parser: argparse.ArgumentParser, *, single_mode: bool = True):
    if single_mode:
        parser.add_argument("--mode", choices=MODES, default="auto_phgt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="default: frozen protocol (100)")
    parser.add_argument("--patience", type=int, default=None, help="default: frozen protocol (20)")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="auto selects CUDA only when it is available")
    parser.add_argument("--output", default=None,
                        help="result JSON (default: artifacts/results/<dataset>_<mode>_seed<seed>.json)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file (default: artifacts/checkpoints/<dataset>_<mode>_seed<seed>.pt)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="train without saving or resuming a checkpoint")
    parser.add_argument("--restart", action="store_true",
                        help="delete an existing checkpoint and train from scratch")
    return parser


def run_name(dataset: str, mode: str, seed: int) -> str:
    return f"{dataset}_{mode}_seed{seed}"


def default_output_path(dataset: str, mode: str, seed: int) -> Path:
    return Path("artifacts/results") / f"{run_name(dataset, mode, seed)}.json"


def resolve_checkpoint(dataset: str, mode: str, seed: int, *, checkpoint=None,
                       no_checkpoint: bool = False, restart: bool = False):
    if no_checkpoint:
        return None
    path = Path(checkpoint) if checkpoint else (
        Path("artifacts/checkpoints") / f"{run_name(dataset, mode, seed)}.pt")
    if restart and path.exists():
        path.unlink()
    return path


def discovery_record(graph, templates, *, dataset: str, target_node_type: str,
                     max_hops: int, k: int, from_file: bool = True) -> dict:
    """Selected templates with their canonical structural scores.

    ``from_file`` states whether the templates came from the persisted discovery file.
    """
    engine = StatisticalDiscoveryModule(graph, target_node_type, max_hops)
    schemas = [list(t.schema) for t in templates]
    scores = engine.calculate_path_frequencies(schemas)
    return {"scoring_method": SCORING_METHOD, "target_node_type": target_node_type,
            "max_hops": max_hops, "k": k,
            "template_file": str(default_template_path(dataset)) if from_file else None,
            "paths": [{"rank": rank, "path": schema, "score": scores[tuple(schema)]}
                      for rank, schema in enumerate(schemas, 1)]}


def select_templates(graph, *, dataset: str, selection: str, k: int, max_hops: int,
                     seed: int, target_node_type: str = "paper",
                     template_dir="artifacts/discovery", path_seed=None, paths=None,
                     ranks=None, train_ids=None, labels=None):
    """Select ``k`` templates from the canonical candidate space, plus a discovery record.

    ``discovered`` is the canonical structural top-k and ``random`` a uniform draw (seeded by
    ``path_seed``, default ``seed``). ``diverse``, ``homophily`` and ``hybrid`` are the
    selectors of :mod:`auto_phgt.selection`; ``fixed`` takes explicit ``paths`` and ``rank``
    takes 1-based canonical ``ranks``. When ``train_ids`` and ``labels`` are given, the
    training-label homophily of every candidate is computed and recorded (training labels
    only). The record lists each selected path with its canonical score and rank, and the
    full canonical ranking.
    """
    engine = StatisticalDiscoveryModule(graph, target_node_type, max_hops)
    candidates = engine.discover_candidate_paths()
    scores = engine.calculate_path_frequencies(candidates)
    assert len(scores) == len(candidates), "candidate paths must be distinct"
    # Same ordering as StatisticalDiscoveryModule.get_top_k_metapaths (stable on ties).
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    rank_of = {path: rank for rank, (path, _) in enumerate(ranking, 1)}
    ranked = [list(path) for path, _ in ranking]
    homophily = None
    if train_ids is not None and labels is not None:
        homophily = homophily_table(graph, ranked, train_ids, labels,
                                    target_node_type=target_node_type, seed=seed)
    template_dir = Path(template_dir)
    extra = {}
    if selection == "discovered":
        path = template_dir / f"{dataset}_discovered_k{k}_maxhops{max_hops}.json"
        templates = discover_or_load_templates(graph, dataset=dataset, k=k, max_hops=max_hops,
                                               target_node_type=target_node_type, path=path)
        schemas = [list(t.schema) for t in templates]
        if schemas != ranked[:k]:
            raise ValueError(f"{path} does not hold the current canonical top-{k}; regenerate it")
        extra["template_file"] = str(path)
    else:
        if selection == "random":
            random_seed = seed if path_seed is None else path_seed
            schemas, indices, _ = random_valid_paths(graph, target_node_type=target_node_type,
                                                     max_hops=max_hops, k=k, seed=random_seed)
            path = template_dir / f"{dataset}_random_k{k}_maxhops{max_hops}_seed{random_seed}.json"
            save_templates(path, schemas, dataset=dataset, target_node_type=target_node_type,
                           max_hops=max_hops, k=k)
            extra.update(random_seed=random_seed, candidate_indices=indices,
                         template_file=str(path))
        elif selection == "diverse":
            schemas = diverse_top_k(ranked, k)
        elif selection in ("homophily", "hybrid"):
            if homophily is None:
                raise ValueError(f"selection {selection!r} needs training ids and labels")
            schemas = (homophily_top_k(ranked, homophily, k) if selection == "homophily"
                       else hybrid_top_k(ranked, scores, homophily, k))
        elif selection == "fixed":
            schemas = [list(p) for p in (paths or [])]
        elif selection == "rank":
            schemas = [ranked[r - 1] for r in (ranks or [])]
        else:
            raise ValueError(f"path selection must be one of {PATH_SELECTIONS}")
        unknown = [p for p in schemas if tuple(p) not in rank_of]
        if unknown or len(schemas) != k or len({tuple(p) for p in schemas}) != k:
            raise ValueError(f"{selection!r} must give {k} distinct candidate paths, "
                             f"got {schemas}")
        templates = load_metapath_templates(schemas)
        for template in templates:
            template.validate(graph, target_node_type)
    discovery = {
        "selection": selection, "scoring_method": SCORING_METHOD,
        "target_node_type": target_node_type, "max_hops": max_hops, "k": k,
        "candidate_count": len(candidates), **extra,
        "paths": [{"rank": i, "path": schema, "score": scores[tuple(schema)],
                   "canonical_rank": rank_of[tuple(schema)],
                   "families": sorted(path_families(schema))}
                  for i, schema in enumerate(schemas, 1)],
        "canonical_ranking": [{"canonical_rank": rank, "path": list(path), "score": score,
                               "families": sorted(path_families(path))}
                              for rank, (path, score) in enumerate(ranking, 1)],
        "homophily": homophily,
    }
    return templates, discovery


def path_statistics(extractor, templates, target_ids, seed: int = 1) -> list:
    """Per-template completion statistics on a fixed draw (does not touch training RNG)."""
    instances = extractor.sample(target_ids, seed=seed)
    stats = instance_statistics(instances, templates)
    return [{"rank": i, **row, "num_targets": int(instances.node_ids.shape[0])}
            for i, row in enumerate(stats, 1)]


def split_record(train_ids, val_ids, test_ids, method: str) -> dict:
    digest = hashlib.sha256(torch.sort(val_ids).values.numpy().tobytes()).hexdigest()
    return {"method": method, "train": int(train_ids.numel()), "val": int(val_ids.numel()),
            "test": int(test_ids.numel()), "val_ids_sha256": digest[:16]}


def build_result(*, dataset, mode, seed, hyperparameters, discovery, device, model,
                 fit, test, checkpoint, started_at, setup=None, extra=None) -> dict:
    best = next((row for row in fit["history"] if row["epoch"] == fit["best_epoch"]), {})
    return {
        "dataset": dataset, "mode": mode, "seed": seed,
        "hyperparameters": hyperparameters,
        "discovery": discovery,
        "environment": {**device_info(device), "git": git_state()},
        "parameters": fit["parameters"],
        "parameter_breakdown": model.num_parameters(),
        "best_epoch": fit["best_epoch"],
        "epochs_completed": fit["epochs_completed"],
        "early_stopped": fit["early_stopped"],
        "resumed_from_epoch": fit["resumed_from_epoch"],
        "validation": {"accuracy": fit["best_val_accuracy"],
                       "macro_f1": best.get("val_macro_f1")},
        "test": test,
        "training_runtime_s": fit["runtime_s"],
        "setup_runtime_s": setup or {},
        "gpu_memory": cuda_memory(device),
        "peak_cpu_rss_mib": peak_cpu_rss_mib(),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "started_at": started_at,
        "finished_at": now(),
        "history": fit["history"],
        **(extra or {}),
        "status": "completed",
    }


class Stages(dict):
    """Wall-clock seconds per named setup stage: ``with stages("load"): ...``."""

    @contextmanager
    def __call__(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self[f"{name}_s"] = time.perf_counter() - start


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_result(result: dict, output=None) -> Path:
    path = Path(output) if output else default_output_path(
        result["dataset"], result["mode"], result["seed"])
    return write_json(path, result)


def summary(result: dict) -> dict:
    """Compact stdout view; the complete record is in the JSON file."""
    return {key: result[key] for key in ("dataset", "mode", "seed", "best_epoch",
                                         "validation", "test", "training_runtime_s",
                                         "parameters")} | {
        "device": result["environment"]["device"],
        "gpu": result["environment"].get("gpu_name")}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def smoke_modes(graph, templates, features, target_ids, device, *, instances_per_path=2,
                sampler=None, model_kwargs=None, seed=0) -> dict:
    """One forward/backward per mode; ``sampler`` switches to the sampled-subgraph path.

    Asserts logits shape, a finite loss, and finite gradients on every parameter
    that received one. Returns per-mode diagnostics including peak GPU memory.
    """
    device = torch.device(device)
    target = "paper"
    labels = graph[target].y
    num_classes = int(labels.max()) + 1
    kwargs = {"d_model": 16, "hgt_layers": 1, "hgt_heads": 2, "fusion_layers": 1,
              "fusion_heads": 2, "dropout": 0.0} | (model_kwargs or {})
    extractor = MetaPathInstanceExtractor(graph, templates,
                                          instances_per_path=instances_per_path, seed=seed)
    ids = torch.as_tensor(target_ids, dtype=torch.long)
    if sampler is not None:
        edges, n_id, count = sampler.sample(target, ids)
        targets = n_id[target][:count]
        batch = {"sampled_nodes": {t: int(v.numel()) for t, v in n_id.items()},
                 "sampled_edges": {"__".join(et): int(ei.size(1)) for et, ei in edges.items()}}
    else:
        edges, n_id, count, targets = graph.edge_index_dict, None, ids.numel(), ids
        batch = {}
    instances = extractor.sample(targets, seed=seed)
    batch["path_instances"] = list(instances.node_ids.shape)
    batch["complete_path_fraction"] = float(instances.complete_mask.float().mean())

    results = {}
    for mode in MODES:
        set_seed(seed)
        model = AutoPHGT.build(graph, templates, num_classes, x_dict=features, mode=mode,
                               **kwargs).to(device)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        logits = model(features, edges, instances=None if mode == "hgt" else instances,
                       target_ids=targets if mode == "hgt" else None, n_id_dict=n_id,
                       target_local=torch.arange(count) if n_id is not None else None)
        loss = F.cross_entropy(logits, labels[targets].to(device))
        loss.backward()
        _sync(device)
        elapsed = time.perf_counter() - start
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert logits.shape == (count, num_classes), logits.shape
        assert logits.device.type == device.type
        assert torch.isfinite(loss), loss
        assert grads and all(torch.isfinite(g).all() for g in grads)
        results[mode] = {"loss": loss.item(), "loss_finite": True,
                         "logits_shape": list(logits.shape),
                         "logits_device": str(logits.device), "backward_ok": True,
                         "params_with_grad": len(grads),
                         "parameters": model.num_parameters()["total"],
                         "forward_backward_s": elapsed, "gpu_memory": cuda_memory(device)}
    return {"batch": batch, "modes": results}
