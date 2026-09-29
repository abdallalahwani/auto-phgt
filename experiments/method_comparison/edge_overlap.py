"""Label-free redundancy from complete CSR-sampled path instances.

Matrix rows and score vectors retain the supplied candidate order. Callers must
supply candidates in their canonical order; greedy ties use the earliest index.
"""

from __future__ import annotations

import operator
from collections.abc import Hashable, Iterable, Sequence, Set
from typing import TYPE_CHECKING

import numpy as np
import torch

from auto_phgt.tokenization import (
    MetaPathInstanceExtractor,
    MetaPathTemplate,
    PathInstances,
    load_metapath_templates,
)

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

__all__ = ["edge_overlap_matrix", "greedy_edge_overlap", "mean_source_jaccards"]

_SOURCE_BATCH_SIZE = 32


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def mean_source_jaccards(
    edge_sets_by_source: Iterable[Sequence[Set[Hashable]]],
) -> np.ndarray:
    """Average per-source Jaccards, never pooled edge sets.

    Each item contains one set per candidate for a single common source. An
    empty union contributes zero, including on the diagonal. The iterable is
    consumed once, so edge sets need only be retained for one source at a time.
    """
    total = None
    source_count = 0
    for edge_sets in edge_sets_by_source:
        if total is None:
            if not edge_sets:
                raise ValueError("at least one candidate is required")
            total = np.zeros((len(edge_sets), len(edge_sets)), dtype=np.float64)
        elif len(edge_sets) != total.shape[0]:
            raise ValueError("every source must contain the same number of candidate sets")
        sizes = [len(edges) for edges in edge_sets]
        for p, left in enumerate(edge_sets):
            if not sizes[p]:
                continue
            total[p, p] += 1.0
            for q in range(p):
                if not sizes[q]:
                    continue
                shared = len(left & edge_sets[q])
                if shared:
                    value = shared / (sizes[p] + sizes[q] - shared)
                    total[p, q] += value
                    total[q, p] += value
        source_count += 1
    if total is None:
        raise ValueError("at least one source is required")
    return total / source_count


def _packed_source_edges(
    instances: PathInstances,
    relation_offsets: Sequence[Sequence[int]],
    node_bits: int,
) -> Iterable[list[set[int]]]:
    nodes = instances.node_ids.numpy()
    complete = instances.complete_mask.numpy()
    for source in range(nodes.shape[0]):
        edge_sets = []
        for candidate, offsets in enumerate(relation_offsets):
            edges: set[int] = set()
            walks = nodes[source, candidate, complete[source, candidate], :len(offsets) + 1]
            # tolist() supplies Python ints before shifts: packed keys may exceed int64.
            for walk in walks.tolist():
                for hop, offset in enumerate(offsets):
                    edges.add(offset | (walk[hop] << node_bits) | walk[hop + 1])
            edge_sets.append(edges)
        yield edge_sets


def edge_overlap_matrix(
    graph: HeteroData,
    templates: Sequence[MetaPathTemplate | Sequence[str]],
    *,
    source_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    max_sources: int = 1024,
    instances_per_path: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Return O(p,q) and JSON-safe sampling/provenance metadata.

    E_p(v) contains distinct typed directed edges from *complete* sampled
    instances only. O is the uniform mean of Jaccard(E_p(v), E_q(v)) over the
    same fixed sources for every pair. Empty unions contribute zero, so a
    diagonal entry is the fraction of sources with a nonempty edge set.

    With no explicit IDs, sample at most ``max_sources`` uniformly without
    replacement and sort those IDs. Explicit IDs must be unique, nonempty,
    in-range integers; their order is preserved and ``max_sources`` does not
    truncate them. All templates must share a source node type. Topology must
    be on CPU, as required by the existing CSR extractor. Labels and feature
    values are never used.

    One extractor is seeded once and advanced across fixed 32-source batches.
    Apart from its CSR topology, memory is O(B*P*I*L + P*I*L + P**2), not
    O(N*P*I*L) Python edge objects. Typed edge keys use collision-free Python
    integers with disjoint relation, source-ID and destination-ID bit fields.
    """
    max_sources = _integer(max_sources, "max_sources", 1)
    instances_per_path = _integer(instances_per_path, "instances_per_path", 1)
    seed = _integer(seed, "seed")
    if seed >= 1 << 64:
        raise ValueError("seed must be less than 2**64")
    paths = load_metapath_templates(templates)
    target_type = paths[0].node_types[0]
    for path in paths:
        path.validate(graph, target_type)
    target_count = graph[target_type].num_nodes
    if target_count is None or target_count < 1:
        raise ValueError("the source node type must contain at least one node")
    target_count = int(target_count)

    if source_ids is None:
        ids = np.sort(
            np.random.default_rng(seed).choice(
                target_count, size=min(max_sources, target_count), replace=False,
            )
        )
        source_sampling = "uniform_without_replacement"
    else:
        if isinstance(source_ids, torch.Tensor):
            source_ids = source_ids.detach().cpu()
        ids = np.asarray(source_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("source_ids must be a nonempty one-dimensional integer sequence")
        if np.any(ids < 0) or np.any(ids >= target_count):
            raise ValueError(f"source_ids must lie in [0, {target_count})")
        if np.unique(ids).size != ids.size:
            raise ValueError("source_ids must not contain duplicates")
        source_sampling = "explicit"
    ids = ids.astype(np.int64, copy=False)

    edge_types = sorted({edge_type for path in paths for edge_type in path.edge_types})
    for edge_type in edge_types:
        if graph[edge_type].edge_index.device.type != "cpu":
            raise ValueError("the existing CSR sampler requires CPU edge_index tensors")
    node_types = {node_type for path in paths for node_type in path.node_types}
    node_counts = [graph[node_type].num_nodes for node_type in node_types]
    if any(count is None or count < 0 for count in node_counts):
        raise ValueError("all path node types must have known nonnegative node counts")
    node_bits = max(1, (max(int(count) for count in node_counts) - 1).bit_length())
    relation_ids = {edge_type: index for index, edge_type in enumerate(edge_types)}
    relation_offsets = [
        [relation_ids[edge_type] << (2 * node_bits) for edge_type in path.edge_types]
        for path in paths
    ]

    extractor = MetaPathInstanceExtractor(
        graph, paths, instances_per_path=instances_per_path, seed=seed,
    )
    complete_counts = np.zeros(len(paths), dtype=np.int64)
    nonempty_counts = np.zeros(len(paths), dtype=np.int64)
    edge_counts = np.zeros(len(paths), dtype=np.int64)

    def sampled_edge_sets() -> Iterable[list[set[int]]]:
        for start in range(0, len(ids), _SOURCE_BATCH_SIZE):
            instances = extractor.sample(ids[start:start + _SOURCE_BATCH_SIZE])
            complete_counts[:] += instances.complete_mask.numpy().sum(axis=(0, 2))
            for edge_sets in _packed_source_edges(instances, relation_offsets, node_bits):
                sizes = np.fromiter((len(edges) for edges in edge_sets), dtype=np.int64)
                nonempty_counts[:] += sizes > 0
                edge_counts[:] += sizes
                yield edge_sets

    overlaps = mean_source_jaccards(sampled_edge_sets())
    metadata = {
        "seed": seed,
        "target_node_type": target_type,
        "candidate_count": len(paths),
        "templates": [list(path.schema) for path in paths],
        "source_ids": ids.tolist(),
        "num_sources": len(ids),
        "max_sources": max_sources,
        "source_sampling": source_sampling,
        "instances_per_path": instances_per_path,
        "source_batch_size": _SOURCE_BATCH_SIZE,
        "sampler": "MetaPathInstanceExtractor",
        "complete_instances_only": True,
        "labels_used": False,
        "aggregation": "mean_of_per_source_jaccards",
        "empty_union_overlap": 0.0,
        "edge_encoding": {
            "kind": "python_int_bit_fields",
            "node_bits": node_bits,
            "edge_types_by_id": [list(edge_type) for edge_type in edge_types],
        },
        "path_statistics": [
            {
                "candidate_index": index,
                "complete_instances": int(complete_counts[index]),
                "sources_with_edges": int(nonempty_counts[index]),
                "mean_unique_edges": float(edge_counts[index] / len(ids)),
            }
            for index in range(len(paths))
        ],
    }
    return overlaps, metadata


def _real_array(values, name: str) -> np.ndarray:
    try:
        result = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real-valued array") from exc
    if result.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    return result.astype(np.float64, copy=False)


def greedy_edge_overlap(
    scores: Sequence[float] | np.ndarray,
    overlaps: Sequence[Sequence[float]] | np.ndarray,
    k: int,
) -> tuple[list[int], list[dict]]:
    """Select by TI_cov(p) * (1 - max_{q selected} O(p,q)).

    Scores are supplied exact TI_cov values, never recomputed or normalized.
    Input indices must be in canonical candidate order; exact ties choose the
    earliest remaining index. The first choice is the TI_cov argmax. A run for
    smaller k is a prefix of a larger run on the same inputs. k=0 is allowed;
    k above the candidate count is an error, not a shortened result.
    """
    k = _integer(k, "k")
    values = _real_array(scores, "scores")
    matrix = _real_array(overlaps, "overlaps")
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("scores must be finite and nonnegative")
    if matrix.ndim != 2 or matrix.shape != (len(values), len(values)):
        raise ValueError("overlaps must be square and match the number of scores")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("overlaps must be finite and in [0, 1]")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("overlaps must be symmetric")
    if k > len(values):
        raise ValueError("k must not exceed the candidate count")

    selected: list[int] = []
    records: list[dict] = []
    remaining = np.ones(len(values), dtype=bool)
    max_overlap = np.zeros(len(values), dtype=np.float64)
    for step in range(k):
        conditional_scores = values * (1.0 - max_overlap)
        conditional_scores[~remaining] = -np.inf
        winner = int(np.argmax(conditional_scores))
        records.append({
            "step": step + 1,
            "selected_index": winner,
            "TI_cov": float(values[winner]),
            "max_overlap": float(max_overlap[winner]),
            "novelty_factor": float(1.0 - max_overlap[winner]),
            "score": float(conditional_scores[winner]),
        })
        selected.append(winner)
        remaining[winner] = False
        np.maximum(max_overlap, matrix[:, winner], out=max_overlap)
    return selected, records
