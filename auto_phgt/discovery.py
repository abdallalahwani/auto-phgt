"""Schema meta-path discovery using the approved structural score.

For a path p with h relations, the score is

    (N_target * product(E_r / N_src(r) for r in p)) ** (1 / h).

E_r / N_src(r) is the mean relation out-degree (branching factor), not a
normalized transition probability. The product estimates structural volume;
it does not count concrete path instances or use labels or edge weights.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

SCORING_METHOD = "expected_structural_volume_geometric_mean_v1"


class StatisticalDiscoveryModule:
    """Canonical implementation from notebook 01, including its path rules."""

    def __init__(self, graph, target_node: str = "paper", max_hops: int = 4):
        self.graph = graph
        self.target_node = target_node
        self.max_hops = max_hops

    def _build_schema_graph(self):
        schema = defaultdict(list)
        for src, rel, dst in self.graph.edge_types:
            schema[src].append((rel, dst))
        return schema

    def discover_candidate_paths(self):
        """Return target-to-target schema paths of 2..max_hops relations.

        As in notebook 01, one-hop target self-relations are excluded and a
        path may continue after an earlier return to the target type.
        """
        schema = self._build_schema_graph()
        candidate_paths = []

        def dfs(current_node, path, hops):
            if hops > 0 and current_node == self.target_node and len(path) > 3:
                candidate_paths.append(path)
            if hops >= self.max_hops:
                return
            for rel, next_node in schema[current_node]:
                dfs(next_node, path + [rel, next_node], hops + 1)

        dfs(self.target_node, [self.target_node], 0)
        return candidate_paths

    def calculate_path_frequencies(self, candidate_paths):
        """Return the approved structural scores, rather than observed counts."""
        scores = {}
        for path in candidate_paths:
            expected_paths = self.graph[path[0]].num_nodes
            for i in range(0, len(path) - 2, 2):
                src, rel, dst = path[i : i + 3]
                num_src = self.graph[src].num_nodes
                num_edges = self.graph[(src, rel, dst)].num_edges
                expected_paths *= num_edges / num_src if num_src > 0 else 0
            hops = (len(path) - 1) // 2
            scores[tuple(path)] = expected_paths ** (1 / hops)
        return scores

    def get_top_k_metapaths(self, k: int = 5):
        paths = self.discover_candidate_paths()
        scores = self.calculate_path_frequencies(paths)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        unique_top_k = []
        seen = set()
        for path, _ in ranked:
            if path not in seen:
                seen.add(path)
                unique_top_k.append(list(path))
                if len(unique_top_k) == k:
                    break
        return unique_top_k


def random_valid_paths(graph, *, target_node_type: str = "paper", max_hops: int = 4,
                       k: int = 5, seed: int = 0):
    """Ablation baseline: ``k`` distinct candidates drawn uniformly, without scoring.

    The candidate space is exactly the canonical schema search's
    (``discover_candidate_paths`` with the same ``max_hops``). Returns the paths in draw
    order, their candidate indices, and the candidate count.
    """
    engine = StatisticalDiscoveryModule(graph, target_node_type, max_hops)
    candidates = engine.discover_candidate_paths()
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} candidate paths; cannot draw k={k}")
    indices = random.Random(seed).sample(range(len(candidates)), k)
    return [list(candidates[i]) for i in indices], indices, len(candidates)


def default_template_path(dataset: str, root: str | Path = "artifacts") -> Path:
    return Path(root) / f"{dataset}_top_k_metapaths.json"


def save_templates(path: str | Path, templates, *, dataset: str,
                   target_node_type: str, max_hops: int, k: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": dataset,
        "target_node_type": target_node_type,
        "scoring_method": SCORING_METHOD,
        "max_hops": max_hops,
        "k": k,
        "templates": [list(template) for template in templates],
    }
    # Concurrent workers may write the same file; each writes a private file and renames it.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_templates(path: str | Path, *, dataset: str, target_node_type: str,
                   max_hops: int, k: int):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": dataset,
        "target_node_type": target_node_type,
        "scoring_method": SCORING_METHOD,
        "max_hops": max_hops,
        "k": k,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"Template file {path} has {key}={record.get(key)!r}; expected {value!r}. "
                "Use a matching file or explicitly regenerate it."
            )
    templates = record.get("templates")
    if not isinstance(templates, list) or len(templates) > k:
        raise ValueError(f"Template file {path} has invalid templates")
    return templates


def discover_or_load_paths(graph, *, dataset: str, target_node_type: str = "paper",
                           max_hops: int = 4, k: int = 5,
                           path: str | Path | None = None, overwrite: bool = False):
    path = Path(path) if path is not None else default_template_path(dataset)
    if overwrite or not path.exists():
        engine = StatisticalDiscoveryModule(graph, target_node_type, max_hops)
        save_templates(path, engine.get_top_k_metapaths(k), dataset=dataset,
                       target_node_type=target_node_type, max_hops=max_hops, k=k)
    return load_templates(path, dataset=dataset, target_node_type=target_node_type,
                          max_hops=max_hops, k=k)
