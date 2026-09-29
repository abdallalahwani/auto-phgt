"""V2 path selectors, compared against the canonical structural score (kept unchanged).

All selectors draw from the canonical candidate space
(``StatisticalDiscoveryModule.discover_candidate_paths``) and return ``k`` schemas:

* ``diverse``   label-free: walk the structural ranking and take a path whenever it adds a
                relation family not yet covered; fill any remaining slots in score order.
* ``homophily`` top-k by meta-path homophily measured on training labels only.
* ``hybrid``    top-k by structural score x homophily lift over chance.

A relation family merges a relation with its reverse (paper->term and term->paper are
one family); same-type relations keep their name (paper-cite-paper, paper-ref-paper).
"""

from __future__ import annotations

import torch

from .tokenization import PAD, MetaPathInstanceExtractor

MIN_HOMOPHILY_SUPPORT = 100


def relation_family(src: str, rel: str, dst: str) -> str:
    return f"{src}-{dst}:{rel}" if src == dst else "-".join(sorted((src, dst)))


def path_families(path) -> frozenset:
    return frozenset(relation_family(*path[i:i + 3]) for i in range(0, len(path) - 2, 2))


def diverse_top_k(ranking, k: int):
    """Greedy relation-family coverage in structural-score order, then fill by score."""
    selected, covered = [], set()
    for path in ranking:
        families = path_families(path)
        if families - covered:
            selected.append(list(path))
            covered |= families
            if len(selected) == k:
                return selected
    for path in ranking:
        if list(path) not in selected:
            selected.append(list(path))
            if len(selected) == k:
                break
    return selected


@torch.no_grad()
def homophily_table(graph, candidates, train_ids, labels, *, target_node_type: str = "paper",
                    samples_per_node: int = 16, seed: int = 0,
                    min_support: int = MIN_HOMOPHILY_SUPPORT):
    """Label agreement between a training node and the training-node endpoints of its paths.

    Endpoints outside the training split, incomplete walks and walks returning to the start
    node are ignored, so no validation or test label is read. ``lift`` is the agreement rate
    minus the chance rate (sum of squared training class shares), clipped at zero; paths with
    fewer than ``min_support`` usable walks get ``homophily = None`` and ``lift = 0``.
    """
    train_ids = torch.as_tensor(train_ids, dtype=torch.long).view(-1)
    labels = torch.as_tensor(labels).view(-1)
    num_nodes = graph[target_node_type].num_nodes
    is_train = torch.zeros(num_nodes, dtype=torch.bool)
    is_train[train_ids] = True
    train_labels = labels[train_ids]
    shares = torch.bincount(train_labels).float() / train_labels.numel()
    chance = float((shares ** 2).sum())
    extractor = MetaPathInstanceExtractor(graph, [list(c) for c in candidates],
                                          instances_per_path=samples_per_node,
                                          target_node_type=target_node_type, seed=seed)
    instances = extractor.sample(train_ids, seed=seed)
    start = train_ids.view(-1, 1)
    rows = []
    for k, template in enumerate(extractor.templates):
        end = instances.node_ids[:, k, :, template.length - 1]
        usable = end != PAD
        usable &= is_train[end.clamp(min=0)] & (end != start)
        support = int(usable.sum())
        if support >= min_support:
            agree = (labels[end.clamp(min=0)] == labels[start]) & usable
            rate = float(agree.sum()) / support
            lift = max(rate - chance, 0.0)
        else:
            rate, lift = None, 0.0
        rows.append({"path": list(template.schema), "homophily": rate, "lift": lift,
                     "support": support})
    return {"chance": chance, "samples_per_node": samples_per_node, "seed": seed,
            "min_support": min_support, "paths": rows}


def homophily_top_k(ranking, table, k: int):
    """Top-k by homophily; ties and paths without support fall back to structural order."""
    by_path = {tuple(row["path"]): row for row in table["paths"]}
    def rate(i):
        value = by_path[tuple(ranking[i])]["homophily"]
        return -1.0 if value is None else value

    order = sorted(range(len(ranking)), key=lambda i: (-rate(i), i))
    return [list(ranking[i]) for i in order[:k]]


def hybrid_top_k(ranking, scores, table, k: int):
    """Top-k by structural score x homophily lift; zero-lift paths fill in structural order."""
    by_path = {tuple(row["path"]): row for row in table["paths"]}
    value = {i: scores[tuple(ranking[i])] * by_path[tuple(ranking[i])]["lift"]
             for i in range(len(ranking))}
    order = sorted(range(len(ranking)), key=lambda i: (-value[i], i))
    return [list(ranking[i]) for i in order[:k]]
