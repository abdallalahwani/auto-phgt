"""
Pure-PyTorch heterogeneous neighbour sampler for mini-batch HGT on large graphs.

PyG's ``HGTLoader`` / ``NeighborLoader`` need the compiled ``pyg-lib`` or ``torch-sparse``
extensions, and wheels for new PyTorch releases (e.g. recent Colab runtimes) often lag
behind. This sampler has no such dependency and returns what ``AutoPHGT.forward`` expects
in mini-batch mode: ``edge_index_dict`` (local indices), ``n_id_dict`` (global IDs, seeds
first) and the number of seeds.

For every hop and relation ``src -> dst``, each frontier node of type ``dst`` draws
``num_neighbors[hop]`` incoming ``src`` neighbours uniformly with replacement,
then de-duplicates the resulting edges. Thus fewer than the budget may remain,
even when a node has enough distinct neighbours. Only sampled edges are kept.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .tokenization import _to_csr

EdgeType = Tuple[str, str, str]


def _unique_pairs(src: Tensor, dst: Tensor, num_dst: int) -> Tuple[Tensor, Tensor]:
    """Distinct ``(src, dst)`` pairs in lexicographic order.

    Returns exactly what ``torch.unique(torch.stack([src, dst]), dim=1)`` returns, but
    de-duplicates the 1-D keys ``src * num_dst + dst``; column-wise ``unique`` is about
    two orders of magnitude slower on CPU at MAG batch sizes.
    """
    key = torch.unique(src * num_dst + dst)
    return key // num_dst, key % num_dst


class HeteroNeighborSampler:
    def __init__(self, graph, num_neighbors: Sequence[int], seed: int = 0):
        self.num_neighbors = list(num_neighbors)
        self.node_types = list(graph.node_types)
        self.edge_types: List[EdgeType] = list(graph.edge_types)
        self.num_nodes = {t: graph[t].num_nodes for t in self.node_types}
        # incoming CSR: for every dst node, the src nodes that send messages to it
        self.in_csr = {et: _to_csr(graph[et].edge_index.flip(0), graph[et[2]].num_nodes)
                       for et in self.edge_types}
        self.generator = torch.Generator().manual_seed(seed)
        self._assoc = {t: torch.full((n,), -1, dtype=torch.long) for t, n in self.num_nodes.items()}

    @torch.no_grad()
    def sample(self, seed_type: str, seed_ids) -> Tuple[Dict[EdgeType, Tensor], Dict[str, Tensor], int]:
        seeds = torch.as_tensor(seed_ids, dtype=torch.long).view(-1)
        if seeds.unique().numel() != seeds.numel():
            raise ValueError("seed ids must be unique")
        n_id: Dict[str, List[Tensor]] = {t: [] for t in self.node_types}
        count = {t: 0 for t in self.node_types}
        assoc = self._assoc

        def add(t: str, ids: Tensor) -> Tensor:
            """Registers new global ids of type t, returns the ones not seen before."""
            ids = ids.unique()
            new = ids[assoc[t][ids] == -1]
            assoc[t][new] = torch.arange(count[t], count[t] + new.numel())
            count[t] += new.numel()
            n_id[t].append(new)
            return new

        assoc[seed_type][seeds] = torch.arange(seeds.numel())
        count[seed_type] = seeds.numel()
        n_id[seed_type].append(seeds)
        frontier: Dict[str, Tensor] = {seed_type: seeds}
        edges: Dict[EdgeType, List[Tuple[Tensor, Tensor]]] = {et: [] for et in self.edge_types}

        try:
            for n in self.num_neighbors:
                next_frontier: Dict[str, List[Tensor]] = {}
                for et in self.edge_types:
                    src_t, _, dst_t = et
                    dst = frontier.get(dst_t)
                    if dst is None or dst.numel() == 0:
                        continue
                    rowptr, col = self.in_csr[et]
                    deg = rowptr[dst + 1] - rowptr[dst]
                    has = deg > 0
                    if not has.any():
                        continue
                    d, dg = dst[has].repeat_interleave(n), deg[has].repeat_interleave(n)
                    off = torch.minimum((torch.rand(d.numel(), generator=self.generator) * dg).long(),
                                        dg - 1)
                    s = col[rowptr[d] + off]
                    ps, pd = _unique_pairs(s, d, self.num_nodes[dst_t])
                    edges[et].append((ps, pd))
                    new = add(src_t, ps)
                    next_frontier.setdefault(src_t, []).append(new)
                frontier = {t: torch.cat(v).unique() for t, v in next_frontier.items()}

            n_id_dict = {t: (torch.cat(v) if v else torch.empty(0, dtype=torch.long))
                         for t, v in n_id.items()}
            edge_index_dict = {}
            for et, parts in edges.items():
                if not parts:
                    continue
                s = torch.cat([p[0] for p in parts]); d = torch.cat([p[1] for p in parts])
                es, ed = _unique_pairs(s, d, self.num_nodes[et[2]])
                edge_index_dict[et] = torch.stack([assoc[et[0]][es], assoc[et[2]][ed]])
        finally:
            for t, v in n_id.items():  # reset the global -> local map for the next batch
                for ids in v:
                    assoc[t][ids] = -1
        return edge_index_dict, n_id_dict, seeds.numel()
