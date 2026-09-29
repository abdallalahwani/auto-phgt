"""The 1-D-key de-duplication must reproduce the original sampler bit for bit."""

import torch
from torch_geometric.data import HeteroData

from auto_phgt.sampling import HeteroNeighborSampler, _unique_pairs


class ReferenceSampler(HeteroNeighborSampler):
    """The sampler exactly as it was before the ``_unique_pairs`` change."""

    @torch.no_grad()
    def sample(self, seed_type, seed_ids):
        seeds = torch.as_tensor(seed_ids, dtype=torch.long).view(-1)
        n_id = {t: [] for t in self.node_types}
        count = {t: 0 for t in self.node_types}
        assoc = self._assoc

        def add(t, ids):
            ids = ids.unique()
            new = ids[assoc[t][ids] == -1]
            assoc[t][new] = torch.arange(count[t], count[t] + new.numel())
            count[t] += new.numel()
            n_id[t].append(new)
            return new

        assoc[seed_type][seeds] = torch.arange(seeds.numel())
        count[seed_type] = seeds.numel()
        n_id[seed_type].append(seeds)
        frontier = {seed_type: seeds}
        edges = {et: [] for et in self.edge_types}
        try:
            for n in self.num_neighbors:
                next_frontier = {}
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
                    off = torch.minimum((torch.rand(d.numel(), generator=self.generator) * dg)
                                        .long(), dg - 1)
                    s = col[rowptr[d] + off]
                    pair = torch.unique(torch.stack([s, d]), dim=1)
                    edges[et].append((pair[0], pair[1]))
                    new = add(src_t, pair[0])
                    next_frontier.setdefault(src_t, []).append(new)
                frontier = {t: torch.cat(v).unique() for t, v in next_frontier.items()}
            n_id_dict = {t: (torch.cat(v) if v else torch.empty(0, dtype=torch.long))
                         for t, v in n_id.items()}
            edge_index_dict = {}
            for et, parts in edges.items():
                if not parts:
                    continue
                s = torch.cat([p[0] for p in parts]); d = torch.cat([p[1] for p in parts])
                ei = torch.unique(torch.stack([s, d]), dim=1)
                edge_index_dict[et] = torch.stack([assoc[et[0]][ei[0]], assoc[et[2]][ei[1]]])
        finally:
            for t, v in n_id.items():
                for ids in v:
                    assoc[t][ids] = -1
        return edge_index_dict, n_id_dict, seeds.numel()


def random_hetero_graph(seed=0):
    """MAG-shaped schema with random edges, including isolated and high-degree nodes."""
    gen = torch.Generator().manual_seed(seed)
    sizes = {"paper": 3000, "author": 5000, "institution": 60, "field_of_study": 400}
    g = HeteroData()
    for t, n in sizes.items():
        g[t].num_nodes = n

    def rel(src, dst, m):
        return torch.stack([torch.randint(0, sizes[src], (m,), generator=gen),
                            torch.randint(0, sizes[dst] // 2, (m,), generator=gen)])

    edges = {("author", "writes", "paper"): rel("author", "paper", 12000),
             ("paper", "cites", "paper"): rel("paper", "paper", 15000),
             ("paper", "has_topic", "field_of_study"): rel("paper", "field_of_study", 9000),
             ("author", "affiliated_with", "institution"): rel("author", "institution", 4000)}
    for (src, name, dst), ei in list(edges.items()):
        g[src, name, dst].edge_index = ei
        if src != dst:
            g[dst, f"rev_{name}", src].edge_index = ei.flip(0)
    return g


def assert_same(a, b):
    assert a[2] == b[2]
    assert a[0].keys() == b[0].keys() and a[1].keys() == b[1].keys()
    for key in a[0]:
        assert torch.equal(a[0][key], b[0][key]), key
    for key in a[1]:
        assert torch.equal(a[1][key], b[1][key]), key


def test_unique_pairs_matches_column_unique():
    gen = torch.Generator().manual_seed(0)
    s = torch.randint(0, 1000, (5000,), generator=gen)
    d = torch.randint(0, 37, (5000,), generator=gen)
    expected = torch.unique(torch.stack([s, d]), dim=1)
    got = torch.stack(_unique_pairs(s, d, 37))
    assert torch.equal(got, expected)
    empty = torch.empty(0, dtype=torch.long)
    assert _unique_pairs(empty, empty, 5)[0].numel() == 0


def test_sampler_is_bitwise_identical_to_reference(toy_graph):
    graphs = [(toy_graph, [4, 2], [torch.tensor([0, 3, 5]), torch.tensor([1, 2])]),
              (random_hetero_graph(), [10, 5],
               [torch.randperm(3000, generator=torch.Generator().manual_seed(i))[:256]
                for i in range(4)])]
    for graph, fanout, batches in graphs:
        for seed in (0, 1, 7):
            new, ref = HeteroNeighborSampler(graph, fanout, seed), ReferenceSampler(graph, fanout,
                                                                                  seed)
            for batch in batches:  # consecutive batches also check the RNG stream
                assert_same(new.sample("paper", batch), ref.sample("paper", batch))
            assert torch.equal(new.generator.get_state(), ref.generator.get_state())
