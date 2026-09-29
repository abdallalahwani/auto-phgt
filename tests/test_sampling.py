import pytest
import torch

from auto_phgt.model import AutoPHGT
from auto_phgt.sampling import HeteroNeighborSampler
from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features


def test_sampled_edges_are_real_and_seeds_first(toy_graph):
    sampler = HeteroNeighborSampler(toy_graph, num_neighbors=[3, 3], seed=0)
    seeds = torch.tensor([3, 1])
    ei, n_id, bs = sampler.sample("paper", seeds)
    assert bs == 2 and torch.equal(n_id["paper"][:2], seeds)
    for t, ids in n_id.items():
        assert ids.unique().numel() == ids.numel()           # no duplicate local nodes
    for et, e in ei.items():
        real = set(map(tuple, toy_graph[et].edge_index.t().tolist()))
        glob = torch.stack([n_id[et[0]][e[0]], n_id[et[2]][e[1]]])
        assert set(map(tuple, glob.t().tolist())) <= real


def test_sampling_with_replacement_is_seeded_and_respects_budget(toy_graph):
    a = HeteroNeighborSampler(toy_graph, num_neighbors=[2], seed=7)
    b = HeteroNeighborSampler(toy_graph, num_neighbors=[2], seed=7)
    edge_a, ids_a, _ = a.sample("paper", torch.tensor([0]))
    edge_b, ids_b, _ = b.sample("paper", torch.tensor([0]))
    assert all(torch.equal(edge_a[et], edge_b[et]) for et in edge_a)
    assert all(torch.equal(ids_a[t], ids_b[t]) for t in ids_a)
    # Replacement and de-duplication can leave fewer than two unique authors.
    authors = ids_a["author"]
    assert 1 <= authors.numel() <= 2


def test_state_is_reset_between_batches(toy_graph):
    sampler = HeteroNeighborSampler(toy_graph, num_neighbors=[2, 2])
    sampler.sample("paper", torch.tensor([0, 1]))
    assert all((a == -1).all() for a in sampler._assoc.values())
    with pytest.raises(ValueError):
        sampler.sample("paper", torch.tensor([0, 0]))


def test_model_runs_on_sampled_subgraph(toy_graph, toy_templates):
    torch.manual_seed(0)
    x = impute_missing_features(toy_graph)
    model = AutoPHGT.build(toy_graph, toy_templates, 3, x_dict=x, d_model=16, hgt_heads=2, fusion_heads=2)
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=2)
    ei, n_id, bs = HeteroNeighborSampler(toy_graph, [4, 4]).sample("paper", torch.tensor([4, 0, 2]))
    targets = n_id["paper"][:bs]
    logits = model(x, ei, instances=ex.sample(targets), n_id_dict=n_id, target_local=torch.arange(bs))
    assert logits.shape == (3, 3)
    torch.nn.functional.cross_entropy(logits, toy_graph["paper"].y[targets]).backward()
