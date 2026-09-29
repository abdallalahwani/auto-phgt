import json
import math

import pytest
import torch
from torch_geometric.data import HeteroData

from auto_phgt.discovery import (SCORING_METHOD, StatisticalDiscoveryModule,
                                 discover_or_load_paths, load_templates,
                                 save_templates)


@pytest.fixture
def discovery_graph():
    graph = HeteroData()
    graph["paper"].num_nodes = 4
    graph["author"].num_nodes = 2
    graph["topic"].num_nodes = 3
    graph["paper", "cites", "paper"].edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 2, 0, 3, 1, 2]])
    graph["paper", "writes", "author"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 1]])
    graph["author", "written", "paper"].edge_index = torch.tensor(
        [[0, 1], [0, 2]])
    graph["paper", "has", "topic"].edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3], [0, 1, 1, 2, 0, 2]])
    graph["topic", "of", "paper"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]])
    return graph


def test_candidate_enumeration_and_one_hop_rule(discovery_graph):
    engine = StatisticalDiscoveryModule(discovery_graph, max_hops=2)
    paths = engine.discover_candidate_paths()
    assert ["paper", "cites", "paper", "cites", "paper"] in paths
    assert ["paper", "writes", "author", "written", "paper"] in paths
    assert ["paper", "has", "topic", "of", "paper"] in paths
    assert ["paper", "cites", "paper"] not in paths
    assert len(paths) == 3


def test_max_hops_and_target_endpoints(discovery_graph):
    one = StatisticalDiscoveryModule(discovery_graph, max_hops=1)
    assert one.discover_candidate_paths() == []
    four = StatisticalDiscoveryModule(discovery_graph, max_hops=4)
    paths = four.discover_candidate_paths()
    assert any((len(path) - 1) // 2 == 4 for path in paths)
    assert all(path[0] == path[-1] == "paper" for path in paths)
    assert all(2 <= (len(path) - 1) // 2 <= 4 for path in paths)
    assert len(paths) == len({tuple(path) for path in paths})


def test_canonical_scoring_equation_and_ranking(discovery_graph):
    engine = StatisticalDiscoveryModule(discovery_graph, max_hops=2)
    scores = engine.calculate_path_frequencies(engine.discover_candidate_paths())
    cite = ("paper", "cites", "paper", "cites", "paper")
    topic = ("paper", "has", "topic", "of", "paper")
    author = ("paper", "writes", "author", "written", "paper")
    assert scores[cite] == pytest.approx((4 * (8 / 4) * (8 / 4)) ** (1 / 2))
    assert scores[topic] == pytest.approx(math.sqrt(4 * (6 / 4) * (3 / 3)))
    assert scores[author] == pytest.approx(math.sqrt(4 * (4 / 4) * (2 / 2)))
    assert engine.get_top_k_metapaths(3) == [list(cite), list(topic), list(author)]


def test_persistence_rejects_other_dataset_and_config(tmp_path, discovery_graph):
    path = tmp_path / "acm_top_k_metapaths.json"
    templates = StatisticalDiscoveryModule(discovery_graph, max_hops=2).get_top_k_metapaths(2)
    save_templates(path, templates, dataset="acm", target_node_type="paper", max_hops=2, k=2)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["scoring_method"] == SCORING_METHOD
    assert record["templates"] == templates
    assert load_templates(path, dataset="acm", target_node_type="paper", max_hops=2, k=2) == templates
    for overrides in ({"dataset": "ogbn-mag"}, {"max_hops": 4}, {"k": 3},
                      {"target_node_type": "author"}):
        config = dict(dataset="acm", target_node_type="paper", max_hops=2, k=2)
        config.update(overrides)
        with pytest.raises(ValueError, match="expected"):
            load_templates(path, **config)
    record["scoring_method"] = "average_edge_count_per_hop"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="scoring_method"):
        load_templates(path, dataset="acm", target_node_type="paper", max_hops=2, k=2)


def test_discover_or_load_uses_canonical_score(tmp_path, discovery_graph):
    path = tmp_path / "toy.json"
    paths = discover_or_load_paths(discovery_graph, dataset="toy", max_hops=2, k=2, path=path)
    assert paths == StatisticalDiscoveryModule(discovery_graph, max_hops=2).get_top_k_metapaths(2)
    assert discover_or_load_paths(discovery_graph, dataset="toy", max_hops=2, k=2, path=path) == paths
