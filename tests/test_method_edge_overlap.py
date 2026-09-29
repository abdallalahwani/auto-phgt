import json

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

import experiments.method_comparison.edge_overlap as edge_overlap
from auto_phgt.tokenization import MetaPathInstanceExtractor, MetaPathTemplate, PathInstances
from experiments.method_comparison.edge_overlap import (
    edge_overlap_matrix,
    greedy_edge_overlap,
    mean_source_jaccards,
)


def _graph(num_nodes, **relations):
    graph = HeteroData()
    graph["p"].num_nodes = num_nodes
    for relation, edges in relations.items():
        graph["p", relation, "p"].edge_index = torch.tensor(
            edges, dtype=torch.long,
        ).reshape(-1, 2).t().contiguous()
    return graph


def _path(*relations):
    schema = ["p"]
    for relation in relations:
        schema.extend([relation, "p"])
    return schema


def test_pure_helper_averages_jaccards_not_pooled_edges():
    sources = [
        [{1}, {1}],
        [{2}, {3, 4, 5}],
    ]
    matrix = mean_source_jaccards(iter(sources))
    assert matrix[0, 1] == 0.5
    pooled_left = sources[0][0] | sources[1][0]
    pooled_right = sources[0][1] | sources[1][1]
    pooled = len(pooled_left & pooled_right) / len(pooled_left | pooled_right)
    assert pooled == 0.2
    assert matrix[0, 1] != pooled
    np.testing.assert_array_equal(matrix, [[1, 0.5], [0.5, 1]])


def test_empty_unions_are_zero_and_sources_are_not_dropped():
    matrix = mean_source_jaccards([
        [{1}, {1}, set()],
        [set(), set(), set()],
    ])
    np.testing.assert_array_equal(
        matrix, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]],
    )


def test_exact_matrix_is_mean_over_the_same_fixed_sources():
    graph = _graph(4, r=[(0, 2), (1, 3)], s=[(2, 2)])
    matrix, metadata = edge_overlap_matrix(
        graph, [_path("r"), _path("r", "s")],
        source_ids=[0, 1], instances_per_path=4,
    )
    # Source 0: {r02} versus {r02, s22}: 1/2. Source 1: {r13}
    # versus an incomplete path's empty set: 0. Pooled Jaccard would be 1/3.
    np.testing.assert_array_equal(matrix, [[1, 0.25], [0.25, 0.5]])
    assert metadata["source_ids"] == [0, 1]
    assert metadata["path_statistics"] == [
        {"candidate_index": 0, "complete_instances": 8,
         "sources_with_edges": 2, "mean_unique_edges": 1.0},
        {"candidate_index": 1, "complete_instances": 4,
         "sources_with_edges": 1, "mean_unique_edges": 1.0},
    ]
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


def test_different_relations_and_directions_remain_distinct():
    graph = _graph(2, r=[(0, 1), (1, 0)], s=[(0, 1)])
    matrix, _ = edge_overlap_matrix(
        graph, [_path("r"), _path("s"), _path("s", "r"), _path("r", "r")],
        source_ids=[0],
    )
    np.testing.assert_allclose(matrix, [
        [1, 0, 0, 0.5],
        [0, 1, 0.5, 0],
        [0, 0.5, 1, 1 / 3],
        [0.5, 0, 1 / 3, 1],
    ])


def test_same_relation_name_with_different_node_types_is_distinct():
    graph = HeteroData()
    for node_type in ("p", "a", "b"):
        graph[node_type].num_nodes = 1
    for destination in ("a", "b"):
        graph["p", "r", destination].edge_index = torch.tensor([[0], [0]])
    matrix, _ = edge_overlap_matrix(
        graph, [["p", "r", "a"], ["p", "r", "b"]], source_ids=[0],
    )
    np.testing.assert_array_equal(matrix, np.eye(2))


def test_incomplete_instances_contribute_no_partial_edges():
    graph = _graph(3, r=[(0, 1)])
    matrix, metadata = edge_overlap_matrix(
        graph, [_path("r"), _path("r", "r")], source_ids=[0, 2],
    )
    np.testing.assert_array_equal(matrix, [[0.5, 0], [0, 0]])
    assert metadata["path_statistics"][1]["mean_unique_edges"] == 0.0
    empty, _ = edge_overlap_matrix(
        graph, [_path("r"), _path("r", "r")], source_ids=[2],
    )
    np.testing.assert_array_equal(empty, np.zeros((2, 2)))


def test_complete_mask_removes_only_failed_instances_not_whole_candidate():
    instances = PathInstances(
        target_ids=torch.tensor([0]),
        node_ids=torch.tensor([[[[0, 1, 3], [0, 2, -1]]]]),
        type_ids=torch.tensor([[0, 0, 0]]),
        lengths=torch.tensor([3]),
        node_types=("p",),
        schemas=(tuple(_path("r", "s")),),
    )
    sets = list(edge_overlap._packed_source_edges(instances, [[0, 16]], 2))
    # Only r(0,1) and s(1,3), never the failed r(0,2).
    assert sets == [[{1, 16 | (1 << 2) | 3}]]


def test_repeated_edges_within_and_across_instances_are_deduplicated():
    graph = _graph(1, r=[(0, 0), (0, 0)])
    matrix, metadata = edge_overlap_matrix(
        graph, [_path("r"), _path("r", "r", "r")],
        source_ids=[0], instances_per_path=32,
    )
    np.testing.assert_array_equal(matrix, np.ones((2, 2)))
    assert [row["mean_unique_edges"] for row in metadata["path_statistics"]] == [1, 1]


def test_integer_packing_is_collision_free_above_64_bits():
    big = 1 << 40
    bits = 41
    instances = PathInstances(
        target_ids=torch.tensor([big]),
        node_ids=torch.tensor([[
            [[big, 1]], [[big, 2]], [[big, 1]], [[1, big]],
        ]]),
        type_ids=torch.zeros((4, 2), dtype=torch.long),
        lengths=torch.tensor([2, 2, 2, 2]),
        node_types=("p",),
        schemas=tuple(tuple(_path("r")) for _ in range(4)),
    )
    sets = next(iter(edge_overlap._packed_source_edges(
        instances, [[0], [0], [1 << (2 * bits)], [0]], bits,
    )))
    keys = [next(iter(edges)) for edges in sets]
    assert all(type(key) is int for key in keys)
    assert keys[0] > np.iinfo(np.int64).max
    assert len(set(keys)) == 4
    np.testing.assert_array_equal(mean_source_jaccards([sets]), np.eye(4))


def test_sampling_is_deterministic_and_does_not_use_global_rng(toy_graph, toy_templates):
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    matrix, metadata = edge_overlap_matrix(
        toy_graph, toy_templates, max_sources=3, instances_per_path=8, seed=7,
    )
    repeated, repeated_metadata = edge_overlap_matrix(
        toy_graph, toy_templates, max_sources=3, instances_per_path=8, seed=7,
    )
    np.testing.assert_array_equal(matrix, repeated)
    assert metadata == repeated_metadata
    assert metadata["source_ids"] == sorted(
        np.random.default_rng(7).choice(6, size=3, replace=False).tolist()
    )
    assert len(set(metadata["source_ids"])) == 3
    assert torch.equal(torch_state, torch.random.get_rng_state())
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


def test_labels_and_feature_values_do_not_change_overlap(toy_graph, toy_templates):
    matrix, metadata = edge_overlap_matrix(toy_graph, toy_templates, seed=13)
    graph = toy_graph.clone()
    del graph["paper"].y
    graph["paper"].x.fill_(10000)
    graph["author"].x.zero_()
    without_labels, no_labels_metadata = edge_overlap_matrix(graph, toy_templates, seed=13)
    np.testing.assert_array_equal(matrix, without_labels)
    assert metadata == no_labels_metadata
    assert metadata["labels_used"] is False


def test_explicit_sources_are_preserved_and_override_max_sources(toy_graph, toy_templates):
    matrix, metadata = edge_overlap_matrix(
        toy_graph, [MetaPathTemplate.from_list(t) for t in toy_templates],
        source_ids=torch.tensor([4, 0]), max_sources=1,
    )
    assert matrix.shape == (3, 3)
    assert metadata["source_ids"] == [4, 0]
    assert metadata["num_sources"] == 2
    assert metadata["source_sampling"] == "explicit"


def test_sampling_batches_bound_memory_without_resampling_sources(monkeypatch, toy_graph, toy_templates):
    calls = []
    original_sample = MetaPathInstanceExtractor.sample

    def record_sample(self, ids, seed=None):
        calls.append(np.asarray(ids).tolist())
        return original_sample(self, ids, seed=seed)

    monkeypatch.setattr(MetaPathInstanceExtractor, "sample", record_sample)
    monkeypatch.setattr(edge_overlap, "_SOURCE_BATCH_SIZE", 2)
    matrix, metadata = edge_overlap_matrix(
        toy_graph, toy_templates, source_ids=[0, 1, 2, 3, 4],
    )
    assert calls == [[0, 1], [2, 3], [4]]
    assert metadata["source_ids"] == [0, 1, 2, 3, 4]
    assert metadata["source_batch_size"] == 2
    repeated, repeated_metadata = edge_overlap_matrix(
        toy_graph, toy_templates, source_ids=[0, 1, 2, 3, 4],
    )
    np.testing.assert_array_equal(matrix, repeated)
    assert metadata == repeated_metadata
    assert calls == [[0, 1], [2, 3], [4]] * 2


def _greedy_inputs():
    scores = np.array([10, 9, 8, 7, 6], dtype=float)
    matrix = np.eye(5)
    for p, q, value in [(0, 1, 0.9), (0, 3, 0.1), (0, 4, 0.2),
                        (2, 3, 1.0), (2, 4, 0.1)]:
        matrix[p, q] = matrix[q, p] = value
    return scores, matrix


def test_greedy_uses_max_overlap_with_all_previously_selected_candidates():
    scores, matrix = _greedy_inputs()
    selected, records = greedy_edge_overlap(scores, matrix, 5)
    assert selected == [0, 2, 4, 1, 3]
    assert records[0] == {
        "step": 1, "selected_index": 0, "TI_cov": 10.0,
        "max_overlap": 0.0, "novelty_factor": 1.0, "score": 10.0,
    }
    assert records[2]["max_overlap"] == 0.2
    assert records[2]["novelty_factor"] == 0.8
    assert records[2]["score"] == pytest.approx(6 * 0.8)
    assert records[3]["max_overlap"] == 0.9
    assert records[-1]["score"] == 0
    assert json.loads(json.dumps(records, allow_nan=False)) == records


def test_k_two_is_prefix_of_k_five_and_inputs_are_not_changed():
    scores, matrix = _greedy_inputs()
    old_scores, old_matrix = scores.copy(), matrix.copy()
    short, short_records = greedy_edge_overlap(scores, matrix, 2)
    long, long_records = greedy_edge_overlap(scores, matrix, 5)
    assert short == long[:2]
    assert short_records == long_records[:2]
    np.testing.assert_array_equal(scores, old_scores)
    np.testing.assert_array_equal(matrix, old_matrix)


def test_greedy_ties_follow_original_canonical_index_order():
    assert greedy_edge_overlap([2, 3, 3, 1], np.eye(4), 4)[0] == [1, 2, 0, 3]
    matrix = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    # After selecting 0, both candidates score 4 despite different TI_cov.
    assert greedy_edge_overlap([10, 8, 4], matrix, 3)[0] == [0, 1, 2]
    assert greedy_edge_overlap([0, 0, 0], np.zeros((3, 3)), 3)[0] == [0, 1, 2]
    assert greedy_edge_overlap([3, 1, 2], np.ones((3, 3)), 3)[0] == [0, 1, 2]


def test_zero_k_and_numpy_integer_k_are_supported():
    assert greedy_edge_overlap([], np.empty((0, 0)), 0) == ([], [])
    assert greedy_edge_overlap([1], [[0]], 0) == ([], [])
    assert greedy_edge_overlap([1], [[0]], np.int64(1))[0] == [0]


@pytest.mark.parametrize("scores", [
    [-1, 1], [np.nan, 1], [np.inf, 1], [[1, 2]], [1 + 2j, 1], ["1", "2"],
])
def test_greedy_rejects_invalid_scores(scores):
    with pytest.raises(ValueError, match="scores"):
        greedy_edge_overlap(scores, np.eye(2), 1)


@pytest.mark.parametrize("matrix", [
    [1, 0],
    np.eye(3),
    np.zeros((2, 3)),
    [[1, 0.1], [0.2, 1]],
    [[1, -0.1], [-0.1, 1]],
    [[1, 1.1], [1.1, 1]],
    [[1, np.nan], [np.nan, 1]],
    [[1, np.inf], [np.inf, 1]],
    [[1, 1j], [1j, 1]],
    [[1, 0], [1]],
])
def test_greedy_rejects_invalid_overlap_matrices(matrix):
    with pytest.raises(ValueError, match="overlaps"):
        greedy_edge_overlap([1, 2], matrix, 1)


@pytest.mark.parametrize("k", [-1, 3, 1.5, True])
def test_greedy_rejects_invalid_k_instead_of_shortening(k):
    with pytest.raises(ValueError, match="k"):
        greedy_edge_overlap([1, 2], np.eye(2), k)


@pytest.mark.parametrize("source_ids", [
    [], [[0]], [0.0], [-1], [6], [0, 0], [True], ["0"],
])
def test_sampling_rejects_invalid_source_ids(source_ids, toy_graph, toy_templates):
    with pytest.raises(ValueError, match="source_ids"):
        edge_overlap_matrix(toy_graph, toy_templates, source_ids=source_ids)


@pytest.mark.parametrize("kwargs", [
    {"max_sources": 0}, {"max_sources": -1}, {"max_sources": 1.5},
    {"max_sources": True}, {"instances_per_path": 0}, {"instances_per_path": 1.5},
    {"seed": -1}, {"seed": 1 << 64}, {"seed": 0.5},
])
def test_sampling_rejects_invalid_budgets_and_seeds(kwargs, toy_graph, toy_templates):
    with pytest.raises(ValueError):
        edge_overlap_matrix(toy_graph, toy_templates, **kwargs)


def test_sampling_rejects_empty_or_incompatible_templates(toy_graph, toy_templates):
    with pytest.raises(ValueError):
        edge_overlap_matrix(toy_graph, [])
    with pytest.raises(ValueError, match="unknown edge type"):
        edge_overlap_matrix(toy_graph, [["paper", "missing", "paper"]])
    with pytest.raises(ValueError, match="does not start"):
        edge_overlap_matrix(toy_graph, [toy_templates[0], ["author", "to", "paper"]])
    with pytest.raises(ValueError, match="at least one node"):
        edge_overlap_matrix(_graph(0, r=[]), [_path("r")])


def test_pure_helper_rejects_empty_or_mismatched_inputs():
    for sources in ([], [[]], [[{1}], [{1}, {2}]]):
        with pytest.raises(ValueError):
            mean_source_jaccards(sources)
