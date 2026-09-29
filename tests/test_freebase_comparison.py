"""Regression checks for the separately frozen Freebase comparison."""

import json

import pytest

from experiments import freebase_comparison as campaign

pytestmark = pytest.mark.skipif(
    not campaign.LOCK.exists(),
    reason="requires locally generated Freebase protocol and frozen V4 result files",
)


def test_frozen_freebase_inputs_and_reference_matrix():
    lock = campaign.check_lock(verify_sources=True)
    assert lock["ks"] == [2, 5]
    assert lock["seeds"] == list(range(5))
    assert len(lock["references"]) == 65
    assert all("freebase" in path for path in lock["references"])
    assert lock["config"]["feat"] == "target_onehot"


def test_selection_is_complete_and_shared_across_seeds():
    lock = campaign.check_lock(verify_sources=True)
    selected = campaign.selection(lock)
    assert len(selected["paths"]) == 14
    assert len(selected["overlap"]) == len(selected["ti_cov"]) == 14
    assert len(selected["selected"]["5"]) == 5
    assert selected["selected"]["2"] == selected["selected"]["5"][:2]
    assert len(set(selected["indices"])) == 5
    assert selected["sampling"]["labels_used"] is False
    assert selected["sampling"]["aggregation"] == "mean_of_per_source_jaccards"


def test_partial_aggregation_does_not_claim_completion():
    report = campaign.aggregate()
    assert report["complete"] == all(row["complete"] for row in report["results"])
    assert len(report["results"]) == 32
    assert len(report["comparisons"]) == 28
    for k in campaign.KS:
        expected = sum(campaign.result_path(k, seed).exists() for seed in campaign.SEEDS)
        own = [row for row in report["results"] if row["method"] == "edgeoverlap" and row["k"] == k]
        assert len(own) == 2
        assert all(row["n"] == expected for row in own)
    events = campaign.OUT / "test_evaluations.jsonl"
    if events.exists():
        seen = [(event["k"], event["seed"]) for event in
                map(json.loads, events.read_text().splitlines())]
        assert len(seen) == len(set(seen))
