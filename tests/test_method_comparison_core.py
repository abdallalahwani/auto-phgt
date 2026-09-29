import json
from collections import Counter

import pytest
import torch

from experiments.method_comparison import plan, selectors, worker


def example_record(spec, config):
    paths = [["paper", "to", "author", "to", "paper"],
             ["paper", "to", "term", "to", "paper"]]
    return {"dataset": spec["dataset"], "seed": spec["seed"], "mode": spec["mode"],
            "k": spec["k"], "lmax": spec["lmax"], "status": "completed",
            "selector": spec["method"], "token_control": None, "config": config,
            "templates": paths, "test": {"micro_f1": 0.7, "macro_f1": 0.6}}


def test_requested_matrix_has_no_mag_or_duplicate_hgt():
    tasks = plan.tasks()
    assert len(tasks) == 165
    assert Counter(task["dataset"] for task in tasks) == {"acm": 55, "dblp": 55, "imdb": 55}
    assert {task["method"] for task in tasks} == set(plan.METHODS)
    assert len({task["id"] for task in tasks}) == 165
    for dataset in plan.DATASETS:
        hgt = [task for task in plan.tasks(dataset) if task["method"] == "hgt"]
        assert len(hgt) == 5 and all(task["k"] is None for task in hgt)
    with pytest.raises(ValueError, match="unknown dataset"):
        plan.tasks("ogbn-mag")


def test_dblp_search_spaces_are_feasible_and_explicit():
    assert {task["lmax"] for task in plan.tasks("dblp") if task["k"] == 2} == {4}
    assert {task["lmax"] for task in plan.tasks("dblp") if task["k"] == 5} == {6}
    assert {task["lmax"] for task in plan.tasks("imdb")} == {4}


def test_legacy_provenance_unwraps_v4_but_not_v3(monkeypatch):
    acm = {"dataset": "acm", "processed_md5": "acm-hash"}
    dblp = {"dataset": "dblp", "processed_md5": "dblp-hash"}
    def read(path):
        if "v4_hedge" in path.parts:
            return {"acm": {"record": acm, "matches_expected": True}}
        return {"dblp": dblp}
    monkeypatch.setattr(plan, "read_json", read)
    assert plan.legacy_dataset_records() == {"acm": acm, "dblp": dblp}


def test_source_paths_reuse_independent_fastpath_and_correct_backbones():
    acm = next(task for task in plan.tasks("acm") if task["method"] == "fastpath")
    dblp = next(task for task in plan.tasks("dblp") if task["method"] == "canonical" and task["k"] == 5)
    assert plan.source_path(acm).name == "acm__fastpath_k2__seed0.json"
    assert plan.source_path(dblp).name == "dblp__canonical_k5_L6__seed0.json"
    assert plan.source_path(next(task for task in plan.tasks("imdb"))) is None
    assert plan.source_path(next(task for task in plan.tasks("dblp") if task["method"] == "fastpath")) is None


@pytest.mark.parametrize("field,value", [
    ("config", {"layers": 3}), ("seed", 99), ("selector", "fastpath_set"),
    ("k", 5), ("lmax", 6), ("status", "failed"), ("token_control", "dummy"),
    ("width", 128), ("templates", []), ("test", {}),
])
def test_reuse_rejects_incompatible_results(field, value):
    spec = next(task for task in plan.tasks("acm") if task["method"] == "canonical" and task["k"] == 2)
    cfg = {"layers": 2}
    record = example_record(spec, cfg)
    plan.validate_reuse(spec, record, cfg)
    record[field] = value
    with pytest.raises(ValueError):
        plan.validate_reuse(spec, record, cfg)


def test_reference_reuse_is_read_only_and_checks_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    monkeypatch.setattr(plan, "OUT", tmp_path / "new")
    spec = next(task for task in plan.tasks("acm") if task["method"] == "canonical" and task["k"] == 2)
    cfg = {"layers": 2}
    source = tmp_path / "old.json"
    source.write_text(json.dumps(example_record(spec, cfg)))
    lock = {"configs": {"acm": cfg}, "reuse": {
        spec["id"]: {"path": "old.json", "sha256": plan.digest(source)}}}
    result = plan.final_record(spec, lock)
    assert result["reused_from"] == "old.json"
    assert not plan.result_path(spec).exists()
    source.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        plan.final_record(spec, lock)


def test_selector_cache_is_protocol_bound(tmp_path):
    path = tmp_path / "cache.json"
    assert selectors.read_cache(path, {"protocol_hash": "a"}) is None
    path.write_text(json.dumps({"protocol_hash": "b"}))
    with pytest.raises(ValueError, match="another protocol"):
        selectors.read_cache(path, {"protocol_hash": "a"})


def test_multilabel_hybrid_uses_only_training_labels(toy_graph, toy_templates, monkeypatch):
    monkeypatch.setattr(selectors, "MIN_HOMOPHILY_SUPPORT", 1)
    toy_graph["paper"].y = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 0], [0, 0], [1, 1]])
    train = torch.tensor([0, 1, 2, 3])
    first = selectors.multilabel_homophily(toy_graph, toy_templates, train, target="paper", seed=3)
    assert first["chance"] == pytest.approx(0.5625)
    toy_graph["paper"].y[4:] = -999
    second = selectors.multilabel_homophily(toy_graph, toy_templates, train, target="paper", seed=3)
    assert first == second
    assert all(row["support"] >= 0 for row in first["paths"])


def test_pilot_never_evaluates_test(toy_graph, tmp_path, monkeypatch):
    toy_graph["paper"].train_mask = torch.tensor([True, True, True, True, True, False])
    toy_graph["paper"].test_mask = ~toy_graph["paper"].train_mask
    config = {**plan.BACKBONE_FIXED, "feat": "given", "layers": 1, "lr": 0.001,
              "wd": 0.0, "dropout": 0.0, "d_model": 16, "heads": 2}
    monkeypatch.setattr(plan, "OUT", tmp_path)
    monkeypatch.setattr(plan, "check_lock", lambda dataset: {
        "protocol_hash": "toy", "configs": {"acm": config}})
    monkeypatch.setattr(plan, "load_graph", lambda dataset: toy_graph)
    monkeypatch.setattr(worker, "cuda_device", lambda: torch.device("cpu"))
    def forbidden(*args, **kwargs):
        raise AssertionError("pilot touched the test evaluator")
    monkeypatch.setattr(worker.train, "evaluate_test", forbidden)
    spec = next(task for task in plan.tasks("acm")
                if task["method"] == "canonical" and task["k"] == 5)
    result = worker.run(spec, pilot=True)
    assert result["status"] == "pilot_completed"
    assert "test" not in result
    assert not (tmp_path / "test_evaluations.jsonl").exists()
    assert not plan.result_path(spec).exists()
    assert result["fit"]["epochs_completed"] == 3
