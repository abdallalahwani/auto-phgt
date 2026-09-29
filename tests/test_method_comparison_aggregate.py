import json

import pytest

from experiments.method_comparison import aggregate, plan


@pytest.fixture
def aggregate_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    monkeypatch.setattr(plan, "OUT", tmp_path / "v5")
    references = {}
    for baseline in ("ti", "ti_set"):
        for k in plan.KS:
            for seed in plan.SEEDS:
                name = f"acm__{baseline}_k{k}__seed{seed}"
                path = tmp_path / f"{name}.json"
                path.write_text(json.dumps({"test": {"micro_f1": 0.7, "macro_f1": 0.6}}))
                references[name] = {"path": path.name, "sha256": plan.digest(path)}
    lock = {"protocol_hash": "test-only", "secondary_acm_references": references}
    monkeypatch.setattr(plan, "check_lock", lambda: lock)
    return lock


def record(spec):
    return {"test": {"micro_f1": 0.7 + spec["seed"] * 0.001,
                     "macro_f1": 0.6 + spec["seed"] * 0.001},
            "parameters": 100,
            "fit": {"train_runtime_s": 1.0}, "environment": {"gpu_name": "test-device"}}


def test_partial_rows_cannot_be_reported_complete(aggregate_lock, monkeypatch):
    first = plan.tasks()[0]["id"]
    monkeypatch.setattr(plan, "final_record",
                        lambda spec, lock: record(spec) if spec["id"] == first else None)
    result = aggregate.aggregate()
    assert result["complete"] is False
    assert result["completed"] == 1 and len(result["missing"]) == 164
    assert all(not row["complete"] for row in result["main"])
    assert all(not comparison["complete"] for comparison in result["paired"])
    assert (plan.OUT / "summary/final_summary.md").exists()


def test_full_matrix_counts_hgt_once(aggregate_lock, monkeypatch):
    monkeypatch.setattr(plan, "final_record", lambda spec, lock: record(spec))
    result = aggregate.aggregate()
    assert result["complete"] is True
    assert result["completed"] == 165 and not result["missing"]
    assert result["by_dataset"] == {"acm": 55, "dblp": 55, "imdb": 55}
    assert len(result["main"]) == 33
    assert all(row["n"] == 5 for row in result["main"])
    assert all(comparison["complete"] for comparison in result["paired"])
    assert all(reference["complete"] for reference in result["secondary_acm"])
