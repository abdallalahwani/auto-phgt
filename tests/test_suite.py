import json
import subprocess
import sys
import time

import pytest
import torch

from auto_phgt.discovery import StatisticalDiscoveryModule, random_valid_paths
from experiments import aggregate as aggregation
from experiments import baseline_suite as suite
from experiments.common import select_templates


def test_inventory_is_complete_unique_and_prioritised():
    specs = suite.inventory()
    assert len(specs) == 32
    groups = {}
    for spec in specs:
        groups[spec["group"]] = groups.get(spec["group"], 0) + 1
    assert groups == {"acm_main": 15, "acm_discovery_ablation": 5, "acm_k_sensitivity": 6,
                      "mag_main": 6}
    assert [s["priority"] for s in specs] == sorted(s["priority"] for s in specs)
    for key in ("id", "result", "checkpoint", "log"):
        assert len({s[key] for s in specs}) == 32
    ids = {s["id"] for s in specs}
    assert {"acm_hgt_seed0_discovered_k5", "acm_auto_phgt_seed0_random_k5",
            "acm_auto_phgt_seed0_discovered_k1", "mag_hgt_seed0_discovered_k5",
            "mag_auto_phgt_seed2_discovered_k5"} <= ids
    assert not any(s["dataset"] == "ogbn-mag" and s["mode"] == "tokens" for s in specs)
    assert specs[15]["id"] == "mag_hgt_seed0_discovered_k5"  # first MAG pair right after ACM main


def fake_specs(tmp_path, n=6, fail=None):
    specs = []
    for i in range(n):
        exp_id = f"acm_hgt_seed{i}_discovered_k5"
        specs.append({"id": exp_id, "priority": 1, "group": "acm_main", "dataset": "acm",
                      "mode": "hgt", "seed": i, "path_selection": "discovered", "k": 5,
                      "threads": 3 if i % 2 else 2,
                      "result": str(tmp_path / "artifacts/results" / f"{exp_id}.json"),
                      "checkpoint": str(tmp_path / "artifacts/checkpoints" / f"{exp_id}.pt"),
                      "log": str(tmp_path / "artifacts/logs" / f"{exp_id}.log"),
                      "fail": exp_id == fail})
    return specs


def fake_launcher(tmp_path, launched):
    trace = tmp_path / "trace.jsonl"

    def launch(spec, worker, gpu):
        launched.append(spec["id"])
        result = {"status": "completed", "experiment_id": spec["id"]}
        code = (f"import json, time, pathlib\n"
                f"t0 = time.time(); time.sleep(0.3)\n"
                f"open({str(trace)!r}, 'a').write(json.dumps({{'gpu': {gpu!r}, 't0': t0, "
                f"'t1': time.time(), 'threads': {spec['threads']}}}) + '\\n')\n")
        if spec.get("fail"):
            code += "raise SystemExit(3)\n"
        else:
            code += (f"p = pathlib.Path({spec['result']!r}); p.parent.mkdir(parents=True, "
                     f"exist_ok=True); p.write_text(json.dumps({result!r}))\n")
        return subprocess.Popen([sys.executable, "-c", code])

    return launch, trace


def test_scheduler_pins_one_experiment_per_gpu_and_isolates_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    specs = fake_specs(tmp_path, fail="acm_hgt_seed2_discovered_k5")
    launched = []
    launch, trace = fake_launcher(tmp_path, launched)
    outcomes = suite.Suite(specs, ["5", "7", "2"], launch=launch, cpu_budget=6,
                           poll_s=0.05).run()
    assert outcomes["acm_hgt_seed2_discovered_k5"] == "failed"
    assert sum(v == "completed" for v in outcomes.values()) == 5
    spans = [json.loads(line) for line in trace.read_text().splitlines()]
    assert {s["gpu"] for s in spans} <= {"5", "7", "2"}
    for a in spans:  # no two overlapping runs on one GPU, CPU budget never exceeded
        overlapping = [b for b in spans if b["t0"] < a["t1"] and a["t0"] < b["t1"]]
        assert len({b["gpu"] for b in overlapping}) == len(overlapping)
        assert sum(b["threads"] for b in overlapping) <= 6
    status = json.loads((tmp_path / "artifacts/status/acm_hgt_seed2_discovered_k5.json")
                        .read_text())
    assert status["status"] == "failed" and status["attempts"][-1]["returncode"] == 3

    launched.clear()
    specs[2]["fail"] = False
    outcomes = suite.Suite(specs, ["5", "7", "2"], launch=launch, cpu_budget=6,
                           poll_s=0.05).run()
    assert launched == ["acm_hgt_seed2_discovered_k5"]  # completed runs are skipped
    assert outcomes == {"acm_hgt_seed2_discovered_k5": "completed"}


def test_deadline_interrupts_running_children(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = fake_specs(tmp_path, n=1)[0]
    outcomes = suite.Suite([spec], ["0"], poll_s=0.05, deadline=time.time() + 0.5,
                           launch=lambda s, w, g: subprocess.Popen(
                               [sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)).run()
    assert outcomes == {spec["id"]: "interrupted"}


def test_child_environment_uses_slurm_gpu_ids():
    spec = suite.inventory()[15]
    env = suite.child_env(spec, worker=3, gpu="6")
    assert env["CUDA_VISIBLE_DEVICES"] == "6" and env["AUTOPHGT_WORKER"] == "3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == str(spec["threads"])
    assert sum(suite.THREADS.values()) <= suite.CPU_BUDGET


def test_freeze_rejects_a_changed_protocol(tmp_path):
    specs = suite.inventory()
    path = tmp_path / "final_suite_config.json"
    first = suite.freeze(specs, path)
    again = suite.freeze(specs, path)
    assert again["frozen_at"] == first["frozen_at"] and len(again["launches"]) == 2
    record = json.loads(path.read_text())
    record["hyperparameters"]["acm"]["lr"] = 0.1
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="differs"):
        suite.freeze(specs, path)


def test_random_paths_are_deterministic_valid_candidates(toy_graph):
    candidates = StatisticalDiscoveryModule(toy_graph, max_hops=4).discover_candidate_paths()
    first = random_valid_paths(toy_graph, max_hops=4, k=3, seed=4)
    assert first == random_valid_paths(toy_graph, max_hops=4, k=3, seed=4)
    paths, indices, count = first
    assert count == len(candidates) and len(set(indices)) == 3
    assert paths == [candidates[i] for i in indices]
    assert any(random_valid_paths(toy_graph, max_hops=4, k=3, seed=s)[0] != paths
               for s in range(5, 10))
    with pytest.raises(ValueError):
        random_valid_paths(toy_graph, max_hops=4, k=count + 1, seed=0)


def test_select_templates_records_paths(tmp_path, toy_graph):
    engine = StatisticalDiscoveryModule(toy_graph, max_hops=4)
    templates, record = select_templates(toy_graph, dataset="toy", selection="discovered",
                                         k=2, max_hops=4, seed=0, template_dir=tmp_path)
    assert [list(t.schema) for t in templates] == engine.get_top_k_metapaths(2)
    assert [p["canonical_rank"] for p in record["paths"]] == [1, 2]
    assert len(record["canonical_ranking"]) == record["candidate_count"]
    templates, record = select_templates(toy_graph, dataset="toy", selection="random", k=2,
                                         max_hops=4, seed=1, template_dir=tmp_path)
    expected = random_valid_paths(toy_graph, max_hops=4, k=2, seed=1)[0]
    assert [list(t.schema) for t in templates] == expected
    assert record["selection"] == "random" and record["random_seed"] == 1
    assert json.loads((tmp_path / "toy_random_k2_maxhops4_seed1.json").read_text())[
        "templates"] == expected


def test_aggregate_reports_every_planned_seed(tmp_path):
    specs = suite.inventory()
    for spec in specs:
        for key in ("result", "checkpoint", "log"):
            spec[key] = str(tmp_path / spec[key])
    (tmp_path / "final_suite_config.json").write_text(json.dumps(
        {"suite": "t", "experiments": specs}))
    done = [s for s in specs if s["group"] == "acm_main" and s["seed"] < 2]
    for i, spec in enumerate(done):
        path = tmp_path / spec["result"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": "completed", "experiment_id": spec["id"],
            "test": {"accuracy": 0.8 + i / 100, "macro_f1": 0.7}, "best_epoch": 5,
            "validation": {"accuracy": 0.9, "macro_f1": 0.85}, "epochs_completed": 25,
            "early_stopped": True, "training_runtime_s": 10.0, "parameters": 100,
            "history": [{"epoch": 1, "loss": 1.0}],
            "discovery": {"paths": [{"rank": 1, "path": ["paper", "to", "term", "to", "paper"],
                                     "score": 3.0, "canonical_rank": 1}]}}))
    failed = next(s for s in specs if s["id"] == "acm_hgt_seed2_discovered_k5")
    status = tmp_path / "status" / f"{failed['id']}.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "failed", "attempts": [
        {"status": "failed", "error_tail": "boom"}]}))
    summary = aggregation.aggregate(tmp_path, tmp_path / "summary")
    hgt = next(r for r in summary["tables"]["acm_main"] if r["condition"] == "HGT")
    assert hgt["seeds_planned"] == 5 and hgt["seeds_completed"] == 2
    assert hgt["failed_seeds"] == "2" and "3(pending)" in hgt["unfinished_seeds"]
    assert hgt["test_accuracy_mean"] == pytest.approx(0.815)
    assert summary["states"]["completed"] == 6 and summary["states"]["failed"] == 1
    for name in ("acm_main", "acm_discovery_ablation", "acm_k_sensitivity", "mag_main"):
        assert (tmp_path / "summary" / f"{name}.csv").exists()
    assert (tmp_path / "summary" / "raw" / "histories.csv").exists()
    diff = summary["paired_differences"]["acm_main: Auto-PHGT - HGT"]["test_accuracy"]
    assert diff["seeds"] == [0, 1]


def test_lock_blocks_a_second_running_suite(tmp_path, monkeypatch):
    path = tmp_path / "suite.lock"
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    first = suite.acquire_lock(path, job_running=lambda job: True)
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    with pytest.raises(SystemExit, match="already running in Slurm job 111"):
        suite.acquire_lock(path, job_running=lambda job: job == "111")
    second = suite.acquire_lock(path, job_running=lambda job: False)  # 111 ended: stale lock
    assert second["slurm_job_id"] == "222"
    suite.release_lock(first, path)
    assert path.exists()  # only the holder's own record is released
    suite.release_lock(second, path)
    assert not path.exists()


def test_select_subset_and_label():
    full = suite.inventory()
    subset = suite.select(full, "mag_*_seed2_discovered_k5,acm_auto_phgt_seed*_discovered_k1")
    assert [s["id"] for s in subset] == ["mag_hgt_seed2_discovered_k5",
                                         "mag_auto_phgt_seed2_discovered_k5",
                                         "acm_auto_phgt_seed0_discovered_k1",
                                         "acm_auto_phgt_seed1_discovered_k1",
                                         "acm_auto_phgt_seed2_discovered_k1"]
    assert suite.subset_label(subset, full) and suite.subset_label(full, full) is None
    assert suite.select(full, None) == full
    with pytest.raises(SystemExit, match="matches no experiment"):
        suite.select(full, "nothing_*")


def test_worker_skips_completed_and_claimed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = next(s for s in suite.inventory() if s["id"] == "mag_hgt_seed2_discovered_k5")
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    claims = tmp_path / "artifacts" / "claims"
    claims.mkdir(parents=True)
    (claims / f"{spec['id']}.json").write_text(json.dumps({"slurm_job_id": "111"}))
    monkeypatch.setattr(suite, "_job_running", lambda job: job == "111")
    with pytest.raises(SystemExit, match="running in Slurm job 111"):
        suite.run_worker(spec["id"])
    result = tmp_path / spec["result"]
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"status": "completed", "experiment_id": spec["id"]}))
    suite.run_worker(spec["id"])  # returns before importing torch or training
    monkeypatch.setattr(suite, "_job_running", lambda job: False)
    assert suite.claim(spec) is None  # a stale claim is taken over
    assert json.loads((claims / f"{spec['id']}.json").read_text())["slurm_job_id"] == "222"
