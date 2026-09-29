"""CPU-only scheduling tests with isolated, project-local persisted paths."""

import json
import os
import shutil
import signal
import subprocess
import uuid
from pathlib import Path

import pytest

from experiments.method_comparison import controller as ctl


def synthetic_specs(dataset):
    specs = []
    for method in ("hgt", "random", "fastpath", "canonical", "hybrid", "edgeoverlap"):
        for k in ((None,) if method == "hgt" else (2, 5)):
            for seed in range(5):
                specs.append({"id": f"{dataset}__{method}__k{k}__seed{seed}",
                              "dataset": dataset, "method": method, "mode": method,
                              "k": k, "seed": seed, "lmax": 6 if dataset == "dblp" and k == 5 else 4})
    return specs


class FakeProcess:
    def __init__(self, runtime, command, kwargs):
        self.runtime = runtime
        self.command = command
        self.kwargs = kwargs
        self.pid = 10_000_000 + len(runtime.processes)
        self.returncode = None
        self.kind = "aggregate" if command[-1] == "experiments.method_comparison.aggregate" else command[-2][2:]
        self.spec = runtime.by_id.get(command[-1]) if self.kind == "task" else None
        self.dataset = self.spec["dataset"] if self.spec else command[-1] if self.kind != "aggregate" else None
        self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        self.log = Path(kwargs["stdout"].name)
        self.started = runtime.now
        self.ended = None
        self.prepared = False
        self.attempt = 1 + sum(proc.spec == self.spec for proc in runtime.processes
                               if self.spec is not None and proc.kind == "task")
        self.duration = runtime.durations.get((self.kind, self.dataset),
                                             3 if self.kind == "prepare" else 1)
        if self.gpu:
            assert self.kind in ("task", "pilot")
            assert all(proc.gpu != self.gpu or proc.poll() is not None for proc in runtime.processes)
        else:
            assert self.kind in ("prepare", "aggregate")
        assert kwargs["start_new_session"] is True
        assert kwargs["close_fds"] is True
        assert all(kwargs["env"][key] == "1" for key in ctl.THREAD_ENV)
        assert kwargs["env"]["PYTORCH_NVML_BASED_CUDA_CHECK"] == "0"

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        runtime = self.runtime
        if self.kind == "prepare":
            elapsed = runtime.now - self.started
            for spec in runtime.specs[self.dataset]:
                if elapsed >= runtime.ready_after.get(spec["method"], self.duration):
                    runtime.publish_selector(spec)
        if runtime.now - self.started < self.duration:
            return None
        code = runtime.codes.get((self.kind, self.dataset), 0)
        if self.spec:
            codes = runtime.task_codes.get(self.spec["id"], [0])
            code = codes[min(self.attempt - 1, len(codes) - 1)]
            if (code == 0 and self.spec["id"] not in runtime.omit_finals) or self.spec["id"] in runtime.commit_on_failure:
                runtime.commit(self.spec)
            with self.log.open("a") as handle:
                handle.write(runtime.failure_log)
        elif self.kind == "pilot" and code == 0 and self.dataset not in runtime.omit_pilots:
            runtime.publish_pilot(self.dataset, self.gpu)
        elif self.kind == "aggregate" and code == 0 and runtime.aggregate_report is not None:
            ctl.atomic_json(runtime.out / "summary" / "summary.json", runtime.aggregate_report)
        self.returncode = code
        self.ended = runtime.now
        return code

    def wait(self, timeout=None):
        code = self.poll()
        if code is None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return code

    def kill(self):
        self.runtime.signals.append((self.pid, signal.SIGKILL))
        self.returncode = -signal.SIGKILL
        self.ended = self.runtime.now


class Runtime:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.out = root / "synthetic-artifacts"
        self.now = 100.0
        self.specs = {dataset: synthetic_specs(dataset) for dataset in ctl.DATASETS}
        self.by_id = {spec["id"]: spec for specs in self.specs.values() for spec in specs}
        self.lock = {"protocol_hash": "synthetic-protocol", "reuse": {}}
        self.processes = []
        self.signals = []
        self.checks = []
        self.codes = {}
        self.task_codes = {}
        self.durations = {}
        self.ready_after = {}
        self.omit_finals = set()
        self.omit_selectors = set()
        self.omit_pilots = set()
        self.commit_on_failure = set()
        self.bad_selector_hash = set()
        self.pilot_overrides = {}
        self.aggregate_report = None
        self.failure_log = ""
        self.ignore_term = False
        self.on_sleep = None
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setattr(ctl.plan, "ROOT", root)
        monkeypatch.setattr(ctl.plan, "OUT", self.out)
        monkeypatch.setattr(ctl.plan, "tasks", lambda ds: self.specs[ds])
        monkeypatch.setattr(ctl.plan, "check_lock", self.check_lock)
        monkeypatch.setattr(ctl.plan, "final_record", self.final_record)
        monkeypatch.setattr(ctl.plan, "required_selection", self.selection_path)
        monkeypatch.setattr(ctl.subprocess, "Popen", self.spawn)
        monkeypatch.setattr(ctl.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(ctl.time, "sleep", self.sleep)
        monkeypatch.setattr(ctl.os, "killpg", self.killpg)
        monkeypatch.setattr(ctl.os, "kill", self.kill)
        monkeypatch.setattr(ctl, "group_members", self.group_members)

    def check_lock(self, dataset, *, full_data_check):
        self.checks.append((dataset, full_data_check))
        return self.lock

    def result_path(self, spec):
        return self.out / "results" / spec["dataset"] / f"{spec['id']}.json"

    def commit(self, spec, *, reused=False):
        ctl.atomic_json(self.result_path(spec),
                        {"task": spec, "protocol_hash": self.lock["protocol_hash"], "status": "completed",
                         "reused_from": "synthetic-prior-result" if reused else None})

    def final_record(self, spec, lock):
        path = self.result_path(spec)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record["protocol_hash"] != lock["protocol_hash"] or record["task"] != spec:
            raise ValueError("synthetic invalid committed result")
        return record

    def selection_path(self, spec):
        if spec["method"] == "hgt":
            return None
        return self.out / "selectors" / spec["dataset"] / f"{spec['id']}.json"

    def publish_selector(self, spec):
        path = self.selection_path(spec)
        if path is None or path.exists() or spec["id"] in self.omit_selectors:
            return
        protocol_hash = "wrong" if spec["id"] in self.bad_selector_hash else self.lock["protocol_hash"]
        selected = [["synthetic", f"relation_{index}", "synthetic"] for index in range(spec["k"])]
        ctl.atomic_json(path, {"protocol_hash": protocol_hash, "dataset": spec["dataset"],
                              "selections": {spec["method"]: {str(spec["k"]): selected}}})

    def publish_pilot(self, dataset, gpu="0"):
        ctl.atomic_json(self.out / "pilots" / f"{dataset}.json",
                        {"protocol_hash": self.lock["protocol_hash"], "dataset": dataset,
                         "status": "pilot_completed", "pilot": True,
                         "task": self.choose(dataset, "canonical", k=5),
                         "config": {"max_epochs": 3, "patience": 3},
                         "fit": {"epochs_completed": 3},
                         "history": [{"epoch": epoch} for epoch in range(1, 4)],
                         "environment": {"device": "cuda", "cuda_available": True,
                                         "cuda_visible_devices": gpu},
                         "validation_only": True, **self.pilot_overrides})

    def complete_except(self, dataset, pending):
        pending_ids = {spec["id"] for spec in pending}
        for spec in self.specs[dataset]:
            if spec["id"] not in pending_ids:
                self.commit(spec, reused=True)

    def choose(self, dataset="acm", method="edgeoverlap", *, k=2, seed=0):
        return next(spec for spec in self.specs[dataset]
                    if (spec["method"], spec["k"], spec["seed"]) == (method, k, seed))

    def spawn(self, command, **kwargs):
        process = FakeProcess(self, command, kwargs)
        self.processes.append(process)
        return process

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()

    def killpg(self, pid, signum):
        assert any(proc.pid == pid for proc in self.processes), "must only signal owned groups"
        self.signals.append((pid, signum))
        if not self.ignore_term:
            for proc in self.processes:
                if proc.pid == pid and proc.poll() is None:
                    proc.returncode = -signum
                    proc.ended = self.now

    def kill(self, pid, signum):
        assert signum == signal.SIGKILL
        process = next(proc for proc in self.processes if proc.pid == pid)
        process.kill()

    def group_members(self, groups):
        return {proc.pid: (proc.pid, str(proc.pid))
                for proc in self.processes if proc.pid in groups and proc.poll() is None}

    def controller(self, datasets=None, gpus=None, **kwargs):
        return ctl.Controller(datasets or ["acm"], ["0", "1"] if gpus is None else gpus,
                              poll_seconds=0.25, shutdown_grace=0.5, **kwargs)

    def progress(self, dataset="acm"):
        return json.loads((self.out / "progress" / f"{dataset}.json").read_text())


@pytest.fixture
def runtime(monkeypatch):
    # Keep even editor-launched pytest runs out of system scratch directories.
    root = Path(__file__).resolve().parents[1] / ".cache" / "v5-controller-tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield Runtime(root, monkeypatch)
    finally:
        shutil.rmtree(root)


def test_zero_exit_skips_verified_cells_and_counts_all_55(runtime):
    pending = [spec for spec in runtime.specs["acm"] if spec["method"] == "edgeoverlap"]
    runtime.complete_except("acm", pending)
    controller = runtime.controller()
    assert controller.run(install_signals=False) == 0
    final_jobs = [proc for proc in runtime.processes if proc.kind == "task"]
    assert {proc.spec["id"] for proc in final_jobs} == {spec["id"] for spec in pending}
    assert len(final_jobs) == 10
    assert runtime.progress()["completed"] == 55
    assert runtime.progress()["reused"] == 45
    assert runtime.progress()["status"] == "completed"
    assert runtime.checks == [("acm", True)]
    assert json.loads(controller.manifest.read_text())["exit_code"] == 0


def test_all_verified_needs_no_cuda_prepare_or_pilot(runtime):
    runtime.complete_except("acm", [])
    assert runtime.controller(gpus=[]).run(install_signals=False) == 0
    assert [proc.kind for proc in runtime.processes] == ["aggregate"]
    assert runtime.progress()["completed"] == 55


def test_requested_dataset_success_does_not_require_global_aggregate_complete(runtime):
    runtime.complete_except("acm", [])
    runtime.aggregate_report = {"complete": False, "completed": 55, "by_dataset": {"acm": 55}}
    controller = runtime.controller(["acm"], gpus=[])
    assert controller.run(install_signals=False) == 0
    summary = json.loads((runtime.out / "summary" / "summary.json").read_text())
    assert summary["complete"] is False
    assert set(controller.states) == {"acm"}
    assert runtime.progress()["completed"] == 55
    assert not (runtime.out / "progress" / "imdb.json").exists()


def test_current_inventory_reuses_80_and_dispatches_exactly_85_new_finals(runtime):
    for dataset, methods in (("acm", {"edgeoverlap"}), ("dblp", {"fastpath", "edgeoverlap"})):
        runtime.complete_except(dataset, [spec for spec in runtime.specs[dataset]
                                         if spec["method"] in methods])
    controller = runtime.controller(["acm", "dblp", "imdb"], ["0", "1", "2"])
    assert controller.run(install_signals=False) == 0
    finals = [proc for proc in runtime.processes if proc.kind == "task"]
    assert len(finals) == 85
    assert {dataset: sum(proc.dataset == dataset for proc in finals)
            for dataset in ctl.DATASETS} == {"acm": 10, "dblp": 20, "imdb": 55}
    assert sum(runtime.progress(dataset)["reused"] for dataset in ctl.DATASETS) == 80
    assert all(runtime.progress(dataset)["completed"] == 55 for dataset in ctl.DATASETS)
    assert len([proc for proc in finals if proc.spec["method"] == "hgt"]) == 5


def test_empty_gpu_mask_never_falls_back_to_cpu(runtime):
    assert runtime.controller(gpus=[]).run(install_signals=False) != 0
    assert all(proc.kind == "aggregate" for proc in runtime.processes)
    assert "no CPU fallback" in runtime.progress()["errors"][0]


def test_hgt_starts_after_pilot_without_waiting_for_selection_prep(runtime):
    pending = [spec for spec in runtime.specs["imdb"] if spec["method"] == "hgt"]
    runtime.complete_except("imdb", pending)
    runtime.durations["prepare", "imdb"] = 20
    assert runtime.controller(["imdb"]).run(install_signals=False) == 0
    finals = [proc for proc in runtime.processes if proc.kind == "task"]
    pilot = next(proc for proc in runtime.processes if proc.kind == "pilot")
    prepare = next(proc for proc in runtime.processes if proc.kind == "prepare")
    assert len(finals) == 5
    assert {proc.spec["seed"] for proc in finals} == set(range(5))
    assert all(proc.started >= pilot.ended for proc in finals)
    assert min(proc.started for proc in finals) < prepare.ended
    assert pilot.started == prepare.started


def test_progressive_selectors_dispatch_before_cpu_prepare_finishes(runtime):
    canonical = runtime.choose(method="canonical")
    fastpath = runtime.choose(method="fastpath")
    runtime.complete_except("acm", [canonical, fastpath])
    runtime.ready_after["canonical"] = 1
    runtime.durations["prepare", "acm"] = 10
    assert runtime.controller().run(install_signals=False) == 0
    prepare = next(proc for proc in runtime.processes if proc.kind == "prepare")
    finals = {proc.spec["method"]: proc for proc in runtime.processes if proc.kind == "task"}
    assert finals["canonical"].started < prepare.ended
    assert finals["fastpath"].started >= prepare.ended


def test_dynamic_pool_is_shared_without_per_dataset_gpu_reservations(runtime):
    pending_acm = [runtime.choose(method="hgt", k=None)]
    pending_acm += [runtime.choose(seed=seed) for seed in (0, 1, 2)]
    pending_dblp = [runtime.choose("dblp", "hgt", k=None)]
    runtime.complete_except("acm", pending_acm)
    runtime.complete_except("dblp", pending_dblp)
    runtime.durations["task", "acm"] = 5
    assert runtime.controller(["acm", "dblp"]).run(install_signals=False) == 0
    acm_finals = [proc for proc in runtime.processes if proc.kind == "task" and proc.dataset == "acm"]
    assert {proc.gpu for proc in acm_finals} == {"0", "1"}
    assert runtime.progress("acm")["completed"] == runtime.progress("dblp")["completed"] == 55
    for dataset in ("acm", "dblp"):
        assert len([proc for proc in runtime.processes if proc.dataset == dataset and proc.kind == "prepare"]) == 1


def test_node_gpu_leases_coordinate_independent_dataset_controllers(runtime):
    first = runtime.controller(["acm"], ["GPU-aaaa"])
    second = runtime.controller(["dblp"], ["GPU-aaaa"])
    slot = first.take_gpu()
    assert slot is not None
    assert second.take_gpu() is None
    slot[1].release()
    acquired = second.take_gpu()
    assert acquired is not None
    acquired[1].release()
    assert first.manifest != second.manifest


def test_global_dataset_lock_is_held_in_python_and_never_unlinked(runtime):
    first = runtime.controller()
    first.acquire_datasets()
    lease = first.dataset_leases[0]
    assert lease.handle is not None
    assert not os.get_inheritable(lease.handle.fileno())
    path = runtime.out / "progress" / "acm.json"
    ctl.atomic_json(path, {"owner": "original"})
    other_host = runtime.controller()
    other_host.host = "another-host"
    assert other_host.run(install_signals=False) == 75
    assert json.loads(path.read_text()) == {"owner": "original"}
    assert lease.handle is not None
    lease.release()
    assert lease.path.exists()
    assert lease.acquire()
    lease.release()


@pytest.mark.parametrize("codes,expected,attempts", [([75, 0], 0, 2), ([75], 1, 2), ([1], 1, 1)])
def test_only_infrastructure_failures_retry_at_most_two_attempts(runtime, codes, expected, attempts):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.task_codes[spec["id"]] = codes
    assert runtime.controller().run(install_signals=False) == expected
    finals = [proc for proc in runtime.processes if proc.kind == "task"]
    assert len(finals) == attempts
    assert runtime.progress()["attempts"][spec["id"]] == attempts


def test_cuda_runtime_nonzero_is_retryable_without_test_metrics(runtime):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.task_codes[spec["id"]] = [1, 0]
    runtime.failure_log = "RuntimeError: CUDA out of memory"
    assert runtime.controller().run(install_signals=False) == 0
    assert runtime.progress()["attempts"][spec["id"]] == 2


def test_nonzero_after_commit_preserves_result_without_re_evaluation(runtime):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.task_codes[spec["id"]] = [75]
    runtime.commit_on_failure.add(spec["id"])
    assert runtime.controller().run(install_signals=False) == 1
    assert runtime.progress()["completed"] == 55
    assert runtime.progress()["attempts"][spec["id"]] == 1
    assert "retained, not retried" in runtime.progress()["errors"][0]
    count = len([proc for proc in runtime.processes if proc.kind == "task"])
    assert runtime.controller().run(install_signals=False) == 0
    assert len([proc for proc in runtime.processes if proc.kind == "task"]) == count


def test_zero_exit_without_committed_result_is_failure_not_retry(runtime):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.omit_finals.add(spec["id"])
    assert runtime.controller().run(install_signals=False) == 1
    assert runtime.progress()["completed"] == 54
    assert runtime.progress()["attempts"][spec["id"]] == 1
    assert "exit 0 without a verified final" in runtime.progress()["errors"][0]


def test_final_reverification_catches_a_previously_valid_result_removed_mid_run(runtime):
    new = runtime.choose()
    prior = runtime.choose(method="hgt", k=None)
    runtime.complete_except("acm", [new])

    def remove_prior_result():
        if any(proc.kind == "task" for proc in runtime.processes):
            runtime.result_path(prior).unlink()
            runtime.on_sleep = None

    runtime.on_sleep = remove_prior_result
    assert runtime.controller().run(install_signals=False) == 1
    assert runtime.progress()["completed"] == 54
    assert prior["id"] in runtime.progress()["pending_ids"]
    assert "final verification found a missing result" in runtime.progress()["errors"][0]


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_wrong_protocol_selection_is_an_explicit_dependency_failure(runtime, missing):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    (runtime.omit_selectors if missing else runtime.bad_selector_hash).add(spec["id"])
    assert runtime.controller().run(install_signals=False) == 1
    assert not any(proc.kind == "task" for proc in runtime.processes)
    assert "dependency verification failed" in runtime.progress()["errors"][0]
    assert ("missing/not ready" if missing else "protocol_hash") in runtime.progress()["errors"][0]


@pytest.mark.parametrize("selections", [{}, {"canonical": {"2": [["a", "r", "a"]]}}])
def test_matching_protocol_without_exactly_k_selected_paths_is_not_ready(runtime, selections):
    spec = runtime.choose(method="canonical")
    state = ctl.DatasetState("acm", runtime.specs["acm"], runtime.lock)
    ctl.atomic_json(runtime.selection_path(spec),
                    {"protocol_hash": runtime.lock["protocol_hash"], "selections": selections})
    with pytest.raises(ValueError, match="selected paths"):
        runtime.controller().selection_ready(state, spec)


def test_shared_cache_readiness_is_checked_for_each_method_and_k(runtime, monkeypatch):
    spec = runtime.choose(method="canonical")
    other_k = runtime.choose(method="canonical", k=5)
    other_method = runtime.choose(method="random")
    runtime.publish_selector(spec)
    shared = runtime.selection_path(spec)
    monkeypatch.setattr(ctl.plan, "required_selection", lambda _: shared)
    state = ctl.DatasetState("acm", runtime.specs["acm"], runtime.lock)
    controller = runtime.controller()
    assert controller.selection_ready(state, spec)
    for other in (other_k, other_method):
        with pytest.raises(ValueError, match="selected paths"):
            controller.selection_ready(state, other)


def test_failed_prepare_does_not_prevent_other_dataset_finishing(runtime):
    for dataset in ("acm", "dblp"):
        runtime.complete_except(dataset, [runtime.choose(dataset)])
    runtime.codes["prepare", "acm"] = 17
    assert runtime.controller(["acm", "dblp"]).run(install_signals=False) == 1
    assert runtime.progress("acm")["status"] == "failed"
    assert runtime.progress("dblp")["completed"] == 55
    assert "prepare dependency exited 17" in runtime.progress("acm")["errors"][0]


@pytest.mark.parametrize("overrides", [
    {"protocol_hash": "wrong"}, {"status": "failed"}, {"environment": {"device": "cpu"}},
    {"config": {"max_epochs": 4}}, {"test": {"ignored": 0.9}}, {"validation_only": False},
    {"history": [{"epoch": 1}]}, {"pilot": False}, {"fit": {"epochs_completed": 2}},
])
def test_pilot_must_be_valid_cuda_validation_only_completion(runtime, overrides):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.pilot_overrides = overrides
    assert runtime.controller().run(install_signals=False) == 1
    assert not any(proc.kind == "task" for proc in runtime.processes)


def test_stale_pilot_cannot_bypass_current_cuda_pilot(runtime):
    runtime.complete_except("acm", [runtime.choose()])
    runtime.publish_pilot("acm")
    runtime.omit_pilots.add("acm")
    assert runtime.controller().run(install_signals=False) == 1
    assert "fresh readiness record" in runtime.progress()["errors"][0]


def test_signal_preserves_already_committed_result_and_reaps_only_owned_groups(runtime):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.durations["task", "acm"] = 50
    runtime.ignore_term = True
    controller = runtime.controller()

    def stop_after_commit():
        if any(proc.kind == "task" for proc in runtime.processes):
            runtime.commit(spec)
            controller.request_stop(signal.SIGTERM)
            runtime.on_sleep = None

    runtime.on_sleep = stop_after_commit
    assert controller.run(install_signals=False) == 143
    assert runtime.progress()["completed"] == 55
    assert runtime.progress()["status"] == "interrupted"
    assert runtime.final_record(spec, runtime.lock) is not None
    assert all(proc.poll() is not None for proc in runtime.processes)
    assert any(sig == signal.SIGTERM for _, sig in runtime.signals)
    assert any(sig == signal.SIGKILL for _, sig in runtime.signals)
    assert not any(proc.kind == "aggregate" for proc in runtime.processes)
    assert all(lease.handle is None for lease in controller.dataset_leases)


def test_deadline_stops_dispatch_and_exposes_partial_completeness(runtime):
    spec = runtime.choose()
    runtime.complete_except("acm", [spec])
    runtime.durations["task", "acm"] = 50
    controller = runtime.controller(budget_hours=0.002)
    assert controller.run(install_signals=False) == 124
    assert runtime.progress()["completed"] == 54
    assert runtime.progress()["pending_ids"] == [spec["id"]]
    assert json.loads(controller.manifest.read_text())["stop_reason"] == "budget deadline"
    assert all(proc.started < controller.deadline for proc in runtime.processes)


def test_busy_node_pool_waits_for_lease_without_oversubscription(runtime):
    runtime.complete_except("acm", [runtime.choose()])
    occupied = runtime.controller(["dblp"], ["0"]).take_gpu()
    assert occupied is not None
    try:
        controller = runtime.controller(gpus=["0"], budget_hours=0.001)
        assert controller.run(install_signals=False) == 124
        assert not any(proc.kind in ("task", "pilot") for proc in runtime.processes)
        assert runtime.progress()["completed"] == 54
    finally:
        occupied[1].release()


def test_nonzero_pilot_stops_finals_and_reports_dependency_exit(runtime):
    runtime.complete_except("acm", [runtime.choose()])
    runtime.codes["pilot", "acm"] = 19
    assert runtime.controller().run(install_signals=False) == 1
    assert not any(proc.kind == "task" for proc in runtime.processes)
    assert "pilot dependency exited 19" in runtime.progress()["errors"][0]


def test_failed_worker_spawn_releases_its_gpu_lease(runtime, monkeypatch):
    controller = runtime.controller(gpus=["0"])
    state = ctl.DatasetState("acm", runtime.specs["acm"], runtime.lock)
    slot = controller.take_gpu()
    assert slot is not None

    def cannot_spawn(*args, **kwargs):
        raise OSError("synthetic process limit")

    monkeypatch.setattr(ctl.subprocess, "Popen", cannot_spawn)
    with pytest.raises(OSError, match="process limit"):
        controller.launch("pilot", state, slot=slot)
    assert slot[1].handle is None
    again = controller.take_gpu()
    assert again is not None
    again[1].release()


def test_aggregate_failure_is_not_reported_as_success(runtime):
    runtime.complete_except("acm", [])
    runtime.codes["aggregate", None] = 7
    controller = runtime.controller(gpus=[])
    assert controller.run(install_signals=False) == 1
    manifest = json.loads(controller.manifest.read_text())
    assert "aggregate exited 7" in manifest["errors"][0]
    assert runtime.progress()["completed"] == 55


@pytest.mark.parametrize("value", ["0,0", "00,0", "0,,1", "-1", "cpu", "GPU-aaaa,GPU-AAAA"])
def test_invalid_or_aliased_identical_gpu_identifiers_are_rejected(value):
    with pytest.raises(ValueError):
        ctl.gpu_tokens(value)


def test_cli_defaults_to_assigned_mask_and_three_hour_budget(monkeypatch):
    seen = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")

    def fake_run(self):
        seen.update(datasets=self.datasets, gpus=self.gpus, budget=self.budget_seconds)
        return 0

    monkeypatch.setattr(ctl.Controller, "run", fake_run)
    assert ctl.main(["--datasets", "acm", "dblp"]) == 0
    assert seen == {"datasets": ["acm", "dblp"], "gpus": ["GPU-aaaa", "GPU-bbbb"], "budget": 10800}


@pytest.mark.parametrize("argv", [
    ["--datasets", "mag"], ["--datasets", "acm", "acm"],
    ["--datasets", "acm", "--budget-hours", "nan"],
    ["--datasets", "acm", "--budget-hours", "0"],
])
def test_cli_rejects_excluded_datasets_duplicate_jobs_and_invalid_budgets(argv):
    with pytest.raises(SystemExit) as error:
        ctl.main(argv)
    assert error.value.code == 2


def test_inventory_requires_all_55_cells_with_hgt_only_once_per_seed():
    specs = synthetic_specs("imdb")
    ctl.validate_specs("imdb", specs)
    with pytest.raises(ValueError, match="55 unique cells"):
        ctl.validate_specs("imdb", specs[:-1])
    specs[-1] = dict(specs[0], id="imdb_duplicate_hgt")
    with pytest.raises(ValueError, match="HGT once per seed"):
        ctl.validate_specs("imdb", specs)
