"""Resume the frozen comparison with one CUDA worker per leased node GPU.

Dataset locks live in this process for its entire lifetime, including shutdown.
Selection preparation is CPU-only; a fresh validation-only pilot gates each
dataset's new finals. Scheduling never reads or compares evaluation metrics.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from experiments.method_comparison import plan

DATASETS = ("acm", "dblp", "imdb")
MAX_FINAL_ATTEMPTS = 2
THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with staging.open("x") as handle:
            json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


class LockBusy(RuntimeError):
    pass


class FileLease:
    """Never unlink lock files: replacing their inode defeats shared-FS flock."""

    def __init__(self, path: Path, metadata: dict):
        self.path = path
        self.metadata = metadata
        self.handle: IO[str] | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        os.set_inheritable(handle.fileno(), False)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        self.handle = handle
        try:
            handle.seek(0)
            handle.truncate()
            json.dump(self.metadata, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            self.release()
            raise
        return True

    def release(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def gpu_tokens(value: str) -> list[str]:
    if not value.strip():
        return []
    tokens = [part.strip() for part in value.split(",")]
    for token in tokens:
        if not re.fullmatch(r"(?:\d+|GPU-[0-9a-fA-F-]+|MIG-[0-9a-fA-F/-]+)", token):
            raise ValueError(f"invalid CUDA device identifier: {token!r}")
    normalized = [str(int(token)) if token.isdigit() else token.lower() for token in tokens]
    if len(set(normalized)) != len(tokens):
        raise ValueError("duplicate CUDA devices would oversubscribe a GPU")
    return [str(int(token)) if token.isdigit() else token for token in tokens]


def validate_specs(dataset: str, specs: list[dict]) -> None:
    expected = {("hgt", None, seed) for seed in range(5)}
    expected |= {
        (method, k, seed)
        for method in ("random", "fastpath", "canonical", "hybrid", "edgeoverlap")
        for k in (2, 5) for seed in range(5)
    }
    cells = [(spec["method"], spec["k"], spec["seed"]) for spec in specs]
    ids = [spec["id"] for spec in specs]
    if len(cells) != 55 or set(cells) != expected or len(set(ids)) != 55:
        raise ValueError(f"{dataset}: expected exactly 55 unique cells, with HGT once per seed")
    if any(spec["dataset"] != dataset for spec in specs):
        raise ValueError(f"{dataset}: mixed-dataset task inventory")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", task_id) for task_id in ids):
        raise ValueError(f"{dataset}: unsafe task identifier")


def file_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def infrastructure_failure(code: int, log: Path) -> bool:
    if code in (75, 137, 143, -signal.SIGKILL, -signal.SIGTERM, -signal.SIGBUS):
        return True
    if code != 1:
        return False
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 65536))
        tail = handle.read().decode(errors="replace").lower()
    return any(marker in tail for marker in (
        "cuda out of memory", "cuda error:", "cuda driver",
        "resource temporarily unavailable", "cannot allocate memory",
        "input/output error", "errno 5", "errno 11", "errno 12",
    ))


def group_members(groups: set[int]) -> dict[int, tuple[int, str]]:
    """Capture PID start times so forced cleanup cannot target a recycled PID."""
    members = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            pgid = int(fields[2])
            if pgid in groups and fields[0] != "Z":
                members[int(entry.name)] = (pgid, fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return members


@dataclass
class DatasetState:
    name: str
    specs: list[dict]
    protocol: dict
    completed: dict[str, bool] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    prepare: str = "pending"
    pilot: str = "pending"
    status: str = "starting"


@dataclass
class Child:
    process: subprocess.Popen
    kind: str
    dataset: str | None
    log: Path
    task: dict | None = None
    gpu: str | None = None
    lease: FileLease | None = None
    pilot_before: tuple[int, int, int] | None = None


class Controller:
    def __init__(
        self, datasets: list[str], gpus: list[str], budget_hours: float = 3,
        *, poll_seconds: float = 1, shutdown_grace: float = 90,
    ):
        if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
            raise ValueError("choose unique datasets from acm, dblp, imdb; MAG is excluded")
        if not math.isfinite(budget_hours) or budget_hours <= 0:
            raise ValueError("budget-hours must be a finite positive number")
        self.datasets = datasets
        self.gpus = gpu_tokens(",".join(gpus))
        self.budget_seconds = budget_hours * 3600
        self.poll_seconds = poll_seconds
        self.shutdown_grace = shutdown_grace
        self.root = plan.ROOT
        self.out = plan.OUT
        self.host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname().split(".")[0])
        self.run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.manifest = self.out / "nodes" / self.host / f"{self.run_id}.json"
        self.states: dict[str, DatasetState] = {}
        self.children: list[Child] = []
        self.dataset_leases: list[FileLease] = []
        self.errors: list[str] = []
        self.selection_cache: dict[tuple[Path, str, int], tuple[int, int, int]] = {}
        self.stop_reason: str | None = None
        self.stop_code = 1
        self.deadline = 0.0
        self.next_dataset = 0
        self.next_gpu = 0
        self.last_snapshot = float("-inf")
        self.phase = "starting"
        self.exit_code: int | None = None

    def metadata(self) -> dict:
        return {"run_id": self.run_id, "host": self.host, "pid": os.getpid()}

    def event(self, event: str, dataset: str | None = None, **fields) -> None:
        record = {**self.metadata(), "time": time.time(), "event": event, **fields}
        if dataset is not None:
            record["dataset"] = dataset
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        names = [dataset] if dataset else list(self.states)
        for name in names:
            path = self.out / "logs" / name / f"controller-{self.run_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(line + "\n")

    def acquire_datasets(self) -> None:
        for dataset in sorted(self.datasets):
            lease = FileLease(self.out / "locks" / f"{dataset}.lock", self.metadata())
            if not lease.acquire():
                raise LockBusy(f"{dataset}: another controller holds the global dataset lock")
            self.dataset_leases.append(lease)

    def fail(self, state: DatasetState, message: str) -> None:
        state.errors.append(message)
        state.status = "failed"
        self.event("dataset_failed", state.name, error=message)

    def completed_record(self, state: DatasetState, spec: dict) -> bool:
        try:
            record = plan.final_record(spec, state.protocol)
        except (OSError, ValueError, KeyError, TypeError):
            state.completed.pop(spec["id"], None)
            raise
        if record is None:
            state.completed.pop(spec["id"], None)
            return False
        state.completed[spec["id"]] = bool(record.get("reused_from"))
        return True

    def initialize(self) -> None:
        for dataset in self.datasets:
            specs = plan.tasks(dataset)
            validate_specs(dataset, specs)
            state = DatasetState(dataset, specs, {})
            self.states[dataset] = state
            try:
                state.protocol = plan.check_lock(dataset, full_data_check=True)
                for spec in specs:
                    self.completed_record(state, spec)
                if len(state.completed) == len(specs):
                    state.status = "completed"
                    state.prepare = state.pilot = "not_needed"
                elif not self.gpus:
                    self.fail(state, "new work requires an explicit nonempty CUDA GPU mask; no CPU fallback")
                else:
                    state.status = "running"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.fail(state, f"inventory/protocol verification failed: {exc}")

    def selection_ready(self, state: DatasetState, spec: dict) -> bool:
        path = plan.required_selection(spec)
        if path is None:
            return True
        signature = file_signature(path)
        if signature is None:
            return False
        cache_key = (path, spec["method"], spec["k"])
        if self.selection_cache.get(cache_key) == signature:
            return True
        with path.open() as handle:
            record = json.load(handle)
        if not isinstance(record, dict) or record.get("protocol_hash") != state.protocol["protocol_hash"]:
            raise ValueError(f"selector {path}: missing or mismatched protocol_hash")
        if record.get("dataset", state.name) != state.name:
            raise ValueError(f"selector {path}: wrong dataset")
        status = record.get("status", "completed")
        if status in ("pending", "running", "preparing"):
            return False
        if status not in ("completed", "ready"):
            raise ValueError(f"selector {path}: invalid readiness status {status!r}")
        selections = record.get("selections")
        by_method = selections.get(spec["method"]) if isinstance(selections, dict) else None
        selected = by_method.get(str(spec["k"])) if isinstance(by_method, dict) else None
        if (not isinstance(selected, list) or len(selected) != spec["k"]
                or any(not isinstance(item, list) or not item
                       or any(not isinstance(step, str) for step in item) for item in selected)
                or len({tuple(item) for item in selected}) != spec["k"]):
            raise ValueError(f"selector {path}: missing or invalid selected paths for "
                             f"{spec['method']} k={spec['k']}")
        if file_signature(path) != signature:
            return False
        self.selection_cache[cache_key] = signature
        return True

    def take_gpu(self) -> tuple[str, FileLease] | None:
        for offset in range(len(self.gpus)):
            index = (self.next_gpu + offset) % len(self.gpus)
            gpu = self.gpus[index]
            key = hashlib.sha256(gpu.lower().encode()).hexdigest()[:20]
            lease = FileLease(self.out / "locks" / "nodes" / self.host / f"gpu-{key}.lock",
                              {**self.metadata(), "gpu": gpu})
            if lease.acquire():
                self.next_gpu = (index + 1) % len(self.gpus)
                return gpu, lease
        return None

    def child_env(self, gpu: str | None) -> dict[str, str]:
        env = os.environ.copy()
        env.update({key: "1" for key in THREAD_ENV})
        env.update(CUDA_VISIBLE_DEVICES=gpu or "", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTORCH_NVML_BASED_CUDA_CHECK="0",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   PYTHONUNBUFFERED="1", XDG_CACHE_HOME=str(self.root / ".cache"),
                   MPLCONFIGDIR=str(self.root / ".cache" / "matplotlib"))
        return env

    def launch(
        self, kind: str, state: DatasetState | None = None, spec: dict | None = None,
        slot: tuple[str, FileLease] | None = None,
    ) -> None:
        gpu, lease = slot if slot else (None, None)
        dataset = state.name if state else None
        label = spec["id"] if spec else kind
        attempt = 1
        if kind == "task":
            assert state is not None and spec is not None
            attempt = state.attempts.get(spec["id"], 0) + 1
            state.attempts[spec["id"]] = attempt
        log = self.out / "logs" / (dataset or "aggregate") / f"{label}-{self.run_id}-a{attempt}.log"
        if kind == "aggregate":
            command = [sys.executable, "-u", "-m", "experiments.method_comparison.aggregate"]
        else:
            command = [sys.executable, "-u", "-m", "experiments.method_comparison.worker",
                       f"--{kind}", spec["id"] if spec else dataset]
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            before = file_signature(self.out / "pilots" / f"{dataset}.json") if kind == "pilot" else None
            with log.open("a") as output:
                process = subprocess.Popen(command, cwd=self.root, env=self.child_env(gpu),
                                           stdin=subprocess.DEVNULL, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           close_fds=True)
        except BaseException:
            if lease is not None:
                lease.release()
            raise
        self.children.append(Child(process, kind, dataset, log, spec, gpu, lease, before))
        if state is not None and kind in ("prepare", "pilot"):
            setattr(state, kind, "running")
        self.event("process_started", dataset, kind=kind, task=spec["id"] if spec else None,
                   gpu=gpu, worker_pid=process.pid, attempt=attempt, log=str(log))

    def verify_pilot(self, state: DatasetState, child: Child) -> None:
        path = self.out / "pilots" / f"{state.name}.json"
        signature = file_signature(path)
        if signature is None or signature == child.pilot_before:
            raise ValueError(f"{state.name}: pilot did not commit a fresh readiness record")
        with path.open() as handle:
            record = json.load(handle)
        if not isinstance(record, dict) or record.get("protocol_hash") != state.protocol["protocol_hash"]:
            raise ValueError(f"{state.name}: pilot protocol_hash mismatch")
        if (record.get("status") != "pilot_completed" or record.get("pilot") is not True
                or record.get("dataset") != state.name):
            raise ValueError(f"{state.name}: invalid pilot completion record")
        expected = next(spec for spec in state.specs
                        if spec["method"] == "canonical" and spec["k"] == 5 and spec["seed"] == 0)
        if record.get("task") != expected:
            raise ValueError(f"{state.name}: pilot must be canonical k=5, seed=0")
        if record.get("test") is not None or record.get("validation_only") is False:
            raise ValueError(f"{state.name}: pilot must not evaluate test data")
        environment = record.get("environment", {})
        if (not isinstance(environment, dict)
                or not str(environment.get("device", "")).startswith("cuda")
                or environment.get("cuda_available") is not True
                or environment.get("cuda_visible_devices") != child.gpu):
            raise ValueError(f"{state.name}: pilot did not use CUDA")
        config, fitted = record.get("config"), record.get("fit")
        if (not isinstance(config, dict) or config.get("max_epochs") != 3
                or not isinstance(fitted, dict) or fitted.get("epochs_completed") != 3
                or not isinstance(record.get("history"), list) or len(record["history"]) != 3):
            raise ValueError(f"{state.name}: pilot must use exactly three epochs")

    def consume_exit(self, child: Child, code: int, *, stopping: bool = False) -> None:
        self.event("process_exited", child.dataset, kind=child.kind, returncode=code,
                   task=child.task["id"] if child.task else None, log=str(child.log))
        if child.dataset is None:
            if code != 0:
                self.errors.append(f"aggregate exited {code}; see {child.log}")
            return
        state = self.states[child.dataset]
        try:
            if child.kind == "task":
                assert child.task is not None
                committed = self.completed_record(state, child.task)
                if stopping:
                    return
                if code == 0:
                    if not committed:
                        raise ValueError(f"{child.task['id']}: exit 0 without a verified final result")
                elif committed:
                    self.fail(state, f"{child.task['id']}: exited {code} after committing; result retained, not retried")
                elif (state.attempts[child.task["id"]] < MAX_FINAL_ATTEMPTS
                      and infrastructure_failure(code, child.log)):
                    self.event("retry_pending", state.name, task=child.task["id"], returncode=code)
                else:
                    self.fail(state, f"{child.task['id']}: exited {code}; see {child.log}")
            elif stopping:
                setattr(state, child.kind, "interrupted")
            elif code != 0:
                setattr(state, child.kind, "failed")
                self.fail(state, f"{child.kind} dependency exited {code}; see {child.log}")
            elif child.kind == "pilot":
                self.verify_pilot(state, child)
                state.pilot = "completed"
            else:
                state.prepare = "completed"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if child.kind in ("prepare", "pilot"):
                setattr(state, child.kind, "failed")
            self.fail(state, str(exc))

    def reap(self, *, stopping: bool = False, selected: list[Child] | None = None) -> None:
        for child in list(self.children) if selected is None else list(selected):
            code = child.process.poll()
            if code is None:
                continue
            code = child.process.wait()
            self.children.remove(child)
            try:
                self.consume_exit(child, code, stopping=stopping)
            finally:
                if child.lease is not None:
                    child.lease.release()

    def stop_children(self, selected: list[Child]) -> None:
        if not selected:
            return
        groups = {child.process.pid for child in selected}
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        until = time.monotonic() + self.shutdown_grace
        while True:
            active = [child for child in selected if child.process.poll() is None]
            members = group_members(groups)
            if not active and not members:
                break
            if time.monotonic() >= until:
                # Recheck identity immediately before signalling each owned PID.
                for pid, identity in members.items():
                    if group_members({identity[0]}).get(pid) == identity:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                for child in active:
                    if child.process.poll() is None:
                        child.process.kill()
                break
            time.sleep(min(self.poll_seconds, 0.25))
        reap_deadline = time.monotonic() + 10
        for child in selected:
            try:
                child.process.wait(timeout=max(0.001, reap_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self.errors.append(f"worker PID {child.process.pid} did not exit after SIGKILL")
        self.reap(stopping=True, selected=selected)

    def check_dependencies(self, state: DatasetState) -> None:
        if state.status != "running" or state.prepare != "completed":
            return
        pending = [spec for spec in state.specs if spec["id"] not in state.completed]
        missing = [str(plan.required_selection(spec)) for spec in pending
                   if not self.selection_ready(state, spec)]
        if missing:
            raise ValueError("preparation exited 0 but required selectors are missing/not ready: "
                             + ", ".join(sorted(set(missing))))

    def next_work(self) -> tuple[DatasetState, str, dict | None] | None:
        running_ids = {child.task["id"] for child in self.children if child.task is not None}
        for offset in range(len(self.datasets)):
            index = (self.next_dataset + offset) % len(self.datasets)
            state = self.states[self.datasets[index]]
            if state.status != "running":
                continue
            try:
                if state.pilot == "pending":
                    self.next_dataset = (index + 1) % len(self.datasets)
                    return state, "pilot", None
                if state.pilot != "completed":
                    continue
                pending = sorted(state.specs, key=lambda spec: spec["method"] != "hgt")
                for spec in pending:
                    if spec["id"] in state.completed or spec["id"] in running_ids:
                        continue
                    if not self.selection_ready(state, spec):
                        continue
                    # A killed worker may already have atomically committed its result.
                    if self.completed_record(state, spec):
                        continue
                    self.next_dataset = (index + 1) % len(self.datasets)
                    return state, "task", spec
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.fail(state, f"dependency verification failed: {exc}")
        return None

    def dispatch(self) -> None:
        while not self.stop_reason and time.monotonic() < self.deadline:
            work = self.next_work()
            if work is None:
                return
            slot = self.take_gpu()
            if slot is None:
                return
            self.check_deadline()
            if self.stop_reason:
                slot[1].release()
                return
            state, kind, spec = work
            try:
                self.launch(kind, state, spec, slot)
            except OSError as exc:
                self.fail(state, f"cannot start {kind}: {exc}")

    def snapshot(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_snapshot < 5:
            return
        self.last_snapshot = now
        datasets = {}
        for name, state in self.states.items():
            active = [child for child in self.children if child.dataset == name]
            record = {**self.metadata(), "dataset": name, "heartbeat_at": time.time(),
                      "status": state.status, "protocol_hash": state.protocol.get("protocol_hash"),
                      "prepare": state.prepare, "pilot": state.pilot, "errors": state.errors,
                      "total": len(state.specs), "completed": len(state.completed),
                      "reused": sum(state.completed.values()),
                      "completed_ids": sorted(state.completed),
                      "pending_ids": [spec["id"] for spec in state.specs if spec["id"] not in state.completed],
                      "attempts": state.attempts,
                      "active": [{"kind": child.kind, "pid": child.process.pid, "gpu": child.gpu,
                                  "task": child.task["id"] if child.task else None,
                                  "log": str(child.log)} for child in active]}
            atomic_json(self.out / "progress" / f"{name}.json", record)
            datasets[name] = record
        atomic_json(self.manifest, {**self.metadata(), "heartbeat_at": time.time(),
                                   "phase": self.phase, "exit_code": self.exit_code,
                                   "stop_reason": self.stop_reason, "gpus": self.gpus,
                                   "datasets": datasets, "errors": self.errors})

    def request_stop(self, signum: int, _frame=None) -> None:
        if self.stop_reason is None:
            self.stop_reason = f"signal {signal.Signals(signum).name}"
            self.stop_code = 128 + signum

    def check_deadline(self) -> None:
        if self.stop_reason is None and time.monotonic() >= self.deadline:
            self.stop_reason = "budget deadline"
            self.stop_code = 124

    def aggregate(self) -> None:
        lease = FileLease(self.out / "locks" / "aggregate.lock", self.metadata())
        self.check_deadline()
        if self.stop_reason:
            return
        while not lease.acquire():
            self.check_deadline()
            if self.stop_reason:
                return
            time.sleep(self.poll_seconds)
        try:
            self.check_deadline()
            if self.stop_reason:
                return
            self.launch("aggregate")
            until = min(self.deadline, time.monotonic() + 120)
            while self.children:
                self.reap()
                self.check_deadline()
                if self.stop_reason or time.monotonic() >= until:
                    self.errors.append("aggregation interrupted or exceeded its 120-second limit")
                    self.stop_children(list(self.children))
                    break
                self.snapshot()
                if self.children:
                    time.sleep(self.poll_seconds)
        finally:
            lease.release()

    def run(self, *, install_signals: bool = True) -> int:
        previous = {}
        acquired = False
        self.deadline = time.monotonic() + self.budget_seconds
        try:
            if install_signals:
                for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    previous[signum] = signal.signal(signum, self.request_stop)
            self.acquire_datasets()
            acquired = True
            self.initialize()
            for state in self.states.values():
                self.check_deadline()
                if state.status == "running" and not self.stop_reason:
                    try:
                        self.launch("prepare", state)
                    except OSError as exc:
                        self.fail(state, f"cannot start prepare: {exc}")
            self.phase = "running"
            self.snapshot(force=True)
            self.event("controller_ready", manifest=str(self.manifest), datasets=self.datasets)
            while True:
                self.reap()
                self.check_deadline()
                if self.stop_reason:
                    break
                for state in self.states.values():
                    try:
                        self.check_dependencies(state)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        self.fail(state, f"dependency verification failed: {exc}")
                    if state.status == "failed":
                        self.stop_children([child for child in self.children if child.dataset == state.name])
                self.dispatch()
                for state in self.states.values():
                    if (state.status == "running" and len(state.completed) == len(state.specs)
                            and state.prepare == "completed" and state.pilot == "completed"
                            and not any(child.dataset == state.name for child in self.children)):
                        state.status = "completed"
                        self.event("dataset_completed", state.name, completed=len(state.completed))
                self.snapshot()
                if not self.children and all(state.status in ("completed", "failed")
                                             for state in self.states.values()):
                    break
                time.sleep(self.poll_seconds)
            if not self.stop_reason:
                for state in self.states.values():
                    if state.status == "completed":
                        try:
                            for spec in state.specs:
                                if not self.completed_record(state, spec):
                                    raise ValueError(f"{spec['id']}: final verification found a missing result")
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            self.fail(state, str(exc))
                self.aggregate()
            self.exit_code = (self.stop_code if self.stop_reason else
                              0 if not self.errors and all(state.status == "completed"
                                                          for state in self.states.values()) else 1)
        except LockBusy as exc:
            self.errors.append(str(exc))
            self.event("lock_busy", error=str(exc))
            self.exit_code = 75
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self.event("controller_failed", error=self.errors[-1])
            self.exit_code = 1
        finally:
            try:
                self.stop_children(list(self.children))
                for state in self.states.values():
                    if state.status not in ("completed", "failed"):
                        state.status = "interrupted" if self.stop_reason else "failed"
                if acquired:
                    self.phase = "completed" if self.exit_code == 0 else "stopped" if self.stop_reason else "failed"
                    self.snapshot(force=True)
                    self.event("controller_finished", exit_code=self.exit_code, reason=self.stop_reason)
            finally:
                for lease in reversed(self.dataset_leases):
                    lease.release()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        return self.exit_code if self.exit_code is not None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, choices=DATASETS)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        help="comma-separated CUDA IDs/UUIDs; defaults to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--budget-hours", type=float, default=3)
    args = parser.parse_args(argv)
    try:
        controller = Controller(args.datasets, gpu_tokens(args.gpus), args.budget_hours)
    except ValueError as exc:
        parser.error(str(exc))
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
