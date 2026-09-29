"""Recover failed V4-Hedge Freebase final runs with exactly one worker per GPU.

This is deliberately separate from the frozen V4 master. It reuses the original frozen
task implementations and task IDs, writes results to their original V4 locations, and
changes no scientific setting. The only behavioral change is scheduling: one full-graph
Freebase process owns each RTX 2080 Ti, preventing the two-process GPU oversubscription
that caused the original CUDA out-of-memory failures.

    python -m experiments.lightweight_recovery --freeze
    python -m experiments.lightweight_recovery --dry-run
    python -m experiments.lightweight_recovery
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from auto_phgt.runtime import git_state, write_json
from experiments import lightweight_campaign as original
from experiments.lightweight_selection import plan

ROOT = plan.ROOT
V4 = plan.V4
RECOVERY = V4 / "recovery"
LOCK = RECOVERY / "protocol_lock.json"
MARGIN_SECONDS = 20 * 60
MAX_ATTEMPTS = 2


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    print(f"[v4-recovery {now_iso()}] {message}", flush=True)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def recovery_code_hashes() -> dict[str, str]:
    files = [Path(__file__)]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in files}


def read_status(task_id: str) -> dict:
    path = original.status_path(task_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def select_targets(specs: list[dict], status_reader=read_status) -> list[dict]:
    """Failed Freebase final tasks only, ordered P0 -> P1 -> P2 then by task ID."""
    targets = []
    for spec in specs:
        if spec["fn"] != "final" or spec["meta"].get("dataset") != "freebase":
            continue
        if status_reader(spec["id"]).get("status") == "failed":
            targets.append(spec)
    return sorted(targets, key=lambda spec: (spec["priority"], spec["id"]))


def freeze() -> dict:
    """Freeze the recovery scope after verifying the parent V4 lock is still valid."""
    original.check_lock()
    parent_lock = V4 / "protocol" / "protocol_lock.json"
    targets = select_targets(original.build_tasks())
    if not targets:
        raise SystemExit("no failed Freebase final tasks need recovery")
    records = []
    for spec in targets:
        status = read_status(spec["id"])
        records.append({
            "id": spec["id"],
            "condition": spec["args"]["cond_id"],
            "seed": spec["args"]["seed"],
            "priority": spec["priority"],
            "original_error": status.get("error"),
        })
    lock = {
        "suite": "autophgt_v4_hedge_recovery",
        "parent_protocol_hash": plan.protocol_hash(),
        "parent_lock_sha256": file_hash(parent_lock),
        "reason": ("The original V4 scheduler assigned two full-graph Freebase workers to each "
                   "10.57 GiB RTX 2080 Ti; all 62 failed tasks ended in CUDA OOM."),
        "scientific_protocol_change": None,
        "scheduling_change": "exactly one failed Freebase final task per physical GPU",
        "retry_policy": "at most two recovery attempts per target",
        "targets": records,
        "target_count": len(records),
        "priority_counts": {
            f"P{priority}": sum(r["priority"] == priority for r in records)
            for priority in (0, 1, 2)
        },
        "code": recovery_code_hashes(),
        "git": git_state(),
        "frozen_at": now_iso(),
        "host": socket.gethostname(),
    }
    write_json(LOCK, lock)
    return lock


def check_lock() -> dict:
    original.check_lock()
    if not LOCK.exists():
        raise SystemExit(f"{LOCK} missing: run --freeze before submission")
    lock = json.loads(LOCK.read_text())
    if lock["parent_protocol_hash"] != plan.protocol_hash():
        raise SystemExit("parent V4 protocol changed after recovery freeze")
    if lock["parent_lock_sha256"] != file_hash(V4 / "protocol" / "protocol_lock.json"):
        raise SystemExit("parent V4 protocol lock changed after recovery freeze")
    if lock["code"] != recovery_code_hashes():
        raise SystemExit("recovery code changed after recovery freeze")
    known = {spec["id"]: spec for spec in original.build_tasks()}
    for record in lock["targets"]:
        spec = known.get(record["id"])
        if spec is None or spec["fn"] != "final" or spec["meta"].get("dataset") != "freebase":
            raise SystemExit(f"invalid recovery target {record['id']!r}")
    return lock


def append_manifest(task_id: str, status: str, **fields) -> None:
    path = RECOVERY / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({
            "task_id": task_id,
            "status": status,
            "time": now_iso(),
            **fields,
        }) + "\n")


@dataclass
class Running:
    spec: dict
    proc: subprocess.Popen
    gpu: str
    attempt: int
    started: float = field(default_factory=time.time)


class Recovery:
    """A one-process-per-GPU recovery pool over the frozen target list."""

    def __init__(self, specs: list[dict], gpus: list[str], deadline: float | None, *,
                 launch=None, poll: float = 5.0):
        self.specs = {spec["id"]: spec for spec in specs}
        self.order = [spec["id"] for spec in specs]
        self.gpus = list(gpus)
        self.deadline = deadline
        self.launch = launch or self._launch
        self.poll = poll
        self.state = {}
        self.attempts = {task_id: 0 for task_id in self.order}
        self.running: dict[str, Running] = {}
        self.stop = False
        for task_id in self.order:
            self.state[task_id] = (
                "done" if read_status(task_id).get("status") == "done" else "pending"
            )

    def remaining(self) -> float:
        return self.deadline - time.time() if self.deadline else 1e12

    def free_gpus(self) -> list[str]:
        used = {run.gpu for run in self.running.values()}
        return [gpu for gpu in self.gpus if gpu not in used]

    def _launch(self, spec: dict, gpu: str):
        logs = RECOVERY / "logs" / "tasks"
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=gpu,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        stdout = (logs / f"{spec['id']}.out").open("a")
        stderr = (logs / f"{spec['id']}.err").open("a")
        proc = subprocess.Popen(
            [sys.executable, "-m", "experiments.lightweight_campaign", "--run-task", spec["id"]],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        stdout.close()
        stderr.close()
        return proc

    def finish(self, task_id: str, code: int) -> None:
        run = self.running.pop(task_id)
        record = read_status(task_id)
        runtime = time.time() - run.started
        if code == 0 and record.get("status") == "done":
            status = "done"
            self.state[task_id] = status
        elif self.attempts[task_id] < MAX_ATTEMPTS and not self.stop:
            status = "retry"
            self.state[task_id] = "pending"
        else:
            status = "failed"
            self.state[task_id] = status
        append_manifest(
            task_id,
            status,
            gpu=run.gpu,
            attempt=run.attempt,
            exit_code=code,
            runtime_s=round(runtime, 1),
            error=record.get("error"),
        )
        log(f"{status}: {task_id} (gpu {run.gpu}, rc={code}, {runtime:.0f}s)")

    def progress(self) -> None:
        counts = {}
        for status in self.state.values():
            counts[status] = counts.get(status, 0) + 1
        write_json(RECOVERY / "progress.json", {
            "time": now_iso(),
            "counts": counts,
            "remaining_s": self.remaining(),
            "running": {
                task_id: {"gpu": run.gpu, "since_s": round(time.time() - run.started)}
                for task_id, run in self.running.items()
            },
        })

    def step(self) -> int:
        for task_id, run in list(self.running.items()):
            code = run.proc.poll()
            if code is not None:
                self.finish(task_id, code)
        launched = 0
        for gpu in self.free_gpus():
            task_id = next(
                (candidate for candidate in self.order if self.state[candidate] == "pending"),
                None,
            )
            if task_id is None:
                break
            spec = self.specs[task_id]
            if spec["est_minutes"] * 60 > self.remaining():
                break
            self.attempts[task_id] += 1
            proc = self.launch(spec, gpu)
            self.running[task_id] = Running(spec, proc, gpu, self.attempts[task_id])
            self.state[task_id] = "running"
            append_manifest(
                task_id,
                "start",
                gpu=gpu,
                attempt=self.attempts[task_id],
                priority=spec["priority"],
            )
            log(f"start: {task_id} (gpu {gpu}, P{spec['priority']})")
            launched += 1
        self.progress()
        return launched

    def stop_all(self, reason: str) -> None:
        log(f"stopping {len(self.running)} worker(s): {reason}")
        for run in self.running.values():
            try:
                os.killpg(run.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        end = time.time() + 60
        while self.running and time.time() < end:
            for task_id, run in list(self.running.items()):
                if run.proc.poll() is not None:
                    self.finish(task_id, run.proc.returncode or 1)
            time.sleep(1)
        for task_id, run in list(self.running.items()):
            run.proc.kill()
            self.finish(task_id, run.proc.wait() or 1)

    def run(self) -> bool:
        log(f"{len(self.order)} targets; GPUs {self.gpus}; exactly one worker per GPU")
        while not self.stop:
            launched = self.step()
            if not self.running and not launched:
                break
            if self.remaining() <= 0:
                self.stop_all("deadline")
                break
            time.sleep(self.poll)
        if self.stop:
            self.stop_all("signal")
        for task_id in self.order:
            if self.state[task_id] == "pending":
                self.state[task_id] = "not_started"
                append_manifest(task_id, "not_started", reason="deadline")
        self.progress()
        return all(status == "done" for status in self.state.values())


def aggregate() -> None:
    logs = RECOVERY / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    with (logs / "aggregate.out").open("a") as stdout, \
            (logs / "aggregate.err").open("a") as stderr:
        result = subprocess.run(
            [sys.executable, "-m", "experiments.lightweight_selection.aggregate"],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=3600,
        )
    if result.returncode:
        raise RuntimeError(f"V4 aggregation failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--margin-minutes", type=float, default=20.0)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.freeze:
        lock = freeze()
        log(f"frozen: {lock['target_count']} targets, {lock['priority_counts']}")
        return
    if args.dry_run:
        targets = select_targets(original.build_tasks())
        for spec in targets:
            print(f"P{spec['priority']} {spec['id']}")
        print(f"{len(targets)} failed Freebase final tasks")
        return
    lock = check_lock()
    known = {spec["id"]: spec for spec in original.build_tasks()}
    specs = [known[record["id"]] for record in lock["targets"]]
    gpus = original.visible_gpus()
    if not gpus:
        raise SystemExit("no GPUs visible: run inside the recovery Slurm allocation")
    left = original.slurm_seconds_left()
    deadline = time.time() + left - args.margin_minutes * 60 if left else None
    recovery = Recovery(specs, gpus, deadline)

    def request_stop(signum, _frame):
        log(f"signal {signum}")
        recovery.stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    complete = recovery.run()
    log("final aggregation")
    aggregate()
    if not complete:
        raise SystemExit("recovery ended with failed or not-started targets")
    log("all recovery targets completed")


if __name__ == "__main__":
    main()
