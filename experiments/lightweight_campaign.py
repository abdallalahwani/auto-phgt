"""V4-Hedge master: one Slurm allocation, a dependency-aware pool of worker processes.

    python -m experiments.lightweight_campaign --freeze     # protocol lock + V4_HEDGE_PROTOCOL.md
    python -m experiments.lightweight_campaign --dry-run    # print the task graph
    python -m experiments.lightweight_campaign              # run the campaign (inside the allocation)

Scheduling: CPU tasks (selectors, decisions) run without a GPU; GPU tasks take units of a GPU
(ACM 1, Freebase 2, of 4 per GPU); a global CPU-thread budget; tasks are considered in priority
order and a lower tier never takes resources while a ready higher-tier task waits for them (so
P1 work fills resources that no ready P0 task can use); P2 starts only once no P0/P1 task is
pending; ``after`` orders tasks without requiring success (the ACM diagnostic runs before any
model training); a task starts only if its estimate fits before the deadline; one retry;
dependents of failed tasks are recorded as dep_failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from experiments.lightweight_selection import plan

UNITS = {"small": 1, "medium": 2, "cpu": 0}
THREADS = {"small": 1, "medium": 1, "cpu": 2}
GPU_UNITS = 4
V4 = plan.V4
DEPENDENCY_FILES = [
    "experiments/residual_selection/data.py", "experiments/residual_selection/train.py", "experiments/residual_selection/plan.py",
    "experiments/common.py", "experiments/run_acm.py", "experiments/protocol.py",
    "auto_phgt/discovery.py", "auto_phgt/selection.py", "auto_phgt/tokenization.py",
    "auto_phgt/model.py", "auto_phgt/training.py", "auto_phgt/evaluation.py",
    "auto_phgt/runtime.py", "auto_phgt/data.py"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[v4 {now_iso()}] {msg}", flush=True)


# --------------------------------------------------------------------------- task graph
def task(tid, fn, args, *, deps=(), after=(), size="cpu", priority=0, est=5.0, gate=None,
         meta=None):
    return {"id": tid, "fn": fn, "args": args, "deps": list(deps), "after": list(after),
            "size": size, "priority": priority, "est_minutes": est, "gate": gate,
            "meta": meta or {}}


def gpu_size(ds: str) -> str:
    return "medium" if ds == "freebase" else "small"


def selector_deps(ds: str, selector: str | None) -> list[str]:
    if selector in ("ti", "ti_set", "ti_raw", "ti_cov", "ti_norm"):
        return [f"{ds}__ti", "acm__ti_decision"]
    if selector in ("fastpath", "fastpath_set"):
        return [f"{ds}__fastpath"]
    if selector == "ti_beam":
        return [f"{ds}__ti_beam"]
    return []


def build_tasks() -> list[dict]:
    est_final = {"acm": 4, "freebase": 15}
    est_tune = {"acm": 8, "freebase": 60}
    tasks = [task("provenance", "provenance", {}, est=5)]
    for ds in plan.DATASETS:
        tasks.append(task(f"{ds}__ti", "ti", {"ds": ds}, est=20 if ds == "freebase" else 5,
                          meta={"dataset": ds, "method": "transition_info"}))
        tasks.append(task(f"{ds}__fastpath", "fastpath", {"ds": ds}, est=20,
                          meta={"dataset": ds, "method": "fastpath"}))
    tasks.append(task("acm__diagnostic", "diagnostic", {}, deps=["acm__ti", "acm__fastpath"],
                      est=5, meta={"dataset": "acm", "method": "diagnostic"}))
    tasks.append(task("acm__ti_decision", "ti_decision", {}, deps=["acm__ti"],
                      after=["acm__diagnostic"], est=1, meta={"dataset": "acm"}))
    for ds in plan.DATASETS:
        tasks.append(task(f"{ds}__ti_beam", "ti_beam", {"ds": ds}, deps=["acm__ti_decision"],
                          priority=2, est=60, meta={"dataset": ds, "method": "ti_beam"}))
    for ds in plan.DATASETS:
        tune_ids = []
        for layers in plan.GRID[ds]["layers"]:
            for lr in plan.GRID[ds]["lr"]:
                tid = f"{ds}__tune__L{layers}__lr{lr:g}"
                tune_ids.append(tid)
                tasks.append(task(tid, "tune", {"ds": ds, "layers": layers, "lr": lr},
                                  after=["acm__diagnostic"], size=gpu_size(ds),
                                  est=est_tune[ds], meta={"dataset": ds, "method": "hgt_tuning"}))
        tasks.append(task(f"{ds}__select_config", "select_config", {"ds": ds}, deps=tune_ids,
                          est=1, meta={"dataset": ds}))
    ext = plan.FREEBASE_EXTENSION
    extendable = set(ext["v4_pool"]) | set(ext["comparators"])
    for cond in plan.conditions():
        ds = cond["dataset"]
        seeds = [(s, False) for s in plan.SEEDS[ds]]
        if ds == "freebase" and cond["name"] in extendable:
            seeds += [(s, True) for s in ext["seeds"]]
        for seed, extended in seeds:
            deps = [f"{ds}__select_config", *selector_deps(ds, cond["selector"])]
            gate = cond["gate"]
            if extended:
                deps.append("freebase__decision")
                gate = "freebase_extension"
            if gate == "not_primary_variant":
                deps.append("acm__ti_decision")
            tasks.append(task(f"{cond['id']}__seed{seed}", "final",
                              {"cond_id": cond["id"], "seed": seed}, deps=deps,
                              size=gpu_size(ds), priority=1 if extended else cond["priority"],
                              est=est_final[ds] * (2 if cond["lmax"] > plan.LMAX else 1),
                              gate=gate, meta={"dataset": ds, "method": cond["name"],
                                               "seed": seed}))
    for cond in plan.bridge_conditions():
        tasks.append(task(cond["id"], "bridge", {"cond_id": cond["id"]},
                          deps=selector_deps("acm", cond["selector"]), size="small",
                          priority=cond["priority"], est=15,
                          meta={"dataset": "acm", "method": cond["name"]}))
    fb_core = [f"{c['id']}__seed{s}" for c in plan.conditions()
               if c["dataset"] == "freebase" and c["priority"] == 0
               for s in plan.SEEDS["freebase"]]
    tasks.append(task("freebase__decision", "freebase_decision", {}, after=fb_core,
                      priority=1, est=1, meta={"dataset": "freebase"}))
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    known = set(ids)
    for t in tasks:
        missing = [d for d in t["deps"] + t["after"] if d not in known]
        assert not missing, (t["id"], missing)
    return tasks


def config_hash(t: dict) -> str:
    blob = json.dumps({"fn": t["fn"], "args": t["args"], "protocol": plan.protocol_hash()},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# --------------------------------------------------------------------------- worker side
def status_path(tid: str) -> Path:
    return V4 / "status" / f"{tid}.json"


def write_status(tid: str, record: dict) -> None:
    from auto_phgt.runtime import write_json
    write_json(status_path(tid), record)


def gate_open(gate, args) -> bool:
    if gate is None:
        return True
    from experiments.lightweight_selection.tasks import gate_open as impl_gate
    return impl_gate(gate, args)


def run_task(tid: str) -> int:
    spec = next(t for t in build_tasks() if t["id"] == tid)
    start = time.time()
    if not gate_open(spec["gate"], spec["args"]):
        write_status(tid, {"id": tid, "status": "skipped",
                           "reason": f"gate {spec['gate']} closed", "time": now_iso()})
        return 0
    from experiments.lightweight_selection import tasks as impl
    impl.setup_threads()
    fn = getattr(impl, spec["fn"])
    try:
        result = fn(**spec["args"])
    except Exception as error:  # recorded, then re-raised for a non-zero exit
        write_status(tid, {"id": tid, "status": "failed", "error": repr(error),
                           "runtime_s": time.time() - start, "time": now_iso()})
        raise
    write_status(tid, {"id": tid, "status": "done", "result": result,
                       "runtime_s": time.time() - start, "time": now_iso()})
    return 0


# --------------------------------------------------------------------------- master side
@dataclass
class Running:
    spec: dict
    proc: subprocess.Popen
    gpu: str | None
    attempt: int
    started: float = field(default_factory=time.time)


TERMINAL = ("done", "skipped", "failed", "dep_failed")


class Master:
    def __init__(self, tasks, gpus, cpu_budget, deadline, *, launch=None, poll=5.0,
                 aggregate=None):
        self.tasks = {t["id"]: t for t in tasks}
        self.order = [t["id"] for t in tasks]
        self.gpus = gpus
        self.free_units = {g: GPU_UNITS for g in gpus}
        self.cpu_budget = cpu_budget
        self.deadline = deadline
        self.poll = poll
        self.state = {tid: "pending" for tid in self.order}
        self.attempts = {tid: 0 for tid in self.order}
        self.running: dict[str, Running] = {}
        self.launch = launch or self._launch
        self.aggregate = aggregate or spawn_aggregate
        self.stop = False
        self.last_aggregate = time.time()
        for tid in self.order:  # resume: completed status files are honoured
            path = status_path(tid)
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("status") in ("done", "skipped"):
                    self.state[tid] = record["status"]

    def manifest(self, tid, status, **fields):
        spec = self.tasks[tid]
        entry = {"task_id": tid, "dataset": spec["meta"].get("dataset"),
                 "method": spec["meta"].get("method", spec["fn"]),
                 "seed": spec["meta"].get("seed"), "config_hash": config_hash(spec),
                 "status": status, "priority": spec["priority"], "time": now_iso(), **fields}
        path = V4 / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _launch(self, spec, gpu):
        logs = V4 / "logs" / "tasks"
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        threads = str(THREADS[spec["size"]])
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, PYTHONUNBUFFERED="1",
                   CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=gpu if gpu is not None else "")
        out = (logs / f"{spec['id']}.out").open("a")
        err = (logs / f"{spec['id']}.err").open("a")
        proc = subprocess.Popen([sys.executable, "-m", "experiments.lightweight_campaign",
                                 "--run-task", spec["id"]], cwd=plan.ROOT, env=env, stdout=out,
                                stderr=err, start_new_session=True)
        out.close()
        err.close()
        return proc

    def deps_state(self, tid):
        spec = self.tasks[tid]
        states = [self.state[d] for d in spec["deps"]]
        if any(s in ("failed", "dep_failed") for s in states):
            return "failed"
        if not all(s in ("done", "skipped") for s in states):
            return "waiting"
        if not all(self.state[a] in TERMINAL for a in spec["after"]):
            return "waiting"
        return "ready"

    def threads_used(self):
        return sum(THREADS[r.spec["size"]] for r in self.running.values())

    def pick_gpu(self, size):
        need = UNITS[size]
        if need == 0:
            return None, True
        options = [g for g in self.gpus if self.free_units[g] >= need]
        if not options:
            return None, False
        return max(options, key=lambda g: (self.free_units[g], -self.gpus.index(g))), True

    def finish(self, tid, code):
        run = self.running.pop(tid)
        if run.gpu is not None:
            self.free_units[run.gpu] += UNITS[run.spec["size"]]
        record = json.loads(status_path(tid).read_text()) if status_path(tid).exists() else {}
        status = record.get("status") if code == 0 else "failed"
        runtime = time.time() - run.started
        if status in ("done", "skipped"):
            self.state[tid] = status
        elif self.attempts[tid] < 2 and not self.stop:
            self.state[tid] = "pending"  # one retry
            status = "retry"
        else:
            self.state[tid] = "failed"
        self.manifest(tid, status, gpu=run.gpu, attempt=run.attempt, exit_code=code,
                      runtime_s=round(runtime, 1), end_time=now_iso(),
                      output_dir=str(V4 / "logs" / "tasks"))
        log(f"{status}: {tid} (gpu {run.gpu}, rc={code}, {runtime:.0f}s)")

    def remaining(self):
        return (self.deadline - time.time()) if self.deadline else 1e12

    def step(self):
        for tid, run in list(self.running.items()):
            code = run.proc.poll()
            if code is not None:
                self.finish(tid, code)
        for tid in self.order:
            if self.state[tid] == "pending" and self.deps_state(tid) == "failed":
                self.state[tid] = "dep_failed"
                self.manifest(tid, "dep_failed")
        pending = {p: any(self.state[t] == "pending" and self.tasks[t]["priority"] == p
                          for t in self.order) for p in (0, 1)}
        waiting = set()  # tiers with a ready task that could not get resources this step
        launched = 0
        for tid in sorted(self.order, key=lambda t: self.tasks[t]["priority"]):
            spec = self.tasks[tid]
            if self.state[tid] != "pending" or self.deps_state(tid) != "ready":
                continue
            if spec["gate"] is not None and not gate_open(spec["gate"], spec["args"]):
                self.state[tid] = "skipped"
                write_status(tid, {"id": tid, "status": "skipped",
                                   "reason": f"gate {spec['gate']} closed", "time": now_iso()})
                self.manifest(tid, "skipped", reason=f"gate {spec['gate']} closed")
                launched += 1  # progress: dependents may now become runnable
                continue
            priority = spec["priority"]
            if priority >= 2 and (pending[0] or pending[1]):
                continue  # P2 only once no P0/P1 task is pending
            if any(q in waiting for q in range(priority)):
                continue  # never take resources from a ready higher-priority task
            if spec["est_minutes"] * 60 > self.remaining():
                continue
            gpu, ok = self.pick_gpu(spec["size"])
            if not ok or self.threads_used() + THREADS[spec["size"]] > self.cpu_budget:
                waiting.add(priority)
                continue
            if gpu is not None:
                self.free_units[gpu] -= UNITS[spec["size"]]
            self.attempts[tid] += 1
            proc = self.launch(spec, gpu)
            self.running[tid] = Running(spec, proc, gpu, self.attempts[tid])
            self.state[tid] = "running"
            launched += 1
            self.manifest(tid, "start", gpu=gpu, attempt=self.attempts[tid],
                          start_time=now_iso())
            log(f"start: {tid} (gpu {gpu}, {spec['size']}, P{spec['priority']})")
        return launched

    def progress(self):
        counts = {}
        for tid, s in self.state.items():
            key = f"P{self.tasks[tid]['priority']}_{s}"
            counts[key] = counts.get(key, 0) + 1
        record = {"time": now_iso(), "counts": counts, "remaining_s": self.remaining(),
                  "running": {tid: {"gpu": r.gpu, "since_s": round(time.time() - r.started)}
                              for tid, r in self.running.items()}}
        V4.mkdir(parents=True, exist_ok=True)
        (V4 / "progress.json").write_text(json.dumps(record, indent=2) + "\n")

    def stop_all(self, reason):
        log(f"stopping {len(self.running)} running task(s): {reason}")
        for run in self.running.values():
            try:
                os.killpg(run.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        end = time.time() + 60
        while self.running and time.time() < end:
            for tid, run in list(self.running.items()):
                if run.proc.poll() is not None:
                    self.finish(tid, run.proc.returncode or 1)
            time.sleep(1)
        for tid, run in list(self.running.items()):
            run.proc.kill()
            self.finish(tid, run.proc.wait() or 1)

    def run(self, aggregate_every=1800.0):
        log(f"{len(self.order)} tasks; GPUs {self.gpus}; CPU budget {self.cpu_budget}; "
            f"remaining {self.remaining() / 3600:.2f} h")
        while not self.stop:
            launched = self.step()
            self.progress()
            if not self.running and not launched:
                break  # nothing can run any more (done, failed, gated, or no time left)
            if self.remaining() < 0:
                self.stop_all("deadline")
                break
            if time.time() - self.last_aggregate > aggregate_every:
                self.last_aggregate = time.time()
                self.aggregate()
            time.sleep(self.poll)
        if self.stop:
            self.stop_all("stop requested")
        for tid in self.order:
            if self.state[tid] == "pending":
                self.manifest(tid, "not_started", reason="deadline or unmet dependencies")
        self.progress()


def spawn_aggregate(wait: bool = False):
    logs = V4 / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    proc = subprocess.Popen([sys.executable, "-m", "experiments.lightweight_selection.aggregate"],
                            cwd=plan.ROOT, env=env, stdout=(logs / "aggregate.out").open("a"),
                            stderr=(logs / "aggregate.err").open("a"))
    if wait:
        proc.wait(timeout=3600)


def slurm_seconds_left() -> float | None:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        text = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", text)
    if not match:
        return None
    d, h, m, s = (int(x) if x else 0 for x in match.groups())
    return ((d * 24 + h) * 60 + m) * 60 + s


# --------------------------------------------------------------------------- freeze
def code_hashes() -> dict:
    files = sorted(Path(plan.ROOT, "experiments", "lightweight_selection").glob("*.py")) + [
        Path(plan.ROOT, "experiments", "lightweight_campaign.py")]
    files += [Path(plan.ROOT, f) for f in DEPENDENCY_FILES]
    return {str(p.relative_to(plan.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in files}


V3_LAUNCH_COMMIT = "caa687e69a87339a72dfe55221adaec956b181c2"
SHARED_CODE = ["experiments/residual_selection", "experiments/residual_campaign.py",
               "auto_phgt", "experiments/common.py", "experiments/run_acm.py",
               "experiments/protocol.py"]


def v3_code_unchanged() -> dict:
    """V3's code and everything it imports must equal the V3 launch commit (read via git;
    no artifacts/v3 file is opened)."""
    proc = subprocess.run(["git", "-C", str(plan.ROOT), "diff", "--name-only", V3_LAUNCH_COMMIT,
                           "--", *SHARED_CODE], capture_output=True, text=True)
    changed = [line for line in proc.stdout.splitlines() if line.strip()]
    untracked = subprocess.run(["git", "-C", str(plan.ROOT), "ls-files", "--others",
                                "--exclude-standard", "--", *SHARED_CODE],
                               capture_output=True, text=True).stdout.split()
    return {"against": V3_LAUNCH_COMMIT, "changed": changed, "untracked": untracked,
            "unchanged": proc.returncode == 0 and not changed and not untracked}


def freeze() -> dict:
    from auto_phgt.runtime import git_state, write_json
    from experiments.residual_selection import data
    from experiments.lightweight_selection import tasks as impl
    from experiments.lightweight_selection.protocol_doc import write_protocol_doc
    tasks = build_tasks()
    diff = subprocess.run(["git", "-C", str(plan.ROOT), "diff", "HEAD"], capture_output=True,
                          text=True).stdout
    status = subprocess.run(["git", "-C", str(plan.ROOT), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    datasets = {ds: impl.dataset_checks(ds, data.load(ds)) for ds in plan.DATASETS}
    record = {"protocol_hash": plan.protocol_hash(), "git": git_state(),
              "git_status_porcelain": status,
              "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16],
              "code": code_hashes(), "v3_code_check": v3_code_unchanged(),
              "versions": impl.versions(), "tasks": len(tasks),
              "task_graph_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()
                                                  ).hexdigest()[:16],
              "datasets": {ds: {k: v for k, v in rec.items() if k != "edges"}
                           for ds, rec in datasets.items()},
              "protocol": plan.protocol_record(), "frozen_at": now_iso(),
              "host": socket.gethostname()}
    write_json(V4 / "protocol" / "protocol_lock.json", record)
    write_json(V4 / "protocol" / "datasets.json", datasets)
    (V4 / "protocol" / "git_diff.patch").write_text(diff)
    write_protocol_doc(record, tasks)
    (V4 / "checkpoints").mkdir(parents=True, exist_ok=True)
    (V4 / "checkpoints" / "README.md").write_text(
        "V4 runs take minutes, so they keep no per-epoch checkpoints; the master resumes at "
        "task granularity from status/*.json (completed runs are never repeated).\n")
    return record


def check_lock() -> None:
    path = V4 / "protocol" / "protocol_lock.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run --freeze before launching")
    lock = json.loads(path.read_text())
    if lock["protocol_hash"] != plan.protocol_hash():
        raise SystemExit("protocol changed after freezing; refusing to run")
    if lock["code"] != code_hashes():
        raise SystemExit("V4 code (or a dependency) changed after freezing; refusing to run")


def visible_gpus() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [g.strip() for g in value.split(",") if g.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-task")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--margin-minutes", type=float, default=20.0)
    args = parser.parse_args()
    os.chdir(plan.ROOT)
    if args.run_task:
        sys.exit(run_task(args.run_task))
    if args.freeze:
        record = freeze()
        log(f"frozen: protocol {record['protocol_hash']}, {record['tasks']} tasks")
        return
    tasks = build_tasks()
    if args.dry_run:
        counts = {}
        for t in tasks:
            key = (t["meta"].get("dataset", "-"), t["fn"], f"P{t['priority']}", t["size"])
            counts[key] = counts.get(key, 0) + 1
        for key in sorted(counts, key=str):
            print(key, counts[key])
        print(len(tasks), "tasks")
        return
    check_lock()
    gpus = visible_gpus()
    if not gpus:
        raise SystemExit("no GPUs visible: run inside the Slurm allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    left = slurm_seconds_left()
    deadline = time.time() + left - args.margin_minutes * 60 if left else None
    master = Master(tasks, gpus, cpus - 1, deadline)

    def request_stop(signum, _frame):
        log(f"signal {signum}")
        master.stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        master.run()
    finally:
        log("final aggregation")
        spawn_aggregate(wait=True)
        log("done")


if __name__ == "__main__":
    main()
