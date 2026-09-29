"""V3 master: one Slurm allocation, a dependency-aware pool of single-GPU worker processes.

    python -m experiments.residual_campaign --freeze       # write protocol lock + V3_PROTOCOL.md
    python -m experiments.residual_campaign --dry-run      # print the task graph
    python -m experiments.residual_campaign                # run the campaign (inside the allocation)

Scheduling: every GPU offers 4 units (small task 1, medium 2, large 4, so at most one large
process per GPU); a global CPU-thread budget; P0 before P1 (P1 starts only once every P0
task has started); a task starts only if its time estimate fits before the deadline; failed
tasks are retried once; dependents of failed tasks are recorded as dep_failed.
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

from experiments.residual_selection import plan
from experiments.residual_selection.plan import (FEATURE_REGIMES, FREEBASE_EXTENSION, HGB_DATASETS,
                                 MAG_CONDITIONS, SEEDS, V3, hgb_conditions, search_lmax)

UNITS = {"small": 1, "medium": 2, "large": 4, "cpu": 0}
THREADS = {"small": 2, "medium": 3, "large": 3, "cpu": 1}
GPU_UNITS = 4
MAG_CONCURRENCY = 8


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[v3 {now_iso()}] {msg}", flush=True)


# --------------------------------------------------------------------------- task graph
def task(tid, fn, args, *, deps=(), size="small", priority=0, est=5.0, gate=None, meta=None):
    return {"id": tid, "fn": fn, "args": args, "deps": list(deps), "size": size,
            "priority": priority, "est_minutes": est, "gate": gate, "meta": meta or {}}


def hgb_size(ds: str) -> str:
    return "medium" if ds == "freebase" else "small"


def build_tasks() -> list[dict]:
    tasks = [task("provenance", "provenance", {}, size="small", est=5)]
    est_tune = {"acm": 12, "dblp": 20, "freebase": 150}
    est_search = {"acm": 10, "dblp": 10, "freebase": 30}
    est_final = {"acm": 3, "dblp": 4, "freebase": 10}
    # MAG first: the longest chains. (V3_SMOKE_NO_MAG=1 is for CPU-only smoke tests only.)
    with_mag = os.environ.get("V3_SMOKE_NO_MAG") != "1"
    if with_mag:
        tasks.append(task("mag__search", "mag_search", {"seed": 0}, size="medium", est=270,
                          meta={"dataset": "ogbn-mag", "method": "rcms_search"}))
    for cond in (MAG_CONDITIONS if with_mag else []):
        for seed in SEEDS["ogbn-mag"]:
            deps = ["mag__search"] if cond["selection"] == "rcms" else []
            tid = f"{cond['id']}__seed{seed}"
            tasks.append(task(tid, "mag_final", {"name": cond["name"], "seed": seed},
                              deps=deps, size="medium", est=480,
                              meta={"dataset": "ogbn-mag", "method": cond["name"], "seed": seed}))
            tasks.append(task(f"{tid}__continue", "mag_continue",
                              {"name": cond["name"], "seed": seed}, deps=[tid], size="medium",
                              priority=1, est=10,
                              meta={"dataset": "ogbn-mag", "method": cond["name"], "seed": seed}))
    for ds in HGB_DATASETS:
        tune_ids = []
        for feat in FEATURE_REGIMES[ds]:
            for layers in plan.GRID["layers"]:
                tid = f"{ds}__tune__{feat}__L{layers}"
                tune_ids.append(tid)
                tasks.append(task(tid, "tune", {"ds": ds, "feat": feat, "layers": layers},
                                  size=hgb_size(ds), est=est_tune[ds],
                                  meta={"dataset": ds, "method": "hgt_tuning"}))
        tasks.append(task(f"{ds}__select_config", "select_config", {"ds": ds}, deps=tune_ids,
                          size="cpu", est=1, meta={"dataset": ds}))
        if ds in ("acm", "dblp"):
            tasks.append(task(f"{ds}__param_match", "param_match", {"ds": ds},
                              deps=[f"{ds}__select_config"], size="cpu", est=3,
                              meta={"dataset": ds}))
    for ds in HGB_DATASETS:
        seeds = SEEDS[ds] + (FREEBASE_EXTENSION["seeds"] if ds == "freebase" else [])
        for seed in seeds:
            extended = ds == "freebase" and seed in FREEBASE_EXTENSION["seeds"]
            gate = "freebase_extension" if extended else None
            ext_deps = ["freebase__decision"] if extended else []
            for lmax in search_lmax(ds):
                priority = 0 if (lmax == 4 or ds == "dblp") else 1
                tasks.append(task(f"{ds}__search__L{lmax}__seed{seed}", "search",
                                  {"ds": ds, "seed": seed, "lmax": lmax},
                                  deps=[f"{ds}__select_config", *ext_deps], size=hgb_size(ds),
                                  priority=priority, est=est_search[ds] * (2 if lmax == 6 else 1),
                                  gate=gate, meta={"dataset": ds, "method": "rcms_search",
                                                   "seed": seed}))
            for cond in hgb_conditions():
                if cond["dataset"] != ds:
                    continue
                deps = [f"{ds}__select_config", *ext_deps]
                if cond["width"] == "pm":
                    deps.append(f"{ds}__param_match")
                if cond["selector"] and cond["selector"].startswith("rcms"):
                    deps.append(f"{ds}__search__L{cond['lmax']}__seed{seed}")
                tasks.append(task(f"{cond['id']}__seed{seed}", "final",
                                  {"cond_id": cond["id"], "seed": seed}, deps=deps,
                                  size=hgb_size(ds), priority=cond["priority"],
                                  est=est_final[ds] * (4 if cond["selector"] == "lmsps" else 1),
                                  gate=gate, meta={"dataset": ds, "method": cond["name"],
                                                   "seed": seed}))
    fb_core = [f"{c['id']}__seed{s}" for c in hgb_conditions()
               if c["dataset"] == "freebase" and c["priority"] == 0 for s in SEEDS["freebase"]]
    tasks.append(task("freebase__decision", "freebase_decision", {}, deps=fb_core, size="cpu",
                      est=1, meta={"dataset": "freebase"}))
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    known = set(ids)
    for t in tasks:
        missing = [d for d in t["deps"] if d not in known]
        assert not missing, (t["id"], missing)
    return tasks


def config_hash(t: dict) -> str:
    blob = json.dumps({"fn": t["fn"], "args": t["args"], "protocol": plan.protocol_hash()},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# --------------------------------------------------------------------------- worker side
def status_path(tid: str) -> Path:
    return V3 / "status" / f"{tid}.json"


def write_status(tid: str, record: dict) -> None:
    from auto_phgt.runtime import write_json
    write_json(status_path(tid), record)


def gate_open(gate) -> bool:
    if gate is None:
        return True
    if gate == "freebase_extension":
        from experiments.residual_selection.tasks import freebase_extension_enabled
        return freebase_extension_enabled()
    raise ValueError(gate)


def run_task(tid: str) -> int:
    spec = next(t for t in build_tasks() if t["id"] == tid)
    start = time.time()
    if not gate_open(spec["gate"]):
        write_status(tid, {"id": tid, "status": "skipped", "reason": f"gate {spec['gate']} closed",
                           "time": now_iso()})
        return 0
    from experiments.residual_selection import tasks as impl
    impl.setup_threads()
    fn = getattr(impl, spec["fn"])
    args = dict(spec["args"])
    if spec["fn"] == "mag_continue":
        args["seconds_left"] = float(os.environ.get("V3_SECONDS_LEFT", "0"))
    try:
        result = fn(**args)
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


class Master:
    def __init__(self, tasks, gpus, cpu_budget, deadline, *, launch=None, poll=5.0):
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
                 "method": spec["meta"].get("method", spec["fn"]), "seed": spec["meta"].get("seed"),
                 "config_hash": config_hash(spec), "status": status, "priority": spec["priority"],
                 "time": now_iso(), **fields}
        path = V3 / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _launch(self, spec, gpu):
        logs = V3 / "logs" / "tasks"
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        threads = str(THREADS[spec["size"]])
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                   PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=gpu if gpu is not None else "",
                   V3_SECONDS_LEFT=str(max(0.0, self.deadline - time.time())
                                       if self.deadline else 1e9))
        out = (logs / f"{spec['id']}.out").open("a")
        err = (logs / f"{spec['id']}.err").open("a")
        proc = subprocess.Popen([sys.executable, "-m", "experiments.residual_campaign", "--run-task",
                                 spec["id"]], cwd=plan.ROOT, env=env, stdout=out, stderr=err,
                                start_new_session=True)
        out.close()
        err.close()
        return proc

    def deps_state(self, tid):
        states = [self.state[d] for d in self.tasks[tid]["deps"]]
        if any(s in ("failed", "dep_failed") for s in states):
            return "failed"
        if all(s in ("done", "skipped") for s in states):
            return "ready"
        return "waiting"

    def threads_used(self):
        return sum(THREADS[r.spec["size"]] for r in self.running.values())

    def mag_running(self):
        return sum(1 for r in self.running.values() if r.spec["fn"].startswith("mag_"))

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
                      output_dir=str(V3 / "logs" / "tasks"))
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
        p0_open = any(self.state[t] == "pending" and self.tasks[t]["priority"] == 0
                      for t in self.order)
        launched = 0
        for tid in self.order:
            spec = self.tasks[tid]
            if self.state[tid] != "pending" or self.deps_state(tid) != "ready":
                continue
            if spec["gate"] is not None and not gate_open(spec["gate"]):
                self.state[tid] = "skipped"
                write_status(tid, {"id": tid, "status": "skipped",
                                   "reason": f"gate {spec['gate']} closed", "time": now_iso()})
                self.manifest(tid, "skipped", reason=f"gate {spec['gate']} closed")
                launched += 1  # progress: dependents and P1 tasks may now become runnable
                continue
            if spec["priority"] > 0 and p0_open:
                continue
            if spec["est_minutes"] * 60 > self.remaining():
                continue
            if self.threads_used() + THREADS[spec["size"]] > self.cpu_budget:
                continue
            if spec["fn"].startswith("mag_") and self.mag_running() >= MAG_CONCURRENCY:
                continue
            gpu, ok = self.pick_gpu(spec["size"])
            if not ok:
                continue
            if gpu is not None:
                self.free_units[gpu] -= UNITS[spec["size"]]
            self.attempts[tid] += 1
            proc = self.launch(spec, gpu)
            self.running[tid] = Running(spec, proc, gpu, self.attempts[tid])
            self.state[tid] = "running"
            launched += 1
            self.manifest(tid, "start", gpu=gpu, attempt=self.attempts[tid], start_time=now_iso())
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
        (V3 / "progress.json").write_text(json.dumps(record, indent=2) + "\n")

    def open_tasks(self):
        return [t for t in self.order if self.state[t] in ("pending", "running")]

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
                spawn_aggregate()
            time.sleep(self.poll)
        if self.stop:
            self.stop_all("stop requested")
        for tid in self.order:
            if self.state[tid] == "pending":
                self.manifest(tid, "not_started", reason="deadline or unmet dependencies")
        self.progress()


def spawn_aggregate(wait: bool = False):
    logs = V3 / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    proc = subprocess.Popen([sys.executable, "-m", "experiments.residual_selection.aggregate"], cwd=plan.ROOT,
                            env=env, stdout=(logs / "aggregate.out").open("a"),
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
    files = sorted(Path(plan.ROOT, "experiments", "residual_selection").glob("*.py")) + [
        Path(plan.ROOT, "experiments", "residual_campaign.py")]
    return {str(p.relative_to(plan.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in files}


def freeze() -> dict:
    from auto_phgt.runtime import git_state, write_json
    from experiments.residual_selection.protocol_doc import write_protocol_doc
    tasks = build_tasks()
    diff = subprocess.run(["git", "-C", str(plan.ROOT), "diff", "HEAD"], capture_output=True,
                          text=True).stdout
    record = {"protocol_hash": plan.protocol_hash(), "git": git_state(),
              "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16],
              "code": code_hashes(), "tasks": len(tasks),
              "task_graph_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()
                                                  ).hexdigest()[:16],
              "protocol": plan.protocol_record(), "frozen_at": now_iso(),
              "host": socket.gethostname()}
    write_json(V3 / "protocol" / "protocol_lock.json", record)
    (V3 / "protocol" / "git_diff.patch").write_text(diff)
    write_protocol_doc(record, tasks)
    from experiments.residual_selection import aggregate, published
    aggregate._csv(V3 / "baselines" / "published_hgb.csv",
                   [r for r in published.rows() if r["source"] == published.HGB])
    aggregate._csv(V3 / "baselines" / "published_external.csv", published.rows())
    return record


def check_lock() -> None:
    path = V3 / "protocol" / "protocol_lock.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run --freeze before launching")
    lock = json.loads(path.read_text())
    if lock["protocol_hash"] != plan.protocol_hash():
        raise SystemExit("protocol changed after freezing; refusing to run")
    if lock["code"] != code_hashes():
        raise SystemExit("V3 code changed after freezing; refusing to run")


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
            key = (t["meta"].get("dataset", "-"), t["fn"], t["priority"])
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
