"""Final experiment suite: independent single-GPU runs on every GPU of one allocation.

There is no multi-GPU training. The parent process never touches CUDA; it launches one
child process per experiment, pins it to one GPU of the Slurm allocation through
CUDA_VISIBLE_DEVICES (so the child sees it as ``cuda:0``), and hands a freed GPU the
next unfinished experiment in priority order. Rerunning the same command skips
completed experiments, resumes those with a compatible checkpoint, and starts the rest.

    python -m experiments.baseline_suite              # inside the 7-GPU allocation
    python -m experiments.baseline_suite --dry-run    # print the plan only
    python -m experiments.baseline_suite --freeze-only  # save the protocol only
    python -m experiments.baseline_suite --aggregate-only
"""

from __future__ import annotations

import argparse
import fnmatch
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
from datetime import datetime, timezone
from pathlib import Path

from .aggregate import aggregate
from .protocol import PROTOCOL, ROOT, git_state, hyperparameters

SUITE = "autophgt_final_v1"
ARTIFACTS = Path("artifacts")
CONFIG_PATH = ARTIFACTS / "final_suite_config.json"
LOCK_PATH = ARTIFACTS / "suite.lock"
CLAIMS = ARTIFACTS / "claims"
THREADS = {"acm": 2, "ogbn-mag": 3}
CPU_BUDGET = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[suite {_now()}] {message}", flush=True)


def experiment_id(dataset: str, mode: str, seed: int, selection: str, k: int) -> str:
    return f"{'acm' if dataset == 'acm' else 'mag'}_{mode}_seed{seed}_{selection}_k{k}"


def inventory() -> list[dict]:
    """All 32 final experiments in scheduling-priority order."""
    specs = []

    def add(priority, group, dataset, mode, seed, selection="discovered", k=5):
        exp_id = experiment_id(dataset, mode, seed, selection, k)
        specs.append({"id": exp_id, "priority": priority, "group": group, "dataset": dataset,
                      "mode": mode, "seed": seed, "path_selection": selection, "k": k,
                      "threads": THREADS[dataset],
                      "result": str(ARTIFACTS / "results" / f"{exp_id}.json"),
                      "checkpoint": str(ARTIFACTS / "checkpoints" / f"{exp_id}.pt"),
                      "log": str(ARTIFACTS / "logs" / f"{exp_id}.log")})

    for seed in range(5):                                   # A. main ACM comparison
        for mode in ("hgt", "tokens", "auto_phgt"):
            add(1, "acm_main", "acm", mode, seed)
    for mode in ("hgt", "auto_phgt"):                      # D. first matched MAG pair
        add(2, "mag_main", "ogbn-mag", mode, 0)
    for seed in range(5):                                   # B. discovery ablation
        add(3, "acm_discovery_ablation", "acm", "auto_phgt", seed, "random")
    for seed in (1, 2):                                     # D. remaining MAG seeds
        for mode in ("hgt", "auto_phgt"):
            add(4, "mag_main", "ogbn-mag", mode, seed)
    for k in (1, 3):                                        # C. k sensitivity
        for seed in range(3):
            add(5, "acm_k_sensitivity", "acm", "auto_phgt", seed, "discovered", k)
    ids = [s["id"] for s in specs]
    paths = [s[key] for s in specs for key in ("result", "checkpoint", "log")]
    assert len(set(ids)) == len(ids), "experiment ids must be unique"
    assert len(set(paths)) == len(paths), "result/checkpoint/log paths must be unique"
    return specs


def frozen_config(specs: list[dict]) -> dict:
    return {
        "suite": SUITE,
        "protocol": PROTOCOL,
        "hyperparameters": {"acm": hyperparameters("acm"),
                            "ogbn-mag": hyperparameters("ogbn-mag")},
        "seeds": {"acm_main": [0, 1, 2, 3, 4], "acm_discovery_ablation": [0, 1, 2, 3, 4],
                  "acm_k_sensitivity": [0, 1, 2], "mag_main": [0, 1, 2]},
        "scheduling": {"cpu_budget": CPU_BUDGET, "threads_per_experiment": THREADS,
                       "order": "priority 1 -> 5, inventory order within a priority"},
        "experiments": [{key: s[key] for key in ("id", "priority", "group", "dataset", "mode",
                                                 "seed", "path_selection", "k", "result",
                                                 "checkpoint")} for s in specs],
    }


def freeze(specs: list[dict], path: Path = CONFIG_PATH) -> dict:
    """Write the protocol on first launch; on later launches require it to be unchanged."""
    current = frozen_config(specs)
    git = git_state()
    launch = {"time": _now(), "host": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git": git}
    if path.exists():
        saved = json.loads(path.read_text())
        changed = [key for key in current if saved.get(key) != current[key]]
        if changed:
            raise SystemExit(f"{path} differs from the current protocol in {changed}; "
                             "refusing to mix protocols. Move the old artifacts aside first.")
        if saved["git"]["commit"] != git["commit"]:
            _log(f"WARNING: frozen at commit {saved['git']['commit']}, now {git['commit']}")
        saved.setdefault("launches", []).append(launch)
        record = saved
    else:
        record = {**current, "frozen_at": launch["time"], "git": git, "launches": [launch]}
    if git["dirty"]:
        _log("WARNING: tracked files differ from the recorded commit")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)
    return record


def _job_running(job_id) -> bool:
    if not job_id:
        return False
    try:
        state = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                               capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        _log(f"WARNING: cannot query Slurm job {job_id}; assuming it is not running")
        return False
    return state in {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}


def acquire_lock(path: Path = LOCK_PATH, job_running=_job_running) -> dict:
    """Refuse to start while another Slurm job is running the suite.

    Two concurrent suites would launch the same experiments and write the same
    checkpoints. A lock left behind by a job that is no longer running is replaced.
    """
    me = os.environ.get("SLURM_JOB_ID")
    if path.exists():
        try:
            holder = json.loads(path.read_text())
        except json.JSONDecodeError:
            holder = {}
        other = holder.get("slurm_job_id")
        if other and other != me and job_running(other):
            raise SystemExit(f"The suite is already running in Slurm job {other} "
                             f"({path}); not starting a second copy.")
    record = {"slurm_job_id": me, "host": socket.gethostname(), "pid": os.getpid(),
              "time": _now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record) + "\n")
    os.replace(tmp, path)
    return record


def select(specs: list[dict], only: str | None) -> list[dict]:
    """Experiments whose id matches one of the comma-separated ids or glob patterns."""
    if not only:
        return specs
    patterns = [p.strip() for p in only.split(",") if p.strip()]
    chosen = [s for s in specs if any(fnmatch.fnmatchcase(s["id"], p) for p in patterns)]
    if not chosen:
        raise SystemExit(f"--only {only!r} matches no experiment")
    return chosen


def subset_label(specs: list[dict], full: list[dict]) -> str | None:
    if len(specs) == len(full):
        return None
    return hashlib.sha1(",".join(s["id"] for s in specs).encode()).hexdigest()[:8]


def claim(spec: dict, job_running=None):
    """Record that this Slurm job runs ``spec``; return another live holder's job id instead.

    Lets a second suite job run a disjoint subset: a worker of one job refuses to start
    an experiment that a running worker of another job already holds.
    """
    job_running = job_running or _job_running
    me = os.environ.get("SLURM_JOB_ID")
    path = CLAIMS / f"{spec['id']}.json"
    if path.exists():
        try:
            holder = json.loads(path.read_text()).get("slurm_job_id")
        except json.JSONDecodeError:
            holder = None
        if holder and holder != me and job_running(holder):
            return holder
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"slurm_job_id": me, "host": socket.gethostname(),
                               "pid": os.getpid(), "time": _now()}) + "\n")
    os.replace(tmp, path)
    return None


def release_lock(record: dict, path: Path = LOCK_PATH) -> None:
    try:
        if json.loads(path.read_text()) == record:
            path.unlink()
    except (OSError, json.JSONDecodeError):
        pass


def is_complete(spec: dict) -> bool:
    path = Path(spec["result"])
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return record.get("status") == "completed" and record.get("experiment_id") == spec["id"]


def status_path(spec: dict) -> Path:
    return ARTIFACTS / "status" / f"{spec['id']}.json"


def update_status(spec: dict, **attempt) -> None:
    path = status_path(spec)
    record = json.loads(path.read_text()) if path.exists() else {"id": spec["id"],
                                                                 "attempts": []}
    if attempt.get("start"):
        record["attempts"].append(attempt)
    else:
        record["attempts"][-1].update(attempt)
    record["status"] = record["attempts"][-1].get("status", "running")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def visible_gpus() -> list[str]:
    """GPU ids exactly as Slurm assigned them; child i gets entry i as its only GPU."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [g.strip() for g in value.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("CUDA_VISIBLE_DEVICES is empty: run inside a GPU allocation")
    return gpus


def check_device_count(gpus: list[str]) -> int:
    """Count devices in a throw-away child so the parent never initialises CUDA."""
    out = subprocess.run([sys.executable, "-c",
                          "import torch; print(torch.cuda.device_count())"],
                         capture_output=True, text=True, check=True, timeout=900)
    count = int(out.stdout.strip().splitlines()[-1])
    if count != len(gpus):
        raise SystemExit(f"torch sees {count} GPUs but CUDA_VISIBLE_DEVICES lists {len(gpus)}")
    return count


def slurm_time_left_s() -> float | None:
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
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def child_env(spec: dict, worker: int, gpu: str) -> dict:
    env = dict(os.environ)
    threads = str(spec["threads"])
    env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               PYTHONUNBUFFERED="1", AUTOPHGT_WORKER=str(worker), AUTOPHGT_GPU=gpu)
    return env


def launch_child(spec: dict, worker: int, gpu: str, module: str = "experiments.baseline_suite"):
    log = Path(spec["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a")
    handle.write(f"\n===== {_now()} start on {socket.gethostname()} worker {worker} "
                 f"GPU {gpu} job {os.environ.get('SLURM_JOB_ID')} =====\n")
    handle.flush()
    proc = subprocess.Popen([sys.executable, "-m", module,
                             "--worker", spec["id"]], cwd=ROOT, env=child_env(spec, worker, gpu),
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    return proc


def _log_tail(path: str, lines: int = 25) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


@dataclass
class Running:
    spec: dict
    proc: object
    worker: int
    gpu: str
    started: float = field(default_factory=time.time)


class Suite:
    def __init__(self, specs, gpus, *, launch=launch_child, cpu_budget=CPU_BUDGET,
                 deadline=None, poll_s=5.0, on_change=None,
                 state_path=ARTIFACTS / "suite_state.json"):
        self.state_path = Path(state_path)
        self.specs = specs
        self.gpus = gpus
        self.launch = launch
        self.cpu_budget = cpu_budget
        self.deadline = deadline
        self.poll_s = poll_s
        self.on_change = on_change or (lambda: None)
        self.running: dict[int, Running] = {}
        self.stop_requested = False
        self.outcomes: dict[str, str] = {}

    def _write_state(self):
        state = {"time": _now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "running": [{"id": r.spec["id"], "worker": w, "gpu": r.gpu,
                              "since_s": round(time.time() - r.started)}
                             for w, r in sorted(self.running.items())],
                 "outcomes_this_launch": self.outcomes}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2) + "\n")

    def _finish(self, worker: int, returncode: int, interrupted: bool = False):
        run = self.running.pop(worker)
        spec = run.spec
        if is_complete(spec):
            outcome = "completed"
        elif interrupted:
            outcome = "interrupted"
        else:
            outcome = "failed"
        self.outcomes[spec["id"]] = outcome
        update_status(spec, status=outcome, end=_now(), returncode=returncode,
                      runtime_s=round(time.time() - run.started, 1),
                      error_tail=None if outcome == "completed" else _log_tail(spec["log"]))
        _log(f"{outcome}: {spec['id']} (worker {worker}, GPU {run.gpu}, rc={returncode}, "
             f"{time.time() - run.started:.0f} s)")
        self._write_state()
        self.on_change()

    def _stop_all(self, reason: str):
        _log(f"stopping {len(self.running)} running experiment(s): {reason}")
        for run in self.running.values():
            try:
                os.killpg(run.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, AttributeError):
                run.proc.terminate()
        end = time.time() + 60
        while self.running and time.time() < end:
            for worker, run in list(self.running.items()):
                code = run.proc.poll()
                if code is not None:
                    self._finish(worker, code, interrupted=True)
            time.sleep(1)
        for worker, run in list(self.running.items()):
            run.proc.kill()
            self._finish(worker, run.proc.wait(), interrupted=True)

    def run(self):
        pending = [s for s in self.specs if not is_complete(s)]
        skipped = len(self.specs) - len(pending)
        _log(f"{len(self.specs)} experiments: {skipped} already complete, {len(pending)} to run "
             f"on {len(self.gpus)} GPU worker(s) {self.gpus}, CPU budget {self.cpu_budget}")
        self._write_state()
        while pending or self.running:
            if self.stop_requested or (self.deadline and time.time() >= self.deadline):
                self._stop_all("stop requested" if self.stop_requested else "deadline reached")
                break
            for worker, run in list(self.running.items()):
                code = run.proc.poll()
                if code is not None:
                    self._finish(worker, code)
            for worker, gpu in enumerate(self.gpus):
                if worker in self.running:
                    continue
                used = sum(r.spec["threads"] for r in self.running.values())
                spec = next((s for s in pending if used + s["threads"] <= self.cpu_budget), None)
                if spec is None:
                    break
                pending.remove(spec)
                resume = bool(spec.get("checkpoint")) and Path(spec["checkpoint"]).exists()
                proc = self.launch(spec, worker, gpu)
                self.running[worker] = Running(spec, proc, worker, gpu)
                update_status(spec, start=_now(), status="running", worker=worker, gpu=gpu,
                              host=socket.gethostname(),
                              slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                              resumed_from_checkpoint=resume)
                _log(f"{'resume' if resume else 'start'}: {spec['id']} on worker {worker} "
                     f"(GPU {gpu}, {spec['threads']} threads); {len(pending)} waiting")
                self._write_state()
            time.sleep(self.poll_s)
        self._write_state()
        return self.outcomes


def run_worker(exp_id: str) -> None:
    """Child entry point: run one experiment on this process's single visible GPU."""
    spec = next(s for s in inventory() if s["id"] == exp_id)
    if is_complete(spec):
        print(f"{exp_id} is already complete; nothing to do", flush=True)
        return
    holder = claim(spec)
    if holder:
        raise SystemExit(f"{exp_id} is running in Slurm job {holder}; not starting a duplicate")
    import torch
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"worker expects exactly one visible GPU, sees "
                         f"{torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES="
                         f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
    extra = {"experiment_id": spec["id"], "suite": SUITE, "group": spec["group"],
             "priority": spec["priority"],
             "assigned_gpu": {"worker": int(os.environ.get("AUTOPHGT_WORKER", -1)),
                              "slurm_gpu_id": os.environ.get("AUTOPHGT_GPU"),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "torch_device": "cuda:0"}}
    kwargs = dict(mode=spec["mode"], seed=spec["seed"], k=spec["k"],
                  path_selection=spec["path_selection"], device="cuda",
                  output=spec["result"], checkpoint=spec["checkpoint"], extra=extra)
    if spec["dataset"] == "acm":
        from .run_acm import run
    else:
        from .run_mag import run
    result = run(**kwargs)
    print(json.dumps({"experiment_id": exp_id, "test": result["test"],
                      "best_epoch": result["best_epoch"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--freeze-only", action="store_true",
                        help="write or verify artifacts/final_suite_config.json and exit")
    parser.add_argument("--only", help="comma-separated experiment ids or glob patterns; run "
                        "just these, e.g. in a second job on a disjoint subset")
    parser.add_argument("--margin-minutes", type=float, default=15.0,
                        help="stop children this long before the Slurm time limit")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        run_worker(args.worker)
        return
    full = inventory()
    specs = select(full, args.only)
    label = subset_label(specs, full)
    if args.aggregate_only:
        aggregate()
        return
    if args.dry_run:
        for spec in specs:
            state = ("complete" if is_complete(spec) else
                     "resume" if Path(spec["checkpoint"]).exists() else "start")
            print(f"P{spec['priority']} {spec['id']:<42} {state}")
        print(f"{len(specs)} experiments")
        return
    if args.freeze_only:
        freeze(full)
        _log(f"protocol frozen in {CONFIG_PATH}")
        return
    lock_path = LOCK_PATH if label is None else ARTIFACTS / f"suite.{label}.lock"
    lock = acquire_lock(lock_path)
    try:
        run_suite(specs, args.margin_minutes, full=full, label=label)
    finally:
        release_lock(lock, lock_path)


def run_suite(specs: list[dict], margin_minutes: float, *, full=None, label=None) -> None:
    freeze(full or specs)
    if label:
        _log(f"running subset {label}: {', '.join(s['id'] for s in specs)}")
    gpus = visible_gpus()
    _log(f"host {socket.gethostname()}, Slurm GPUs {gpus}, torch sees "
         f"{check_device_count(gpus)} device(s)")
    left = slurm_time_left_s()
    deadline = time.time() + left - margin_minutes * 60 if left else None
    if left:
        _log(f"Slurm time left {left / 3600:.2f} h; children stop "
             f"{margin_minutes:.0f} min before the limit")

    def refresh():
        try:
            aggregate()
        except Exception as error:  # a summary problem must never stop the experiments
            _log(f"aggregation failed: {error!r}")

    state_path = ARTIFACTS / ("suite_state.json" if label is None else f"suite_state.{label}.json")
    suite = Suite(specs, gpus, deadline=deadline, on_change=refresh, state_path=state_path)

    def request_stop(signum, _frame):
        _log(f"received signal {signum}")
        suite.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        suite.run()
    finally:
        refresh()
        done = sum(is_complete(s) for s in specs)
        _log(f"finished this launch: {done}/{len(specs)} experiments complete overall")


if __name__ == "__main__":
    main()
