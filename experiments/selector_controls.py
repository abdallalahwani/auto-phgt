"""V2 follow-up suite, motivated by the v1 failure analysis and frozen before it runs.

V1 (``experiments/baseline_suite.py``, ``artifacts/``) stays unchanged. V2 has two parts
that run as separate Slurm jobs, each an engine instance of the v1 scheduler with its own
directory ``artifacts/v2/<part>/``:

* ``mag``  canonical runs continued to a 300-epoch cap, random paths, dummy-token control;
* ``acm``  path-selector comparison, controls, 100 random path sets, single-path utility.

    python -m experiments.selector_controls --freeze-only
    python -m experiments.selector_controls --part mag --dry-run
    python -m experiments.selector_controls --part acm --workers-per-gpu 2 --cpu-budget 16
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

from . import baseline_suite as engine
from .protocol import ROOT, git_state, hyperparameters

SUITE = "autophgt_v2"
V2 = Path("artifacts/v2")
CONFIG_PATH = V2 / "v2_config.json"
V1_CHECKPOINTS = Path("artifacts/checkpoints")
MAG_EPOCHS = 300
ACM_CANDIDATES = 88
RANDOM_SETS = 100
RANDOM_SET_SEED0 = 1000
THREADS = {"acm": 2, "ogbn-mag": 3}
MANUAL_ACM = [["paper", "to", "author", "to", "paper"],
              ["paper", "to", "subject", "to", "paper"]]
ACM_SELECTORS = ("discovered", "random", "diverse", "homophily", "hybrid")

PROTOCOL_V2 = {
    "purpose": ("Follow-up experiments motivated by the v1 failure analysis. V1 results and "
                "code paths are frozen and not overwritten; every v2 run is new or a copy."),
    "primary_hypotheses": {
        "H1": ("ACM: the canonical structural top-5 ranks below the median of 100 random valid "
               "5-path sets (mean test accuracy over seeds 0-2; also reported on validation)."),
        "H2": "ACM: the label-free diversity selector beats the canonical selector (k=5).",
        "H3": ("ACM: the training-label homophily selector beats the canonical selector and is "
               "at least as good as random selection (k=5)."),
        "H4": ("ACM: Auto-PHGT with canonical paths beats its dummy-token control, i.e. the "
               "gain over HGT is not only extra capacity."),
        "H5": ("ogbn-mag: Auto-PHGT with canonical vs random paths under the 300-epoch budget; "
               "no direction assumed."),
    },
    "exploratory": [
        "k=2 comparison of all selectors including the manual set {PAP, PSP}",
        "structural x homophily hybrid selector",
        "shuffled-token controls (canonical and random paths)",
        "width-matched HGT (d=68: 711k parameters vs 727k for Auto-PHGT)",
        "single-path utility of all 88 ACM candidates vs structural score, homophily, completion",
        "random path-set properties vs accuracy",
        "ogbn-mag dummy-token control; ogbn-mag HGT and Auto-PHGT continued to 300 epochs",
    ],
    "statistics": ("paired differences over matched seeds with 95% t-intervals; Holm correction "
                   "over the H2-H4 paired tests; H1 as a percentile; MAG (n=3) descriptive plus "
                   "paired t"),
    "test_policy": ("model selection on validation only; each run evaluates test once; the ACM "
                    "test set was already seen in v1, so ACM v2 is not fully confirmatory and a "
                    "new dataset is planned for confirmation"),
    "selectors": {
        "discovered": "canonical structural top-k (unchanged)",
        "random": ("k distinct candidates uniformly without replacement, random.Random(seed); "
                   "for random path sets the seed is 1000 + set index, independent of training"),
        "diverse": ("label-free: walk the structural ranking, take a path whenever it adds a "
                    "relation family not yet covered (a relation and its reverse are one family;"
                    " same-type relations keep their name), then fill by structural order"),
        "homophily": ("top-k by label agreement between each training paper and the "
                      "training-paper endpoints of 16 sampled instances per path (seed = run "
                      "seed); walks back to the start are ignored; no validation/test labels; "
                      "paths with < 100 usable walks rank last"),
        "hybrid": "top-k by structural score x max(homophily - chance, 0); chance = sum of "
                  "squared training class shares",
        "fixed": "explicit paths (manual {PAP, PSP})",
        "rank": "the candidate with the given canonical rank (single-path utility)",
    },
    "controls": {
        "dummy": ("K*I learned input-independent tokens (+ semantic segment embedding, semantic "
                  "LayerNorm and dropout), never masked, replace the semantic tokens"),
        "shuffle": ("semantic tokens and masks permuted across the targets of a batch (global "
                    "RNG while training, fixed seed 1 at evaluation)"),
        "hgt_d68": "HGT with hidden size 68 to match Auto-PHGT's parameter count on ACM",
    },
    "mag": ("canonical v1 runs continue from copies of their final v1 checkpoints with the max "
            "epoch count raised from 100 to 300 and nothing else changed; runs that early-"
            "stopped in v1 finish immediately with their v1 weights; random-path and dummy-token"
            " runs start fresh under the same 300-epoch protocol"),
    "compute": ("ACM runs keep no per-epoch checkpoints; two ACM workers share a GPU; MAG "
                "workers may share GPUs; RTX 2080 Ti; ACM 2 and MAG 3 CPU threads per run"),
}


def _spec(part, exp_id, priority, group, dataset, mode, seed, selection, k, **extra):
    root = V2 / part
    return {"id": exp_id, "part": part, "priority": priority, "group": group,
            "dataset": dataset, "mode": mode, "seed": seed, "path_selection": selection, "k": k,
            "threads": THREADS[dataset],
            "result": str(root / "results" / f"{exp_id}.json"),
            "checkpoint": (str(root / "checkpoints" / f"{exp_id}.pt")
                           if dataset == "ogbn-mag" else None),
            "log": str(root / "logs" / f"{exp_id}.log"), **extra}


def mag_inventory() -> list[dict]:
    specs = []
    for seed in range(3):
        specs.append(_spec("mag", f"v2_mag_auto_phgt_seed{seed}_random_k5_e300", 1,
                           "mag_selection", "ogbn-mag", "auto_phgt", seed, "random", 5,
                           epochs=MAG_EPOCHS))
    for seed in range(3):
        for mode in ("hgt", "auto_phgt"):
            specs.append(_spec(
                "mag", f"v2_mag_{mode}_seed{seed}_discovered_k5_e300", 2, "mag_main_e300",
                "ogbn-mag", mode, seed, "discovered", 5, epochs=MAG_EPOCHS,
                continue_from=str(V1_CHECKPOINTS / f"mag_{mode}_seed{seed}_discovered_k5.pt")))
    for seed in range(3):
        specs.append(_spec("mag", f"v2_mag_auto_phgt_dummy_seed{seed}_discovered_k5_e300", 3,
                           "mag_controls", "ogbn-mag", "auto_phgt", seed, "discovered", 5,
                           epochs=MAG_EPOCHS, token_control="dummy"))
    return specs


def acm_inventory() -> list[dict]:
    specs = []
    for seed in range(10):
        for selection in ACM_SELECTORS:
            specs.append(_spec("acm", f"v2_acm_auto_phgt_seed{seed}_{selection}_k5", 1,
                               "acm_selectors_k5", "acm", "auto_phgt", seed, selection, 5))
        for exp_id, mode, selection, extra in (
                (f"v2_acm_hgt_seed{seed}_discovered_k5", "hgt", "discovered", {}),
                (f"v2_acm_hgt_d68_seed{seed}_discovered_k5", "hgt", "discovered",
                 {"d_model": 68}),
                (f"v2_acm_tokens_seed{seed}_discovered_k5", "tokens", "discovered", {}),
                (f"v2_acm_auto_phgt_dummy_seed{seed}_discovered_k5", "auto_phgt", "discovered",
                 {"token_control": "dummy"}),
                (f"v2_acm_auto_phgt_shuffle_seed{seed}_discovered_k5", "auto_phgt",
                 "discovered", {"token_control": "shuffle"}),
                (f"v2_acm_auto_phgt_shuffle_seed{seed}_random_k5", "auto_phgt", "random",
                 {"token_control": "shuffle"})):
            specs.append(_spec("acm", exp_id, 1, "acm_controls", "acm", mode, seed, selection, 5,
                               **extra))
    for seed in range(10):
        for selection in ACM_SELECTORS:
            specs.append(_spec("acm", f"v2_acm_auto_phgt_seed{seed}_{selection}_k2", 2,
                               "acm_selectors_k2", "acm", "auto_phgt", seed, selection, 2))
        specs.append(_spec("acm", f"v2_acm_auto_phgt_seed{seed}_manual_k2", 2,
                           "acm_selectors_k2", "acm", "auto_phgt", seed, "fixed", 2,
                           paths=MANUAL_ACM))
    for index in range(RANDOM_SETS):
        for seed in range(3):
            specs.append(_spec("acm", f"v2_acm_auto_phgt_seed{seed}_randomset{index:03d}_k5", 3,
                               "acm_random_sets", "acm", "auto_phgt", seed, "random", 5,
                               path_seed=RANDOM_SET_SEED0 + index, random_set=index))
    for rank in range(1, ACM_CANDIDATES + 1):
        for seed in range(3):
            specs.append(_spec("acm", f"v2_acm_auto_phgt_seed{seed}_rank{rank:02d}_k1", 4,
                               "acm_single_path", "acm", "auto_phgt", seed, "rank", 1,
                               ranks=[rank]))
    return specs


def inventory() -> list[dict]:
    specs = mag_inventory() + acm_inventory()
    for key in ("id", "result", "log"):
        values = [s[key] for s in specs]
        assert len(set(values)) == len(values), f"{key} values must be unique"
    checkpoints = [s["checkpoint"] for s in specs if s["checkpoint"]]
    assert len(set(checkpoints)) == len(checkpoints)
    return specs


def use_part(part: str) -> None:
    """Point the shared scheduler's status/claim directories at this part."""
    engine.ARTIFACTS = V2 / part
    engine.CLAIMS = V2 / part / "claims"


def frozen_config(specs) -> dict:
    mag = hyperparameters("ogbn-mag")
    mag["epochs"] = MAG_EPOCHS
    return {"suite": SUITE, "protocol": PROTOCOL_V2,
            "hyperparameters": {"acm": hyperparameters("acm"), "ogbn-mag": mag},
            "experiments": [{key: s.get(key) for key in (
                "id", "part", "priority", "group", "dataset", "mode", "seed", "path_selection",
                "k", "result", "checkpoint", "epochs", "token_control", "d_model", "path_seed",
                "paths", "ranks", "continue_from")} for s in specs]}


def freeze(specs, path: Path = CONFIG_PATH) -> dict:
    """Write the v2 protocol on first use; afterwards refuse to run a different one."""
    current = frozen_config(specs)
    git = git_state()
    launch = {"time": engine._now(), "host": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git": git}
    if path.exists():
        saved = json.loads(path.read_text())
        changed = [key for key in current if saved.get(key) != current[key]]
        if changed:
            raise SystemExit(f"{path} differs from the current v2 protocol in {changed}")
        saved.setdefault("launches", []).append(launch)
        record = saved
    else:
        record = {**current, "frozen_at": launch["time"], "git": git, "launches": [launch]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)
    return record


def worker_kwargs(spec: dict) -> dict:
    """Arguments of run_acm.run / run_mag.run for one v2 experiment."""
    kwargs = dict(mode=spec["mode"], seed=spec["seed"], k=spec["k"],
                  path_selection=spec["path_selection"], device="cuda", output=spec["result"],
                  template_dir=str(V2 / "discovery"))
    for key in ("epochs", "d_model", "path_seed", "paths", "ranks", "token_control"):
        if spec.get(key) is not None:
            kwargs[key] = spec[key]
    if spec["checkpoint"]:
        kwargs["checkpoint"] = spec["checkpoint"]
    else:
        kwargs["no_checkpoint"] = True
    return kwargs


def prepare_continuation(spec: dict) -> None:
    """Copy a v1 checkpoint to the v2 path with only the max epoch count raised."""
    import torch
    from auto_phgt.runtime import save_checkpoint
    state = torch.load(spec["continue_from"], map_location="cpu", weights_only=False)
    state["config"]["hyperparameters"]["epochs"] = spec["epochs"]
    state["continued_from"] = spec["continue_from"]
    save_checkpoint(spec["checkpoint"], state)


def run_worker(exp_id: str) -> None:
    spec = next(s for s in inventory() if s["id"] == exp_id)
    use_part(spec["part"])
    if engine.is_complete(spec):
        print(f"{exp_id} is already complete; nothing to do", flush=True)
        return
    holder = engine.claim(spec)
    if holder:
        raise SystemExit(f"{exp_id} is running in Slurm job {holder}; not starting a duplicate")
    import torch
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"worker expects exactly one visible GPU, sees "
                         f"{torch.cuda.device_count()}")
    if spec.get("continue_from") and not Path(spec["checkpoint"]).exists():
        prepare_continuation(spec)
    kwargs = worker_kwargs(spec)
    kwargs["extra"] = {"experiment_id": exp_id, "suite": SUITE, "part": spec["part"],
                       "group": spec["group"], "priority": spec["priority"],
                       "random_set": spec.get("random_set"),
                       "continued_from": spec.get("continue_from"),
                       "assigned_gpu": {"worker": int(os.environ.get("AUTOPHGT_WORKER", -1)),
                                        "slurm_gpu_id": os.environ.get("AUTOPHGT_GPU"),
                                        "torch_device": "cuda:0"}}
    if spec["dataset"] == "acm":
        from .run_acm import run
    else:
        from .run_mag import run
    result = run(**kwargs)
    print(json.dumps({"experiment_id": exp_id, "test": result["test"],
                      "best_epoch": result["best_epoch"]}), flush=True)


def launch_child(spec, worker, gpu):
    return engine.launch_child(spec, worker, gpu, module="experiments.selector_controls")


def run_part(part: str, *, workers_per_gpu: int, cpu_budget: int, margin_minutes: float):
    specs = [s for s in inventory() if s["part"] == part]
    gpus = engine.visible_gpus()
    engine._log(f"v2 part {part}: host {socket.gethostname()}, Slurm GPUs {gpus}, torch sees "
                f"{engine.check_device_count(gpus)} device(s)")
    left = engine.slurm_time_left_s()
    deadline = time.time() + left - margin_minutes * 60 if left else None
    workers = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    suite = engine.Suite(specs, workers, launch=launch_child, cpu_budget=cpu_budget,
                         deadline=deadline, state_path=V2 / part / "suite_state.json")

    def request_stop(signum, _frame):
        engine._log(f"received signal {signum}")
        suite.stop_requested = True

    import signal
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    suite.run()
    done = sum(engine.is_complete(s) for s in specs)
    engine._log(f"v2 part {part}: {done}/{len(specs)} experiments complete overall")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--part", choices=("mag", "acm"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-budget", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "20")))
    parser.add_argument("--margin-minutes", type=float, default=15.0)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        run_worker(args.worker)
        return
    specs = inventory()
    if args.freeze_only:
        freeze(specs)
        engine._log(f"v2 protocol frozen in {CONFIG_PATH} ({len(specs)} experiments)")
        return
    if not args.part:
        parser.error("--part is required")
    use_part(args.part)
    if args.dry_run:
        part = [s for s in specs if s["part"] == args.part]
        counts = {}
        for spec in part:
            state = "complete" if engine.is_complete(spec) else "pending"
            counts.setdefault(spec["group"], {}).setdefault(state, 0)
            counts[spec["group"]][state] += 1
        print(json.dumps(counts, indent=1))
        print(f"{len(part)} experiments in part {args.part}")
        return
    freeze(specs)
    lock_path = V2 / args.part / "suite.lock"
    lock = engine.acquire_lock(lock_path)
    try:
        run_part(args.part, workers_per_gpu=args.workers_per_gpu, cpu_budget=args.cpu_budget,
                 margin_minutes=args.margin_minutes)
    finally:
        engine.release_lock(lock, lock_path)


if __name__ == "__main__":
    main()
