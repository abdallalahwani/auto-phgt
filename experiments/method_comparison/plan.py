"""Frozen final comparison, retaining compatible V3/V4 runs without retraining them."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from experiments.residual_selection.plan import AUTO_PHGT, BACKBONE_FIXED

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v5"
LOCK = OUT / "protocol_lock.json"
DATASETS = {
    "acm": {"name": "ACM", "root": "data/acm", "target": "paper", "multilabel": False},
    "dblp": {"name": "DBLP", "root": "data/hgb/dblp", "target": "author", "multilabel": False},
    "imdb": {"name": "IMDB", "root": "data/hgb/imdb", "target": "movie", "multilabel": True},
}
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("hgt", "random", "fastpath", "canonical", "hybrid", "edgeoverlap")
KS = (2, 5)
EDGE_OVERLAP = {"max_sources": 1024, "instances_per_path": 32, "seed": 0,
                "empty_union_overlap": 0.0, "relevance": "TI_cov"}
FASTPATH = {"folds": 3, "fold_seed_offset": 4000, "smoothing_eps": 0.1}
IMDB_CONFIG = {**BACKBONE_FIXED, "feat": "given", "layers": 2, "lr": 0.001,
               "wd": 0.0001, "dropout": 0.2}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def software_versions() -> dict:
    return {"python": platform.python_version(),
            **{name: version(name) for name in (
                "torch", "torch-geometric", "numpy", "scipy", "scikit-learn")}}


def lmax(dataset: str, k: int | None) -> int:
    return 6 if dataset == "dblp" and k == 5 else 4


def tasks(dataset: str | None = None) -> list[dict]:
    if dataset is not None and dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    out = []
    for ds in ([dataset] if dataset is not None else DATASETS):
        for method in METHODS:
            for k in ((None,) if method == "hgt" else KS):
                hops = lmax(ds, k)
                name = "hgt" if k is None else f"{method}_k{k}_L{hops}"
                for seed in SEEDS:
                    out.append({"id": f"{ds}__{name}__seed{seed}", "dataset": ds,
                                "method": method, "mode": "hgt" if k is None else "auto_phgt",
                                "k": k, "lmax": hops, "seed": seed})
    return out


def source_path(spec: dict) -> Path | None:
    ds, method, k, seed = (spec[key] for key in ("dataset", "method", "k", "seed"))
    if method == "edgeoverlap" or ds == "imdb" or (ds == "dblp" and method == "fastpath"):
        return None
    if ds == "acm":
        name = "hgt" if method == "hgt" else f"{method}_k{k}"
        return ROOT / "artifacts/v4_hedge/results/acm" / f"acm__{name}__seed{seed}.json"
    name = "hgt_strong" if method == "hgt" else f"{method}_k{k}"
    if method != "hgt" and k == 5:
        name += "_L6"
    return ROOT / "artifacts/v3/results/dblp" / f"dblp__{name}__seed{seed}.json"


def configs() -> dict:
    out = {"imdb": dict(IMDB_CONFIG)}
    for ds in ("acm", "dblp"):
        spec = next(item for item in tasks(ds) if item["method"] == "hgt")
        path = source_path(spec)
        if path is None or not path.exists():
            raise ValueError(f"required existing HGT configuration is missing for {ds}")
        out[ds] = read_json(path)["config"]
    return out


def validate_reuse(spec: dict, record: dict, config: dict) -> None:
    expected = {"dataset": spec["dataset"], "seed": spec["seed"], "mode": spec["mode"],
                "k": spec["k"], "lmax": spec["lmax"], "status": "completed"}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{spec['id']}: incompatible reused {key}")
    selector = None if spec["method"] == "hgt" else spec["method"]
    if record.get("selector") != selector or record.get("token_control") is not None:
        raise ValueError(f"{spec['id']}: incompatible reused selector/control")
    if record.get("width") is not None or record.get("config") != config:
        raise ValueError(f"{spec['id']}: incompatible reused backbone")
    if spec["k"] is not None and len(record.get("templates") or []) != spec["k"]:
        raise ValueError(f"{spec['id']}: reused path count differs")
    if not {"micro_f1", "macro_f1"} <= record.get("test", {}).keys():
        raise ValueError(f"{spec['id']}: reused metrics are incomplete")


def result_path(spec: dict) -> Path:
    return OUT / "results" / spec["dataset"] / f"{spec['id']}.json"


def selector_path(dataset: str, hops: int, kind: str, seed: int | None = None) -> Path:
    suffix = "" if seed is None else f"__seed{seed}"
    return OUT / "selectors" / dataset / f"L{hops}__{kind}{suffix}.json"


def required_selection(spec: dict) -> Path | None:
    method = spec["method"]
    if method == "hgt":
        return None
    kind = method if method in ("fastpath", "edgeoverlap") else "baselines"
    seed = None if method == "edgeoverlap" else spec["seed"]
    return selector_path(spec["dataset"], spec["lmax"], kind, seed)


def code_hashes() -> dict:
    files = list((ROOT / "experiments/method_comparison").glob("*.py"))
    files += list((ROOT / "auto_phgt").glob("*.py"))
    files += [ROOT / path for path in (
        "experiments/common.py", "experiments/residual_selection/data.py", "experiments/residual_selection/train.py",
        "experiments/residual_selection/plan.py", "experiments/lightweight_selection/transitions.py",
        "experiments/lightweight_selection/selectors.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def processed_path(dataset: str) -> Path:
    spec = DATASETS[dataset]
    return ROOT / spec["root"] / spec["name"].lower() / "processed/data.pt"


def load_graph(dataset: str):
    from torch_geometric.datasets import HGBDataset
    spec = DATASETS[dataset]
    return HGBDataset(root=str(ROOT / spec["root"]), name=spec["name"])[0]


def legacy_dataset_records() -> dict:
    return {
        "acm": read_json(ROOT / "artifacts/v4_hedge/protocol/datasets.json")["acm"]["record"],
        "dblp": read_json(ROOT / "artifacts/v3/provenance/datasets.json")["dblp"],
    }


def freeze() -> dict:
    from auto_phgt.runtime import git_state, write_json
    from experiments.residual_selection import data
    import time

    if LOCK.exists():
        raise FileExistsError("V5 is already frozen; refusing to overwrite its protocol")
    for old_lock in (
        ROOT / "artifacts/v3/protocol/protocol_lock.json",
        ROOT / "artifacts/v4_hedge/protocol/protocol_lock.json",
    ):
        for relative, expected in read_json(old_lock)["code"].items():
            if digest(ROOT / relative)[:len(expected)] != expected:
                raise ValueError(f"frozen dependency changed: {relative}")
    cfg = configs()
    datasets, reused = {}, {}
    for ds, spec in DATASETS.items():
        graph = load_graph(ds)
        path = processed_path(ds)
        train, val, test = data.split(graph, spec["target"], 0)
        stat = path.stat()
        datasets[ds] = {
            **spec, "processed_sha256": digest(path), "processed_md5": digest(path, "md5"),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "nodes": {name: graph[name].num_nodes for name in graph.node_types},
            "edges": {"__".join(et): graph[et].num_edges for et in graph.edge_types},
            "label_shape": list(graph[spec["target"]].y.shape),
            "split_seed0": {"train": train.numel(), "val": val.numel(), "test": test.numel()},
        }
    sources = legacy_dataset_records()
    for ds, record in sources.items():
        if record["processed_md5"] != datasets[ds]["processed_md5"]:
            raise ValueError(f"{ds}: dataset differs from the reused campaign")
    for spec in tasks():
        source = source_path(spec)
        if source is not None and source.exists():
            validate_reuse(spec, read_json(source), cfg[spec["dataset"]])
            reused[spec["id"]] = {"path": str(source.relative_to(ROOT)), "sha256": digest(source)}
    ti_source = ROOT / "artifacts/v4_hedge/selectors/transition_info/acm.json"
    secondary = {}
    for method in ("ti", "ti_set"):
        for k in KS:
            for seed in SEEDS:
                source = ROOT / "artifacts/v4_hedge/results/acm" / f"acm__{method}_k{k}__seed{seed}.json"
                spec = {"id": source.stem, "dataset": "acm", "method": method, "k": k,
                        "lmax": 4, "seed": seed, "mode": "auto_phgt"}
                validate_reuse(spec, read_json(source), cfg["acm"])
                secondary[source.stem] = {"path": str(source.relative_to(ROOT)), "sha256": digest(source)}
    settings = {
        "suite": "autophgt_v5_final", "datasets": datasets, "seeds": list(SEEDS),
        "software": software_versions(),
        "methods": list(METHODS), "ks": list(KS), "configs": cfg, "auto_phgt": AUTO_PHGT,
        "edge_overlap": EDGE_OVERLAP, "fastpath": FASTPATH,
        "reuse": reused, "tasks": tasks(), "secondary_acm_references": secondary,
        "ti_acm_source": {"path": str(ti_source.relative_to(ROOT)), "sha256": digest(ti_source)},
        "test_label_provenance": {
            "announcement": "https://github.com/THUDM/HGB#heterogeneous-graph-benchmark",
            "public_release_date": "2023-03-02",
            "folder": "https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX",
            "imdb_archive_id": "18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2",
            "verification": "fresh IMDB download uses the same archive ID as the official public-test-label release; the installed PyG docstring retains the older randomized-label warning",
        },
        "rules": {
            "hgt": "one run per dataset/seed, shared between both k tables",
            "dblp": "preserve V3: k=2 uses L<=4; k=5 uses L<=6; not a pure k-only comparison",
            "fastpath": "independent V4 negative cross-fitted CE, not FastPath-Set",
            "imdb": "binary BCE, fixed sigmoid threshold 0.5, validation BCE stopping; F1 is not accuracy",
            "imdb_fastpath": "negative OOF binary cross-entropy; per-class training-fold priors",
            "imdb_hybrid": "structural score times positive lift of training-label Jaccard over its training-only chance expectation",
            "initialization": "ACM/DBLP preserve the legacy build-then-fit seeding workflow; IMDB seeds before model construction",
            "test": "no test-driven branching; pilot uses training/validation only; every new final evaluates test once",
            "cost": "reuse completed compatible runs; one worker per GPU; CPU selection cached across both k values",
            "scope": "ACM, DBLP, IMDB only; MAG and RCMS excluded",
            "hypothesis": "EdgeOverlap vs type/fingerprint novelty can use existing ACM TI-Set results as secondary context; no new TI-Set training is requested",
        },
    }
    blob = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    settings.update(protocol_hash=hashlib.sha256(blob).hexdigest()[:16], code=code_hashes(),
                    frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git=git_state())
    write_json(LOCK, settings)
    return settings


def check_lock(dataset: str | None = None, *, full_data_check: bool = False) -> dict:
    record = read_json(LOCK)
    if record["code"] != code_hashes():
        raise ValueError("V5 source/dependency changed after freezing")
    if record["software"] != software_versions():
        raise ValueError("V5 Python/dependency versions changed after freezing")
    if dataset is not None:
        path, expected = processed_path(dataset), record["datasets"][dataset]
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"{dataset}: processed dataset changed after freezing")
        if full_data_check and digest(path) != expected["processed_sha256"]:
            raise ValueError(f"{dataset}: processed dataset checksum mismatch")
    return record


def final_record(spec: dict, lock: dict) -> dict | None:
    reference = lock["reuse"].get(spec["id"])
    if reference is not None:
        path = ROOT / reference["path"]
        if digest(path) != reference["sha256"]:
            raise ValueError(f"{spec['id']}: reused result changed")
        record = read_json(path)
        validate_reuse(spec, record, lock["configs"][spec["dataset"]])
        return {**record, "v5_task": spec, "reused_from": reference["path"]}
    path = result_path(spec)
    if not path.exists():
        return None
    record = read_json(path)
    if record.get("protocol_hash") != lock["protocol_hash"] or record.get("task") != spec:
        raise ValueError(f"{spec['id']}: existing result belongs to a different protocol")
    if record.get("status") != "completed":
        raise ValueError(f"{spec['id']}: final result is not complete")
    return record
