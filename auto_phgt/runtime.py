"""Device selection, run metadata, result files, and lightweight checkpoints."""

from __future__ import annotations

import json
import os
import platform
import random
import resource
import socket
import subprocess
from pathlib import Path

import numpy as np
import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_device(spec: str = "auto") -> torch.device:
    """Map ``auto``/``cpu``/``cuda`` (or ``cuda:N``) to a device; ``auto`` needs real CUDA."""
    spec = str(spec).lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available. "
                           "Run inside a Slurm GPU allocation (sbatch --gpus=1).")
    return device


def to_device(tensors: dict, device) -> dict:
    """Move every tensor of a dict; use only for graphs small enough to fit, e.g. ACM."""
    return {key: value.to(device) for key, value in tensors.items()}


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def device_info(device) -> dict:
    device = torch.device(device)
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(gpu_name=props.name, gpu_total_memory_mib=props.total_memory // 2**20,
                    gpu_capability=f"{props.major}.{props.minor}",
                    gpu_uuid=str(getattr(props, "uuid", "")) or None,
                    gpu_pci_bus_id=getattr(props, "pci_bus_id", None))
    return info


def cuda_memory(device) -> dict:
    device = torch.device(device)
    if device.type != "cuda":
        return {}
    return {"allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "max_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "max_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def peak_cpu_rss_mib() -> float:
    """Peak resident memory of this process (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def read_git_head(root) -> str | None:
    """Commit of HEAD read from ``.git`` itself; compute nodes may lack the git binary."""
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def git_state(root: str | Path | None = None) -> dict:
    """Current commit and whether tracked files differ from it; ``None`` outside Git."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"], check=True, capture_output=True,
                               text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": read_git_head(root), "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def write_json(path: str | Path, record: dict) -> Path:
    """Write atomically, so an interrupted job cannot leave a truncated result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def rng_state(*generators) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "generators": [g.get_state() for g in generators]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict, *generators) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for generator, saved in zip(generators, state["generators"]):
        generator.set_state(saved)
    if "cuda" in state and torch.cuda.is_available() \
            and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, state: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, config: dict | None = None):
    """Return a saved state, or ``None`` if absent; reject checkpoints of another run."""
    path = Path(path)
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if config is not None and state.get("config") != config:
        raise ValueError(f"Checkpoint {path} was written for a different configuration. "
                         "Delete it or pass --restart to train from scratch.")
    return state
