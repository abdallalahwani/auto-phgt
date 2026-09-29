"""Reusable training loops; experiments choose their dataset and model mode."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .evaluation import evaluate_full_graph, evaluate_sampled_graph, parameter_count
from .runtime import (load_checkpoint, model_device, rng_state, save_checkpoint,
                      set_rng_state, to_device, write_json)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_ids(graph, target_node_type: str = "paper", seed: int = 0,
              acm_validation_size: int = 180):
    """Use official masks; carve validation from ACM's train mask if absent."""
    node = graph[target_node_type]
    train = node.train_mask.nonzero().view(-1)
    test = node.test_mask.nonzero().view(-1)
    if getattr(node, "val_mask", None) is not None:
        val = node.val_mask.nonzero().view(-1)
    else:
        if train.numel() <= acm_validation_size:
            raise ValueError("Training mask is too small for the ACM validation carve-out")
        generator = torch.Generator().manual_seed(seed)
        train = train[torch.randperm(train.numel(), generator=generator)]
        val, train = train[:acm_validation_size], train[acm_validation_size:]
    if min(train.numel(), val.numel(), test.numel()) == 0:
        raise ValueError("Train, validation, and test splits must be nonempty")
    return train, val, test


def _record(epoch, loss, val, elapsed, **extra):
    return {"epoch": epoch, "loss": float(loss), "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"], "epoch_runtime_s": elapsed,
            "val_runtime_s": val["runtime_s"], **extra}


def _progress(**fields):
    print("[progress] " + json.dumps(fields), flush=True)


def progress_path(checkpoint_path):
    """Human-readable sidecar of a checkpoint: counters and history, no tensors."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(checkpoint_path.stem + ".progress.json")


class _TrainingLoop:
    """Early stopping, best-state tracking, and optional per-epoch checkpoints.

    A checkpoint stores the model and optimizer states, epoch counters, the best
    validation score and weights, the history, the run configuration, the seed and
    the RNG states. Rerunning with the same configuration resumes after the last
    completed epoch; a different configuration raises an error.
    """

    def __init__(self, model, optimizer, *, seed, patience, checkpoint_path, config,
                 generators, progress=False):
        self.model = model
        self.progress = progress
        self.optimizer = optimizer
        self.seed = seed
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.generators = generators
        self.state = {"epoch": 0, "best_epoch": 0, "best_val_accuracy": -1.0,
                      "best_state": None, "history": [], "runtime_s": 0.0, "stopped": False}
        self.resumed_from_epoch = None
        saved = load_checkpoint(checkpoint_path, config) if checkpoint_path else None
        if saved is not None:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            set_rng_state(saved["rng"], *generators)
            self.state.update({key: saved[key] for key in self.state})
            self.resumed_from_epoch = saved["epoch"]
        self._base_runtime = self.state["runtime_s"]
        self._start = time.perf_counter()

    def epochs(self, epochs: int):
        return range(0) if self.state["stopped"] else range(self.state["epoch"] + 1, epochs + 1)

    def end_epoch(self, epoch, loss, val, elapsed, **extra) -> bool:
        """Record one epoch, checkpoint it, and return True when early stopping triggers."""
        state = self.state
        state["history"].append(_record(epoch, loss, val, elapsed, **extra))
        state["epoch"] = epoch
        if val["accuracy"] > state["best_val_accuracy"]:
            state["best_val_accuracy"] = val["accuracy"]
            state["best_epoch"] = epoch
            state["best_state"] = {key: value.detach().cpu().clone()
                                   for key, value in self.model.state_dict().items()}
        state["stopped"] = epoch - state["best_epoch"] >= self.patience
        state["runtime_s"] = self._base_runtime + time.perf_counter() - self._start
        if self.checkpoint_path:
            save_checkpoint(self.checkpoint_path, {
                **state, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "config": self.config,
                "seed": self.seed, "rng": rng_state(*self.generators),
            })
            write_json(progress_path(self.checkpoint_path), {
                key: state[key] for key in ("epoch", "best_epoch", "best_val_accuracy",
                                            "runtime_s", "stopped", "history")})
        if self.progress:
            _progress(**state["history"][-1], best_epoch=state["best_epoch"],
                      checkpointed=bool(self.checkpoint_path))
        return state["stopped"]

    def finish(self):
        state = self.state
        if state["best_state"] is None:
            raise RuntimeError("No epoch was trained, so there is no best model to restore")
        self.model.load_state_dict(state["best_state"])
        return {"model": self.model, "history": state["history"],
                "best_epoch": state["best_epoch"],
                "best_val_accuracy": state["best_val_accuracy"],
                "runtime_s": self._base_runtime + time.perf_counter() - self._start,
                "parameters": parameter_count(self.model),
                "epochs_completed": state["epoch"], "early_stopped": state["stopped"],
                "resumed_from_epoch": self.resumed_from_epoch}


def fit_full_graph(model, graph, x_dict, extractor, train_ids, val_ids, *,
                   epochs: int = 100, patience: int = 20, lr: float = 2e-3,
                   weight_decay: float = 1e-3, seed: int = 0,
                   checkpoint_path=None, config=None, progress: bool = False):
    """Train on ACM-sized graphs, selecting the best validation-accuracy epoch.

    The edge indices are moved to the model's device once. ``x_dict`` may be on
    CPU or on the model's device; the caller decides.
    """
    set_seed(seed)
    extractor.generator.manual_seed(seed)
    device = model_device(model)
    labels = graph[model.target_node_type].y
    train_labels = labels[train_ids].to(device)
    edges = to_device(graph.edge_index_dict, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loop = _TrainingLoop(model, optimizer, seed=seed, patience=patience,
                         checkpoint_path=checkpoint_path, config=config,
                         generators=[extractor.generator], progress=progress)
    for epoch in loop.epochs(epochs):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        instances = None if model.mode == "hgt" else extractor.sample(train_ids)
        logits = model(x_dict, edges, instances=instances,
                       target_ids=train_ids if instances is None else None)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        val = evaluate_full_graph(model, x_dict, edges, extractor, labels, val_ids)
        if loop.end_epoch(epoch, loss.item(), val, time.perf_counter() - epoch_start):
            break
    return loop.finish()


def fit_sampled_graph(model, graph, x_dict, extractor, train_sampler, eval_sampler,
                      train_ids, val_ids, *, epochs: int = 100, patience: int = 20,
                      batch_size: int = 512, lr: float = 1e-3,
                      weight_decay: float = 1e-3, seed: int = 0,
                      checkpoint_path=None, config=None, progress: bool = False,
                      progress_every: int = 250):
    """Train on sampled MAG subgraphs and select by full validation accuracy.

    The graph, features and labels stay on CPU. Only each batch's sampled edges,
    gathered feature rows, path tensors and labels are moved to the model device.
    """
    set_seed(seed)
    extractor.generator.manual_seed(seed)
    train_sampler.generator.manual_seed(seed)
    device = model_device(model)
    labels = graph[model.target_node_type].y
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loop = _TrainingLoop(model, optimizer, seed=seed, patience=patience,
                         checkpoint_path=checkpoint_path, config=config,
                         generators=[extractor.generator, train_sampler.generator, generator],
                         progress=progress)
    for epoch in loop.epochs(epochs):
        epoch_start = time.perf_counter()
        model.train()
        order = train_ids[torch.randperm(train_ids.numel(), generator=generator)]
        total_loss = 0.0
        batches = 0
        sampling_s = 0.0  # CPU neighbour + path sampling; the rest is transfer + GPU work
        for offset in range(0, order.numel(), batch_size):
            seeds = order[offset : offset + batch_size]
            sample_start = time.perf_counter()
            edges, n_id, count = train_sampler.sample(model.target_node_type, seeds)
            targets = n_id[model.target_node_type][:count]
            instances = None if model.mode == "hgt" else extractor.sample(targets)
            sampling_s += time.perf_counter() - sample_start
            optimizer.zero_grad()
            logits = model(x_dict, edges, instances=instances,
                           target_ids=targets if instances is None else None,
                           n_id_dict=n_id, target_local=torch.arange(count))
            loss = F.cross_entropy(logits, labels[targets].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            if progress and batches % progress_every == 0:
                _progress(epoch=epoch, batches=batches,
                          elapsed_s=time.perf_counter() - epoch_start, sampling_s=sampling_s)
        train_s = time.perf_counter() - epoch_start
        val = evaluate_sampled_graph(model, x_dict, extractor, eval_sampler,
                                     labels, val_ids, batch_size=batch_size)
        if loop.end_epoch(epoch, total_loss / batches, val,
                          time.perf_counter() - epoch_start, train_runtime_s=train_s,
                          batches=batches, sampling_s=sampling_s):
            break
    return loop.finish()
