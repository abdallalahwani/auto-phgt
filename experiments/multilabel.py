"""Multi-label training and independent FastPath scoring.

The backbone, optimizer/schedule configuration, and path-instance draws follow
the shared HGB-style training protocol.
Only the objective and metrics change: mean BCE over nodes and labels, and F1 at
``sigmoid(logits) >= 0.5``. No categorical accuracy or set-aware probe is used.

PyG HGB IMDB supplies [N, C] multi-hot labels: C is the output width, and masks,
not label values, identify labeled rows. The verified public IMDB archive includes
test labels (HGB's 2023-03-02 release), despite PyG's legacy randomized-label warning.
Keep the official train/test masks; validation is carved only from official training.
Categorical class-count and homophily helpers cannot consume these labels as-is.
"""

from __future__ import annotations

import math
import time
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as F

from auto_phgt.runtime import cuda_memory, to_device
from auto_phgt.training import set_seed
from experiments.hgb_training import _forward, backbone, predict

_BCE_CLIP = 1e-12
_FOLD_RULE = "label-independent seeded permutation dealt round-robin"


def _targets(labels, ids, name):
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("multi-label targets must have shape [n_nodes, n_labels]")
    ids = torch.as_tensor(ids)
    if (ids.ndim != 1 or ids.numel() == 0
            or ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError(f"{name} must be a nonempty one-dimensional integer ID sequence")
    ids = ids.detach().cpu().long()
    if bool(((ids < 0) | (ids >= labels.shape[0])).any()):
        raise ValueError(f"{name} contains an out-of-range node ID")
    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} contains duplicate node IDs")
    selected = labels[ids.to(labels.device)]
    if not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError("multi-label targets must be finite binary indicators")
    return ids, selected.detach().float()


def _check_logits(labels, logits):
    if logits.ndim != 2 or logits.shape != labels.shape:
        raise ValueError("logits and multi-label targets must have the same [n, C] shape")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")


def _indicator_f1(labels, predictions) -> dict:
    truth, pred = labels.astype(bool), predictions.astype(bool)
    tp = (truth & pred).sum(axis=0)
    denominator = truth.sum(axis=0) + pred.sum(axis=0)
    per_label = np.divide(2.0 * tp, denominator, out=np.zeros(tp.shape, dtype=float),
                          where=denominator > 0)
    total = int(denominator.sum())
    return {"micro_f1": float(2 * tp.sum() / total) if total else 0.0,
            "macro_f1": float(per_label.mean())}


def multilabel_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict:
    """Micro/macro F1 with a fixed inclusive sigmoid threshold of 0.5; zero-division is 0."""
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("multi-label targets must have shape [n, C]")
    _, labels = _targets(labels, torch.arange(labels.shape[0]), "metric_ids")
    _check_logits(labels, logits)
    return _indicator_f1(labels.cpu().numpy(),
                         (logits.detach().sigmoid() >= 0.5).cpu().numpy())


def fit_multilabel(model, graph, x_dict, extractor, train_ids, val_ids, cfg, seed: int) -> dict:
    """Fit using BCEWithLogits and best-validation-loss restoration.

    ``cfg`` is the caller's configuration, completed by ``backbone``.
    Only the supplied training/validation labels are read. The incoming model mode is
    restored, including when a forward pass fails.
    """
    cfg = backbone(cfg)
    for key in ("max_epochs", "patience"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], Integral) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    labels = graph[model.target_node_type].y
    train_ids, y_train = _targets(labels, train_ids, "train_ids")
    val_ids, y_val = _targets(labels, val_ids, "val_ids")
    if bool(torch.isin(train_ids, val_ids).any()):
        raise ValueError("training and validation IDs must be disjoint")
    token_mode = model.mode != "hgt"
    if token_mode and extractor is None:
        raise ValueError("token-based modes require an extractor")
    set_seed(seed)
    if extractor is not None:
        extractor.generator.manual_seed(seed)
    device = next(model.parameters()).device
    y_train, y_val = y_train.to(device), y_val.to(device)
    edges = to_device(graph.edge_index_dict, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], total_steps=cfg["max_epochs"], pct_start=cfg["pct_start"])
    val_instances = extractor.sample(val_ids, seed=1) if token_mode else None
    best = {"loss": math.inf, "epoch": 0, "state": None, "micro_f1": None, "macro_f1": None}
    history, stale, start = [], 0, time.perf_counter()
    was_training = model.training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            instances = extractor.sample(train_ids) if token_mode else None
            logits = _forward(model, x_dict, edges, train_ids.to(device), instances)
            _check_logits(y_train, logits)
            loss = F.binary_cross_entropy_with_logits(logits, y_train.to(logits))
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_logits = _forward(model, x_dict, edges, val_ids.to(device), val_instances)
                _check_logits(y_val, val_logits)
                val_loss = float(F.binary_cross_entropy_with_logits(val_logits, y_val.to(val_logits)))
            scores = multilabel_metrics(y_val, val_logits)
            history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                            "val_micro_f1": scores["micro_f1"], "val_macro_f1": scores["macro_f1"]})
            if val_loss < best["loss"]:
                best = {"loss": val_loss, "epoch": epoch, **scores,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
        model.load_state_dict(best["state"])
    finally:
        model.train(was_training)
    return {"best_epoch": best["epoch"], "best_val_loss": best["loss"],
            "val": {"micro_f1": best["micro_f1"], "macro_f1": best["macro_f1"]},
            "epochs_completed": len(history), "early_stopped": len(history) < cfg["max_epochs"],
            "train_runtime_s": time.perf_counter() - start, "gpu_memory": cuda_memory(device),
            "parameters": sum(p.numel() for p in model.parameters()), "history": history}


def evaluate_multilabel(model, graph, x_dict, extractor, ids) -> dict:
    """Evaluate only ``ids``, with a fixed seed-1 path draw and model-mode preservation."""
    ids, labels = _targets(graph[model.target_node_type].y, ids, "ids")
    logits = predict(model, graph, x_dict, extractor, ids, seed=1)
    return {**multilabel_metrics(labels, logits), "n": int(ids.numel())}


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return array.astype(np.float64, copy=False)


def _binary_array(labels):
    labels = _real_array(labels, "labels")
    if labels.ndim != 2 or labels.shape[0] < 2 or labels.shape[1] == 0:
        raise ValueError("labels must have shape [n_train >= 2, n_labels >= 1]")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary indicators")
    return labels


def propagate_multilabel(block, labels, fold_ids):
    """Return unsmoothed probabilities ``[F,n,C]``, known mass ``[F,n]``, priors ``[F,C]``.

    Rows/columns of the dense nonnegative ``block`` follow the given TRAIN-label order.
    Its diagonal is removed without modifying the caller's matrix. For fold j, only
    endpoints outside j contribute labels or the per-label prevalence prior. Each
    probability is divided by *known endpoint mass*, not the number of positive labels.
    No-evidence rows fall back to the known-fold prior. Folds must be contiguous IDs
    starting at zero, with at least two nonempty folds.
    """
    labels = _binary_array(labels)
    n = labels.shape[0]
    weights = _real_array(block, "block")
    if weights.shape != (n, n) or (weights < 0).any():
        raise ValueError("block must be a nonnegative square matrix aligned with labels")
    folds = np.asarray(fold_ids)
    if folds.shape != (n,) or folds.dtype.kind not in "iu":
        raise ValueError("fold_ids must be a one-dimensional integer array aligned with labels")
    unique = np.unique(folds)
    if unique.size < 2 or not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("fold_ids must contain at least two contiguous nonempty folds from zero")
    weights = weights.copy()
    np.fill_diagonal(weights, 0.0)
    probabilities = np.empty((unique.size, n, labels.shape[1]), dtype=np.float64)
    mass = np.empty((unique.size, n), dtype=np.float64)
    priors = np.empty((unique.size, labels.shape[1]), dtype=np.float64)
    for j in range(unique.size):
        known = folds != j
        prior = labels[known].mean(axis=0)
        evidence = weights[:, known]
        w = evidence.sum(axis=1)
        if not np.isfinite(w).all():
            raise ValueError("known endpoint mass must be finite")
        numerator = evidence @ labels[known]
        probabilities[j] = np.divide(
            numerator, w[:, None], out=np.broadcast_to(prior, numerator.shape).copy(),
            where=w[:, None] > 0)
        mass[j], priors[j] = w, prior
    return probabilities, mass, priors


def multilabel_fastpath_scores(block, labels, *, folds=3, seed=0, smoothing_eps=0.1) -> dict:
    """Independent FastPath score, replacing categorical CE with mean OOF BCE.

    Balanced shuffled folds depend only on n, fold count and seed, never on labels.
    ``Q_FP = -BCE`` is ranked descending (canonical rank breaks ties), without any
    set-aware fitting. ``CE`` is an alias of ``BCE`` for the existing result schema.
    F1 uses unsmoothed OOF probabilities; smoothing toward the known-fold
    prior and clipping to [1e-12, 1-1e-12] are used only to evaluate BCE.
    """
    labels = _binary_array(labels)
    n = labels.shape[0]
    if isinstance(folds, bool) or not isinstance(folds, Integral) or not 2 <= folds <= n:
        raise ValueError("folds must be an integer between 2 and n_train")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(smoothing_eps, bool) or not isinstance(smoothing_eps, Real)
            or not math.isfinite(smoothing_eps) or not 0 <= smoothing_eps <= 1):
        raise ValueError("smoothing_eps must be a finite number in [0, 1]")
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[np.random.default_rng(seed).permutation(n)] = np.arange(n) % folds
    probabilities, mass, priors = propagate_multilabel(block, labels, fold_ids)
    idx = np.arange(n)
    oof, known_mass = probabilities[fold_ids, idx], mass[fold_ids, idx]
    smoothed = (1.0 - smoothing_eps) * oof + smoothing_eps * priors[fold_ids]
    clipped = np.clip(smoothed, _BCE_CLIP, 1.0 - _BCE_CLIP)
    bce = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
    sizes = np.bincount(fold_ids, minlength=folds)
    return {"Q_FP": -bce, "CE": bce, "BCE": bce,
            **_indicator_f1(labels, oof >= 0.5),
            "coverage": float((known_mass > 0).mean()), "labelled_mass": float(known_mass.mean()),
            "folds": int(folds), "fold_seed": int(seed), "fold_rule": _FOLD_RULE,
            "fold_ids": fold_ids.tolist(), "fold_sizes": sizes.tolist(),
            "known_sizes": (n - sizes).tolist(), "smoothing_eps": float(smoothing_eps),
            "bce_clip": _BCE_CLIP}
