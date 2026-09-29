"""Classification metrics and deterministic evaluation helpers."""

from __future__ import annotations

import time

import torch
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels, logits):
    truth = torch.as_tensor(labels).view(-1).cpu().numpy()
    prediction = logits.argmax(-1).view(-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def parameter_count(model) -> int:
    """Count distinct parameters registered in the selected model mode."""
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate_full_graph(model, x_dict, edge_index_dict, extractor, labels, target_ids,
                        seed: int = 1):
    ids = torch.as_tensor(target_ids, dtype=torch.long).view(-1)
    if ids.numel() == 0:
        raise ValueError("Cannot evaluate an empty target split")
    prior_mode = model.training
    model.eval()
    start = time.perf_counter()
    try:
        with torch.no_grad():
            instances = None if model.mode == "hgt" else extractor.sample(ids, seed=seed)
            logits = model(x_dict, edge_index_dict, instances=instances,
                           target_ids=ids if instances is None else None)
    finally:
        model.train(prior_mode)
    return {**classification_metrics(labels[ids], logits),
            "runtime_s": time.perf_counter() - start}


def evaluate_sampled_graph(model, x_dict, extractor, sampler, labels, target_ids,
                           batch_size: int = 512, seed: int = 1):
    """Evaluate a large graph with fixed path and neighbor draws per call."""
    ids = torch.as_tensor(target_ids, dtype=torch.long).view(-1)
    if ids.numel() == 0:
        raise ValueError("Cannot evaluate an empty target split")
    prior_mode = model.training
    model.eval()
    sampler.generator.manual_seed(seed)
    start = time.perf_counter()
    logits_parts = []
    try:
        with torch.no_grad():
            for offset in range(0, ids.numel(), batch_size):
                seeds = ids[offset : offset + batch_size]
                edges, n_id, count = sampler.sample(model.target_node_type, seeds)
                targets = n_id[model.target_node_type][:count]
                instances = (None if model.mode == "hgt" else
                             extractor.sample(targets, seed=seed + offset))
                logits_parts.append(model(
                    x_dict, edges, instances=instances,
                    target_ids=targets if instances is None else None,
                    n_id_dict=n_id, target_local=torch.arange(count),
                ).cpu())
    finally:
        model.train(prior_mode)
    logits = torch.cat(logits_parts)
    return {**classification_metrics(labels[ids], logits),
            "runtime_s": time.perf_counter() - start}
