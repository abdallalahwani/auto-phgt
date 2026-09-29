"""HGB-style full-graph training shared by strong HGT and every Auto-PHGT variant.

AdamW with a OneCycle schedule (HGB train_hgt.py), early stopping on validation loss with
patience 30 and at most 300 epochs, restoring the best-validation-loss weights. The test
split is never touched here; final runs evaluate it once through ``evaluate_test``.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from auto_phgt.model import AutoPHGT
from auto_phgt.runtime import cuda_memory, to_device
from auto_phgt.tokenization import MetaPathInstanceExtractor
from auto_phgt.training import set_seed

from .hgb_protocol import AUTO_PHGT, BACKBONE_FIXED


def backbone(cfg: dict) -> dict:
    """A tuned configuration completed with the fixed backbone settings."""
    return {**BACKBONE_FIXED, **cfg}


def build_model(graph, templates, num_classes, x_dict, cfg, mode, *, token_control=None,
                width=None, instances=AUTO_PHGT["instances_per_path"]):
    cfg = backbone(cfg)
    kwargs = {}
    if token_control == "dummy":
        kwargs["num_dummy_tokens"] = len(templates) * instances
    return AutoPHGT.build(
        graph, templates, num_classes, x_dict=x_dict, d_model=width or cfg["d_model"],
        pooling=AUTO_PHGT["pooling"], token_dropout=AUTO_PHGT["token_dropout"],
        hgt_layers=cfg["layers"], hgt_heads=cfg["heads"],
        fusion_layers=AUTO_PHGT["fusion_layers"], fusion_heads=AUTO_PHGT["fusion_heads"],
        ffn_mult=AUTO_PHGT["ffn_mult"], dropout=cfg["dropout"], mode=mode,
        token_control=token_control, **kwargs)


def extractor_for(graph, templates, seed, instances=AUTO_PHGT["instances_per_path"]):
    return MetaPathInstanceExtractor(graph, templates, instances_per_path=instances, seed=seed)


def micro_macro(y, logits) -> dict:
    truth = y.view(-1).cpu().numpy()
    pred = logits.argmax(-1).view(-1).cpu().numpy()
    return {"micro_f1": float(f1_score(truth, pred, average="micro")),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0))}


def _forward(model, x_dict, edges, ids, instances):
    if model.mode == "hgt":
        return model(x_dict, edges, target_ids=ids)
    return model(x_dict, edges, instances=instances)


@torch.no_grad()
def predict(model, graph, x_dict, extractor, ids, seed: int = 1):
    """Logits for ``ids`` with a fixed path-instance draw; restores the training mode."""
    device = next(model.parameters()).device
    prior = model.training
    model.eval()
    try:
        edges = to_device(graph.edge_index_dict, device)
        instances = None if model.mode == "hgt" else extractor.sample(ids, seed=seed)
        return _forward(model, x_dict, edges, ids.to(device), instances)
    finally:
        model.train(prior)


def fit(model, graph, x_dict, extractor, train_ids, val_ids, cfg, seed: int):
    cfg = backbone(cfg)
    set_seed(seed)
    if extractor is not None:
        extractor.generator.manual_seed(seed)
    device = next(model.parameters()).device
    labels = graph[model.target_node_type].y
    y_train, y_val = labels[train_ids].to(device), labels[val_ids].to(device)
    edges = to_device(graph.edge_index_dict, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], total_steps=cfg["max_epochs"], pct_start=cfg["pct_start"])
    token_mode = model.mode != "hgt"
    val_instances = extractor.sample(val_ids, seed=1) if token_mode else None
    best = {"loss": math.inf, "epoch": 0, "state": None, "micro_f1": None, "macro_f1": None}
    history, stale, start = [], 0, time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        instances = extractor.sample(train_ids) if token_mode else None
        loss = F.cross_entropy(_forward(model, x_dict, edges, train_ids.to(device), instances),
                               y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = _forward(model, x_dict, edges, val_ids.to(device), val_instances)
            val_loss = float(F.cross_entropy(val_logits, y_val))
        scores = micro_macro(y_val, val_logits)
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
    return {"best_epoch": best["epoch"], "best_val_loss": best["loss"],
            "val": {"micro_f1": best["micro_f1"], "macro_f1": best["macro_f1"]},
            "epochs_completed": len(history), "early_stopped": len(history) < cfg["max_epochs"],
            "train_runtime_s": time.perf_counter() - start, "gpu_memory": cuda_memory(device),
            "parameters": sum(p.numel() for p in model.parameters()), "history": history}


def evaluate_test(model, graph, x_dict, extractor, test_ids) -> dict:
    """The single test evaluation of a final run."""
    logits = predict(model, graph, x_dict, extractor, test_ids, seed=1)
    return {**micro_macro(graph[model.target_node_type].y[test_ids], logits),
            "n": int(test_ids.numel())}
