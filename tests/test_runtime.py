import json

import pytest
import torch

from auto_phgt.model import AutoPHGT
from auto_phgt.runtime import (git_state, load_checkpoint, resolve_device, save_checkpoint,
                               write_json)
from auto_phgt.sampling import HeteroNeighborSampler
from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features
from auto_phgt.training import fit_full_graph, fit_sampled_graph
from experiments.common import build_result, discovery_record, smoke_modes

MODES = ("hgt", "tokens", "auto_phgt")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA allocation"))]


def masks(graph):
    graph["paper"].train_mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
    graph["paper"].test_mask = ~graph["paper"].train_mask
    return torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5])


def build(graph, templates, mode, device="cpu"):
    torch.manual_seed(0)
    features = impute_missing_features(graph)
    extractor = MetaPathInstanceExtractor(graph, templates, instances_per_path=2, seed=0)
    model = AutoPHGT.build(graph, templates, 3, x_dict=features, d_model=16, hgt_layers=1,
                           hgt_heads=2, fusion_layers=1, fusion_heads=2, dropout=0.0,
                           mode=mode)
    return model.to(device), features, extractor


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", MODES)
def test_full_graph_training_on_device(toy_graph, toy_templates, mode, device):
    train, val = masks(toy_graph)
    model, features, extractor = build(toy_graph, toy_templates, mode, device)
    features = {key: value.to(device) for key, value in features.items()}
    result = fit_full_graph(model, toy_graph, features, extractor, train, val,
                            epochs=2, patience=5, seed=0)
    assert result["epochs_completed"] == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in result["history"])
    assert next(model.parameters()).device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", MODES)
def test_sampled_training_keeps_graph_on_cpu(toy_graph, toy_templates, mode, device):
    train, val = masks(toy_graph)
    model, features, extractor = build(toy_graph, toy_templates, mode, device)
    samplers = HeteroNeighborSampler(toy_graph, [4], seed=0), HeteroNeighborSampler(
        toy_graph, [4], seed=1)
    result = fit_sampled_graph(model, toy_graph, features, extractor, *samplers, train, val,
                               epochs=1, patience=5, batch_size=2, seed=0)
    assert result["epochs_completed"] == 1
    row = result["history"][0]
    assert row["batches"] == 2
    assert 0 <= row["sampling_s"] <= row["train_runtime_s"] <= row["epoch_runtime_s"]
    assert row["val_runtime_s"] >= 0
    assert {x.device.type for x in features.values()} == {"cpu"}
    assert {e.device.type for e in toy_graph.edge_index_dict.values()} == {"cpu"}


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path, toy_graph, toy_templates):
    train, val = masks(toy_graph)
    config = {"run": "toy"}

    model, features, extractor = build(toy_graph, toy_templates, "auto_phgt")
    full = fit_full_graph(model, toy_graph, features, extractor, train, val,
                          epochs=3, patience=10, seed=0)

    path = tmp_path / "ckpt.pt"
    model, features, extractor = build(toy_graph, toy_templates, "auto_phgt")
    first = fit_full_graph(model, toy_graph, features, extractor, train, val, epochs=1,
                           patience=10, seed=0, checkpoint_path=path, config=config)
    assert first["resumed_from_epoch"] is None
    saved = torch.load(path, weights_only=False)
    assert {"model", "optimizer", "epoch", "best_epoch", "best_val_accuracy", "config",
            "seed", "rng"} <= saved.keys()
    assert saved["epoch"] == 1 and saved["seed"] == 0 and saved["config"] == config

    model, features, extractor = build(toy_graph, toy_templates, "auto_phgt")
    resumed = fit_full_graph(model, toy_graph, features, extractor, train, val, epochs=3,
                             patience=10, seed=0, checkpoint_path=path, config=config)
    assert resumed["resumed_from_epoch"] == 1
    assert [r["epoch"] for r in resumed["history"]] == [1, 2, 3]
    assert [r["loss"] for r in resumed["history"]] == pytest.approx(
        [r["loss"] for r in full["history"]])
    assert resumed["best_epoch"] == full["best_epoch"]


def test_checkpoint_rejects_other_config_and_skips_finished_run(tmp_path, toy_graph,
                                                                toy_templates):
    train, val = masks(toy_graph)
    path = tmp_path / "ckpt.pt"
    model, features, extractor = build(toy_graph, toy_templates, "hgt")
    fit_full_graph(model, toy_graph, features, extractor, train, val, epochs=2, patience=1,
                   seed=0, checkpoint_path=path, config={"lr": 1})
    with pytest.raises(ValueError, match="different configuration"):
        load_checkpoint(path, {"lr": 2})
    saved = load_checkpoint(path, {"lr": 1})
    saved["stopped"] = True
    save_checkpoint(path, saved)
    model, features, extractor = build(toy_graph, toy_templates, "hgt")
    again = fit_full_graph(model, toy_graph, features, extractor, train, val, epochs=5,
                           patience=1, seed=0, checkpoint_path=path, config={"lr": 1})
    assert again["epochs_completed"] == saved["epoch"] and again["early_stopped"]


def test_result_record_is_reproducible_json(tmp_path, toy_graph, toy_templates):
    train, val = masks(toy_graph)
    model, features, extractor = build(toy_graph, toy_templates, "auto_phgt")
    fit = fit_full_graph(model, toy_graph, features, extractor, train, val, epochs=2,
                         patience=5, seed=0)
    discovery = discovery_record(toy_graph, extractor.templates, dataset="toy",
                                 target_node_type="paper", max_hops=4, k=3)
    result = build_result(dataset="toy", mode="auto_phgt", seed=0,
                          hyperparameters={"d_model": 16}, discovery=discovery,
                          device=torch.device("cpu"), model=model, fit=fit,
                          test={"accuracy": 0.5, "macro_f1": 0.4}, checkpoint=None,
                          started_at="now", setup={"load_s": 1.0})
    loaded = json.loads(write_json(tmp_path / "r.json", result).read_text())
    for key in ("dataset", "mode", "seed", "hyperparameters", "discovery", "environment",
                "parameters", "best_epoch", "validation", "test", "training_runtime_s",
                "setup_runtime_s", "peak_cpu_rss_mib"):
        assert key in loaded
    assert [p["path"] for p in loaded["discovery"]["paths"]] == toy_templates
    assert all(p["score"] > 0 for p in loaded["discovery"]["paths"])
    assert loaded["environment"]["device"] == "cpu"
    assert set(loaded["environment"]["git"]) == {"commit", "dirty"}
    assert loaded["validation"]["macro_f1"] is not None


def test_git_state_outside_repository(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampled", [False, True])
def test_smoke_modes(toy_graph, toy_templates, device, sampled):
    features = impute_missing_features(toy_graph)
    sampler = HeteroNeighborSampler(toy_graph, [2], seed=0) if sampled else None
    out = smoke_modes(toy_graph, toy_templates, features, torch.tensor([0, 1, 2]), device,
                      sampler=sampler)
    assert set(out["modes"]) == set(MODES)
    for mode in out["modes"].values():
        assert mode["logits_shape"] == [3, 3] and mode["backward_ok"] and mode["loss_finite"]
        assert mode["logits_device"].startswith(device)


def test_progress_lines(capsys, toy_graph, toy_templates):
    train, val = masks(toy_graph)
    model, features, extractor = build(toy_graph, toy_templates, "tokens")
    fit_full_graph(model, toy_graph, features, extractor, train, val, epochs=2, patience=5,
                   seed=0, progress=True)
    lines = [json.loads(line.removeprefix("[progress] "))
             for line in capsys.readouterr().out.splitlines() if line.startswith("[progress]")]
    assert [line["epoch"] for line in lines] == [1, 2]
    assert all("val_accuracy" in line and "best_epoch" in line for line in lines)
