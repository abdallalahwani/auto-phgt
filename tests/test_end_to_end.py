import json

import pytest
import torch

from experiments import run_acm, run_mag

SMALL = {"d_model": 16, "k": 2, "max_hops": 4, "instances_per_path": 2, "hgt_layers": 1,
         "hgt_heads": 2, "fusion_layers": 1, "fusion_heads": 2, "ffn_mult": 2, "dropout": 0.1,
         "pooling": "mean", "token_dropout": 0.1, "weight_decay": 1e-3, "epochs": 2,
         "patience": 5}


@pytest.mark.parametrize("mode", ["hgt", "auto_phgt"])
@pytest.mark.parametrize("selection", ["discovered", "random"])
def test_run_acm_end_to_end_and_resume(tmp_path, monkeypatch, toy_graph, mode, selection):
    toy_graph["paper"].train_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    toy_graph["paper"].test_mask = ~toy_graph["paper"].train_mask
    monkeypatch.setattr(run_acm, "load_acm", lambda root: toy_graph)
    monkeypatch.setattr(run_acm, "protocol_hyperparameters", lambda dataset: {
        **SMALL, "lr": 2e-3, "acm_validation_size": 1, "training": "full_graph"})
    kwargs = dict(mode=mode, seed=0, path_selection=selection, device="cpu",
                  output=tmp_path / "r.json", checkpoint=tmp_path / "c.pt",
                  template_dir=tmp_path / "discovery", extra={"experiment_id": "x"})
    first = run_acm.run(**kwargs)
    record = json.loads((tmp_path / "r.json").read_text())
    assert record["status"] == "completed" and record["experiment_id"] == "x"
    assert record["path_selection"] == selection and record["k"] == 2
    assert len(record["discovery"]["paths"]) == 2
    assert record["split"]["train"] == 4 and record["split"]["val"] == 1
    assert (record["path_statistics"] is None) == (mode == "hgt")
    assert record["total_runtime_s"] > 0 and record["history"]
    assert (tmp_path / "c.progress.json").exists()
    again = run_acm.run(**kwargs)
    assert again["resumed_from_epoch"] == 2
    assert {k: again["test"][k] for k in ("accuracy", "macro_f1")} == {
        k: first["test"][k] for k in ("accuracy", "macro_f1")}


@pytest.mark.parametrize("mode", ["hgt", "auto_phgt"])
def test_run_mag_end_to_end_keeps_graph_on_cpu(tmp_path, monkeypatch, toy_graph, mode):
    toy_graph["paper"].train_mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
    toy_graph["paper"].val_mask = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
    toy_graph["paper"].test_mask = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
    monkeypatch.setattr(run_mag, "load_mag", lambda root: toy_graph)
    monkeypatch.setattr(run_mag, "protocol_hyperparameters", lambda dataset: {
        **SMALL, "lr": 1e-3, "batch_size": 2, "num_neighbors": [2, 2],
        "training": "sampled_subgraph"})
    record = run_mag.run(mode=mode, seed=1, device="cpu", output=tmp_path / "r.json",
                         checkpoint=tmp_path / "c.pt", template_dir=tmp_path / "d",
                         extra={"experiment_id": "y"})
    assert record["cpu_resident"] == {"features": ["cpu"], "edge_index": ["cpu"]}
    assert len(record["sampled_batch_statistics"]) == 3
    assert record["split"] == {**record["split"], "train": 3, "val": 2, "test": 1}
    assert {"batches", "sampling_s", "train_runtime_s"} <= record["history"][0].keys()
    assert json.loads((tmp_path / "r.json").read_text())["experiment_id"] == "y"
