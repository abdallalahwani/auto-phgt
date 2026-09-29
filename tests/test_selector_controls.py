import json
import subprocess
import sys

import pytest
import torch

from auto_phgt.discovery import StatisticalDiscoveryModule
from auto_phgt.model import AutoPHGT
from auto_phgt.selection import (diverse_top_k, homophily_table, homophily_top_k,
                                 hybrid_top_k, path_families, relation_family)
from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features
from experiments import baseline_suite as engine
from experiments import run_mag, selector_controls
from experiments.common import select_templates

PTP = ["paper", "to", "term", "to", "paper"]
PAP = ["paper", "to", "author", "to", "paper"]
PCP_TP = ["paper", "cite", "paper", "to", "term", "to", "paper"]


def ranking_and_scores(graph):
    engine_ = StatisticalDiscoveryModule(graph, max_hops=4)
    scores = engine_.calculate_path_frequencies(engine_.discover_candidate_paths())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [list(p) for p, _ in ranked], scores


def test_relation_families():
    assert relation_family("paper", "to", "term") == relation_family("term", "to", "paper")
    assert relation_family("paper", "cite", "paper") == "paper-paper:cite"
    assert path_families(PCP_TP) == {"paper-paper:cite", "paper-term"}


def test_diverse_selection_covers_new_families_then_fills_by_score():
    ranking = [PTP, PTP + ["to", "term", "to", "paper"], PCP_TP, PAP]
    assert diverse_top_k(ranking, 3) == [PTP, PCP_TP, PAP]
    assert diverse_top_k(ranking, 4) == [PTP, PCP_TP, PAP, ranking[1]]


def test_homophily_uses_training_labels_only(toy_graph):
    ranked, _ = ranking_and_scores(toy_graph)
    train = torch.tensor([0, 1, 2, 3])
    labels = toy_graph["paper"].y.clone()
    table = homophily_table(toy_graph, ranked, train, labels, samples_per_node=32, min_support=1)
    changed = labels.clone()
    changed[4:] = (changed[4:] + 1) % 3          # non-training labels must not matter
    again = homophily_table(toy_graph, ranked, train, changed, samples_per_node=32,
                            min_support=1)
    assert [r["homophily"] for r in table["paths"]] == [r["homophily"] for r in again["paths"]]
    shares = torch.bincount(labels[train]).float() / 4
    assert table["chance"] == pytest.approx(float((shares ** 2).sum()))
    for row in table["paths"]:
        assert row["support"] >= 0 and 0 <= row["lift"] <= 1
    strict = homophily_table(toy_graph, ranked, train, labels, min_support=10**6)
    assert all(r["homophily"] is None and r["lift"] == 0 for r in strict["paths"])


def test_homophily_and_hybrid_ordering():
    ranking = [PTP, PCP_TP, PAP]
    table = {"paths": [{"path": PTP, "homophily": 0.4, "lift": 0.05},
                       {"path": PCP_TP, "homophily": None, "lift": 0.0},
                       {"path": PAP, "homophily": 0.9, "lift": 0.55}]}
    assert homophily_top_k(ranking, table, 2) == [PAP, PTP]
    scores = {tuple(PTP): 100.0, tuple(PCP_TP): 50.0, tuple(PAP): 5.0}
    assert hybrid_top_k(ranking, scores, table, 3) == [PTP, PAP, PCP_TP]
    table["paths"][0]["homophily"] = 0.0
    assert homophily_top_k(ranking, table, 3) == [PAP, PTP, PCP_TP]


@pytest.mark.parametrize("selection,extra", [
    ("diverse", {}), ("homophily", {}), ("hybrid", {}), ("fixed", {"paths": [PAP, PTP]}),
    ("rank", {"ranks": [3, 1]}), ("random", {"path_seed": 1234})])
def test_select_templates_v2(tmp_path, toy_graph, selection, extra):
    ranked, _ = ranking_and_scores(toy_graph)
    templates, record = select_templates(
        toy_graph, dataset="toy", selection=selection, k=2, max_hops=4, seed=0,
        template_dir=tmp_path, train_ids=torch.tensor([0, 1, 2, 3]),
        labels=toy_graph["paper"].y, **extra)
    schemas = [list(t.schema) for t in templates]
    assert len(schemas) == 2 and len({tuple(s) for s in schemas}) == 2
    assert all(s in ranked for s in schemas)
    assert record["homophily"] is not None and len(record["homophily"]["paths"]) == len(ranked)
    if selection == "rank":
        assert schemas == [ranked[2], ranked[0]]
    if selection == "random":
        assert record["random_seed"] == 1234


def test_select_templates_rejects_bad_fixed_paths(tmp_path, toy_graph):
    with pytest.raises(ValueError, match="distinct candidate paths"):
        select_templates(toy_graph, dataset="toy", selection="fixed", k=2, max_hops=4, seed=0,
                         template_dir=tmp_path, paths=[PAP, PAP])
    with pytest.raises(ValueError, match="needs training ids"):
        select_templates(toy_graph, dataset="toy", selection="homophily", k=2, max_hops=4,
                         seed=0, template_dir=tmp_path)


def build(graph, templates, control, instances=2):
    torch.manual_seed(0)
    x = impute_missing_features(graph)
    model = AutoPHGT.build(graph, templates, 3, x_dict=x, d_model=16, hgt_layers=1, hgt_heads=2,
                           fusion_layers=1, fusion_heads=2, dropout=0.0, token_control=control,
                           num_dummy_tokens=len(templates) * instances)
    extractor = MetaPathInstanceExtractor(graph, templates, instances_per_path=instances)
    return model, x, extractor


def test_dummy_tokens_ignore_path_content(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates, "dummy")
    ids = torch.arange(6)
    model.eval()
    first = model(x, toy_graph.edge_index_dict, instances=ex.sample(ids, seed=1))
    second = model(x, toy_graph.edge_index_dict, instances=ex.sample(ids, seed=2))
    assert torch.allclose(first, second)
    model.train()
    logits = model(x, toy_graph.edge_index_dict, instances=ex.sample(ids))
    torch.nn.functional.cross_entropy(logits, toy_graph["paper"].y).backward()
    assert model.dummy_tokens.grad is not None and model.dummy_tokens.grad.abs().sum() > 0
    assert model.num_parameters()["dummy tokens"] == len(toy_templates) * 2 * 16


def test_shuffled_tokens_are_permuted_deterministically_in_eval(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates, "shuffle")
    base, _, _ = build(toy_graph, toy_templates, None)
    base.load_state_dict(model.state_dict())
    model.eval(); base.eval()
    inst = ex.sample(torch.arange(6), seed=1)
    out1 = model(x, toy_graph.edge_index_dict, instances=inst)
    out2 = model(x, toy_graph.edge_index_dict, instances=inst)
    assert torch.equal(out1, out2)
    tokens = base.tokenizer(x, inst)
    shuffled = model._apply_token_control(tokens)
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(1))
    assert torch.equal(shuffled.semantic_tokens, tokens.semantic_tokens[perm])
    assert torch.equal(shuffled.semantic_mask, tokens.semantic_mask[perm])


def test_token_control_validation(toy_graph, toy_templates):
    with pytest.raises(ValueError, match="mode 'hgt'"):
        AutoPHGT.build(toy_graph, toy_templates, 3, d_model=16, hgt_heads=2, mode="hgt",
                       token_control="dummy", num_dummy_tokens=6)
    with pytest.raises(ValueError, match="num_dummy_tokens"):
        AutoPHGT.build(toy_graph, toy_templates, 3, d_model=16, hgt_heads=2,
                       token_control="dummy")
    model, x, ex = build(toy_graph, toy_templates, "dummy", instances=2)
    with pytest.raises(ValueError, match="dummy tokens"):
        model(x, toy_graph.edge_index_dict, instances=MetaPathInstanceExtractor(
            toy_graph, toy_templates, instances_per_path=3).sample(torch.arange(6)))


def test_v2_inventory():
    specs = selector_controls.inventory()
    parts = {}
    for spec in specs:
        parts.setdefault(spec["part"], []).append(spec)
    assert len(parts["mag"]) == 12 and len(parts["acm"]) == 734
    groups = {}
    for spec in parts["acm"]:
        groups[spec["group"]] = groups.get(spec["group"], 0) + 1
    assert groups == {"acm_selectors_k5": 50, "acm_controls": 60, "acm_selectors_k2": 60,
                      "acm_random_sets": 300, "acm_single_path": 264}
    for part in parts.values():
        assert [s["priority"] for s in part] == sorted(s["priority"] for s in part)
    assert all(s["checkpoint"] is None for s in parts["acm"])
    assert all(s["epochs"] == 300 and s["checkpoint"] for s in parts["mag"])
    continued = [s for s in parts["mag"] if s.get("continue_from")]
    assert len(continued) == 6
    assert {s["path_seed"] for s in parts["acm"] if s["group"] == "acm_random_sets"} == set(
        range(1000, 1100))
    assert sorted({tuple(s["ranks"]) for s in parts["acm"] if s["group"] == "acm_single_path"}
                  ) == [(r,) for r in range(1, 89)]
    assert all(not s["id"].startswith(("acm_", "mag_")) for s in specs)  # never v1 ids


def test_worker_kwargs():
    specs = {s["id"]: s for s in selector_controls.inventory()}
    acm = selector_controls.worker_kwargs(specs["v2_acm_hgt_d68_seed3_discovered_k5"])
    assert acm["no_checkpoint"] and acm["d_model"] == 68 and "checkpoint" not in acm
    mag = selector_controls.worker_kwargs(specs["v2_mag_auto_phgt_dummy_seed1_discovered_k5_e300"])
    assert mag["epochs"] == 300 and mag["token_control"] == "dummy" and mag["checkpoint"]
    manual = selector_controls.worker_kwargs(specs["v2_acm_auto_phgt_seed0_manual_k2"])
    assert manual["path_selection"] == "fixed" and manual["paths"] == selector_controls.MANUAL_ACM


def test_v2_freeze(tmp_path):
    specs = selector_controls.inventory()
    path = tmp_path / "v2.json"
    first = selector_controls.freeze(specs, path)
    assert selector_controls.freeze(specs, path)["frozen_at"] == first["frozen_at"]
    assert first["hyperparameters"]["ogbn-mag"]["epochs"] == 300
    assert first["hyperparameters"]["acm"]["epochs"] == 100
    record = json.loads(path.read_text())
    record["protocol"]["selectors"]["diverse"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="differs"):
        selector_controls.freeze(specs, path)


def test_mag_continuation_resumes_with_only_epochs_raised(tmp_path, monkeypatch, toy_graph):
    toy_graph["paper"].train_mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
    toy_graph["paper"].val_mask = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
    toy_graph["paper"].test_mask = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
    monkeypatch.setattr(run_mag, "load_mag", lambda root: toy_graph)
    hp = {"d_model": 16, "k": 2, "max_hops": 4, "instances_per_path": 2, "hgt_layers": 1,
          "hgt_heads": 2, "fusion_layers": 1, "fusion_heads": 2, "ffn_mult": 2, "dropout": 0.1,
          "pooling": "mean", "token_dropout": 0.1, "weight_decay": 1e-3, "epochs": 2,
          "patience": 50, "lr": 1e-3, "batch_size": 2, "num_neighbors": [2, 2],
          "training": "sampled_subgraph"}
    monkeypatch.setattr(run_mag, "protocol_hyperparameters", lambda dataset: dict(hp))
    v1 = tmp_path / "v1.pt"
    run_mag.run(mode="auto_phgt", seed=0, device="cpu", output=tmp_path / "v1.json",
                checkpoint=v1, template_dir=tmp_path / "d")
    spec = {"continue_from": str(v1), "checkpoint": str(tmp_path / "v2.pt"), "epochs": 4}
    selector_controls.prepare_continuation(spec)
    result = run_mag.run(mode="auto_phgt", seed=0, device="cpu", epochs=4,
                         output=tmp_path / "v2.json", checkpoint=spec["checkpoint"],
                         template_dir=tmp_path / "d")
    assert result["resumed_from_epoch"] == 2 and result["epochs_completed"] == 4
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3, 4]
    assert torch.load(v1, weights_only=False)["config"]["hyperparameters"]["epochs"] == 2


def test_scheduler_accepts_runs_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = {"id": "x", "threads": 1, "checkpoint": None, "log": str(tmp_path / "x.log"),
            "result": str(tmp_path / "x.json")}
    code = (f"import json; open({spec['result']!r}, 'w').write(json.dumps("
            f"{{'status': 'completed', 'experiment_id': 'x'}}))")
    outcomes = engine.Suite([spec], ["0", "0"], poll_s=0.05, state_path=tmp_path / "s.json",
                            launch=lambda s, w, g: subprocess.Popen([sys.executable, "-c",
                                                                     code])).run()
    assert outcomes == {"x": "completed"}
