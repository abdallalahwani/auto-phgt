import json
import subprocess
import sys
import time

import pytest
import torch

from experiments import residual_campaign as master
from experiments.residual_selection import aggregate, data, plan, rcms, tasks
from experiments.residual_selection.train import build_model, evaluate_test, extractor_for, fit, predict

CFG = {"feat": "given", "layers": 1, "lr": 1e-2, "wd": 1e-4, "dropout": 0.0,
       "max_epochs": 6, "patience": 3, "heads": 2, "d_model": 16}
PTP = ["paper", "to", "term", "to", "paper"]
PAP = ["paper", "to", "author", "to", "paper"]


def with_masks(g):
    g["paper"].train_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    g["paper"].test_mask = ~g["paper"].train_mask
    return g


# --------------------------------------------------------------------------- plan
def test_plan_conditions_are_consistent():
    conds = plan.hgb_conditions()
    ids = [c["id"] for c in conds]
    assert len(ids) == len(set(ids))
    dblp = [c for c in conds if c["dataset"] == "dblp" and c["k"] == 5]
    assert all(c["lmax"] == 6 for c in dblp)  # only 4 candidates within 4 hops
    assert plan.search_lmax("dblp") == [4, 6] and plan.search_lmax("acm") == [4, 6]
    for ds in plan.HGB_DATASETS:
        ref = plan.DATASETS[ds]["hgb_reference"]
        assert ref["feat"] in plan.FEATURE_REGIMES[ds] and ref["layers"] in plan.GRID["layers"]
        assert ref["lr"] in plan.GRID["lr"] and ref["wd"] in plan.GRID["wd"][ds]
        assert ref["dropout"] in plan.GRID["dropout"]
    assert len(plan.protocol_hash()) == 16


def test_task_graph_is_a_valid_dag():
    graph = master.build_tasks()
    ids = {t["id"] for t in graph}
    assert len(ids) == len(graph)
    order = {t["id"]: i for i, t in enumerate(graph)}
    for t in graph:
        assert all(d in ids for d in t["deps"])
    finals = [t for t in graph if t["fn"] == "final" and "rcms" in t["args"]["cond_id"]]
    assert all(any("__search__" in d for d in t["deps"]) for t in finals)
    rcms_mag = [t for t in graph if t["fn"] == "mag_final" and t["args"]["name"] == "rcms_k5"]
    assert all("mag__search" in t["deps"] for t in rcms_mag)
    ext = [t for t in graph if t["gate"] == "freebase_extension"]
    assert ext and all("freebase__decision" in t["deps"] for t in ext)
    assert not any(t["fn"] == "final" and t["args"]["seed"] >= 5 and not t["gate"] for t in graph)
    assert order  # built without error


# --------------------------------------------------------------------------- data
def test_split_and_feature_regimes(toy_graph):
    g = with_masks(toy_graph)
    train, val, test = data.split(g, "paper", seed=3)
    assert val.numel() == int(5 * 0.2) and train.numel() == 4 and test.tolist() == [5]
    assert data.split(g, "paper", seed=3)[1].tolist() == val.tolist()
    assert set(data.features(g, "paper", "given")) == {"paper", "author"}
    assert set(data.features(g, "paper", "target_onehot")) == {"paper"}
    zero = data.features(g, "paper", "target_zero")
    assert zero["author"].abs().sum() == 0 and zero["term"].shape == (3, 10)
    assert set(data.features(g, "paper", "imputed")) == {"paper", "author", "term"}
    rep = data.label_free_repr(g, data.features(g, "paper", "target_zero"), d=4, seed=0)
    assert rep["author"].abs().sum() == 0 and rep["paper"].shape == (6, 4)
    rep2 = data.label_free_repr(g, data.features(g, "paper", "given"), d=4, seed=0)
    assert torch.equal(rep2["term"], data.label_free_repr(g, data.features(g, "paper", "given"),
                                                          d=4, seed=0)["term"])


# --------------------------------------------------------------------------- rcms components
def test_path_reprs_and_screen(toy_graph):
    g = with_masks(toy_graph)
    rep = data.label_free_repr(g, data.features(g, "paper", "given"), d=4, seed=0)
    ids = torch.arange(6)
    z = rcms.path_reprs(g, [PTP, PAP], rep, ids, seed=0, instances=4)
    assert z.shape == (2, 6, 5)
    assert z[0, 5, -1] == 0 and torch.all(z[0, 5, :-1] == 0)  # paper 5 has no terms
    assert z[1, :, -1].min() > 0
    kept, dropped = rcms.screen(g, [PTP, PAP], ids, seed=0)
    assert [s for s, _ in kept] == [PTP, PAP] and dropped == []


def synthetic(n=600, seed=0):
    """A: noisy indicator of class 0; B: near-duplicate of A; C: indicator of class 1; N: noise."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 3, (n,), generator=g)

    def ind(c):
        return (((y == c).float()[:, None] + 0.3 * torch.randn(n, 1, generator=g)).repeat(1, 3)
                + 0.5 * torch.randn(n, 3, generator=g))

    a = ind(0)
    z = torch.stack([a, a + 0.05 * torch.randn(n, 3, generator=g), ind(1), torch.randn(n, 3, generator=g)])
    return y, rcms.standardize(z, dim=1), rcms.stratified_folds(y, 3, 0)


def test_probe_utility_detects_signal_and_redundancy():
    y, z, folds = synthetic()
    dev = torch.device("cpu")
    util = rcms.probe_ce(None, None, y, folds, 3, dev)[0] - rcms.probe_ce(None, z, y, folds, 3, dev)
    assert min(util[:3]) > 0.2 and abs(float(util[3])) < 0.05
    chosen, steps, first = rcms.greedy(None, z, y, folds, 3, 3, dev, use_h=False)
    assert chosen[:2] == [2, 0]  # strongest signal, then the complementary one (not A's duplicate)
    assert steps[1]["utility"] > 0.2 and steps[2]["utility"] < 0.02  # B adds nothing given A
    assert set(first) == {0, 1, 2, 3}
    sel = rcms.selections(steps, first, steps, [1, 2], 4)
    assert sel["rcms_indep"]["2"] == [2, 0] and sel["rcms"]["2"] == [2, 0]


def test_backbone_conditioning_removes_explained_signal():
    y, z, folds = synthetic()
    h = torch.nn.functional.one_hot(y, 3).float() * 3  # H already explains everything
    _, _, first = rcms.greedy(rcms.standardize(h, 0), z, y, folds, 3, 1, torch.device("cpu"))
    assert max(first.values()) < 0.05


def test_stratified_folds_are_balanced():
    y = torch.tensor([0] * 30 + [1] * 9 + [2] * 3)
    folds = rcms.stratified_folds(y, 3, 0)
    for c in range(3):
        counts = torch.bincount(folds[y == c], minlength=3)
        assert counts.max() - counts.min() <= 1


def test_residual_cka_is_bounded():
    y, z, _ = synthetic()
    out = rcms.residual_cka(None, torch.cat([z, torch.ones(4, z.size(1), 1)], 2),
                            {"dup": [0, 1], "indep": [0, 3]})
    assert out["dup"]["mean_pairwise_cka"] > 0.9 and out["indep"]["mean_pairwise_cka"] < 0.2
    h = rcms.standardize(z[0].clone(), 0)  # residualising on A's own content removes the overlap
    assert rcms.residual_cka(h, torch.cat([z, torch.ones(4, z.size(1), 1)], 2),
                             {"dup": [0, 1]})["dup"]["mean_pairwise_cka"] < 0.5


# --------------------------------------------------------------------------- train
@pytest.mark.parametrize("mode", ["hgt", "auto_phgt"])
def test_fit_and_single_test_evaluation(toy_graph, toy_templates, mode):
    g = with_masks(toy_graph)
    train, val, test = data.split(g, "paper", 0)
    x = data.features(g, "paper", "given")
    model = build_model(g, toy_templates, 3, x, CFG, mode)
    ex = extractor_for(g, toy_templates, 0) if mode != "hgt" else None
    res = fit(model, g, x, ex, train, val, CFG, 0)
    assert 1 <= res["best_epoch"] <= 6 and res["epochs_completed"] <= 6
    assert set(res["val"]) == {"micro_f1", "macro_f1"}
    assert predict(model, g, x, ex, test).shape == (1, 3)
    assert evaluate_test(model, g, x, ex, test)["n"] == 1


# --------------------------------------------------------------------------- tasks end to end
@pytest.fixture
def fake_campaign(tmp_path, monkeypatch, toy_graph):
    g = with_masks(toy_graph)
    monkeypatch.setattr(tasks, "V3", tmp_path)
    monkeypatch.setattr(data, "load", lambda ds: g)
    monkeypatch.setattr(data, "target", lambda ds: "paper")
    monkeypatch.setattr(tasks, "chosen_config", lambda ds: dict(CFG))
    monkeypatch.setitem(tasks.RCMS, "probe_steps", 30)
    monkeypatch.setitem(tasks.RCMS, "coverage_probe_nodes", 6)
    return tmp_path


def test_search_and_final_runs_end_to_end(fake_campaign):
    record = tasks.search("acm", 0, 4)
    rec = json.loads((fake_campaign / "searches" / "acm_L4_seed0.json").read_text())
    assert rec["candidates"]["kept"] >= 2 and len(rec["steps_full"]) == min(5, rec["candidates"]["kept"])
    assert set(rec["selected"]) == {"rcms", "rcms_indep", "rcms_nohgt"}
    assert all(len(v["2"]) == 2 for v in rec["selected"].values())
    assert len(rec["fold_hgt"]) == 3 and record["kept"] == rec["candidates"]["kept"]
    for cond in ("acm__rcms_k2", "acm__hybrid_k2", "acm__hgt_strong", "acm__rcms_k5_dummy"):
        tasks.final(cond, 0)
        out = json.loads((fake_campaign / "results" / "acm" / f"{cond}__seed0.json").read_text())
        assert out["status"] == "completed" and "micro_f1" in out["test"]
        assert out["validation"]["micro_f1"] is not None
    events = (fake_campaign / "logs" / "test_evaluations.jsonl").read_text().splitlines()
    assert len(events) == 4
    out = json.loads((fake_campaign / "results" / "acm" / "acm__rcms_k2__seed0.json").read_text())
    assert out["templates"] == rec["selected"]["rcms"]["2"]


def test_freebase_decision_uses_validation_only(fake_campaign):
    for cond in plan.hgb_conditions():
        if cond["dataset"] == "freebase" and cond["priority"] == 0:
            for seed in range(5):
                path = tasks.result_path(cond["id"], seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"validation": {"micro_f1": 0.5 + 0.02 * seed},
                                            "test": {"micro_f1": "NEVER READ"}}))
    decision = tasks.freebase_decision()
    assert decision["extend"] is True and tasks.freebase_extension_enabled()


# --------------------------------------------------------------------------- scheduler
def fake_tasks():
    return [master.task("a", "x", {}, size="small"), master.task("b", "x", {}, deps=["a"]),
            master.task("c", "x", {}, deps=["a"], size="medium"),
            master.task("fail", "x", {}, size="small"),
            master.task("after_fail", "x", {}, deps=["fail"]),
            master.task("p1", "x", {}, priority=1, size="small"),
            master.task("big", "x", {}, size="large"), master.task("cpu", "x", {}, size="cpu")]


def test_master_schedules_dependencies_retries_and_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "V3", tmp_path)
    trace = tmp_path / "trace.jsonl"
    launched = []

    def launch(spec, gpu):
        launched.append(spec["id"])
        status = master.status_path(spec["id"])
        code = 1 if spec["id"] == "fail" else 0
        script = (f"import json, time, pathlib; t0 = time.time(); time.sleep(0.2); "
                  f"open({str(trace)!r}, 'a').write(json.dumps({{'id': {spec['id']!r}, 'gpu': {gpu!r}, "
                  f"'size': {spec['size']!r}, 't0': t0, 't1': time.time()}}) + '\\n'); ")
        if code == 0:
            script += (f"p = pathlib.Path({str(status)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
                       f"p.write_text(json.dumps({{'status': 'done'}}))")
        else:
            script += "raise SystemExit(1)"
        return subprocess.Popen([sys.executable, "-c", script])

    m = master.Master(fake_tasks(), ["3", "5"], cpu_budget=7, deadline=None, launch=launch, poll=0.05)
    m.run(aggregate_every=float("inf"))
    assert m.state["fail"] == "failed" and launched.count("fail") == 2
    assert m.state["after_fail"] == "dep_failed"
    assert all(m.state[t] == "done" for t in ("a", "b", "c", "p1", "big", "cpu"))
    spans = [json.loads(line) for line in trace.read_text().splitlines()]
    starts = {s["id"]: s["t0"] for s in spans}
    ends = {s["id"]: s["t1"] for s in spans}
    assert starts["b"] >= ends["a"] and starts["c"] >= ends["a"]
    for s in spans:
        live = [o for o in spans if o["t0"] < s["t1"] and s["t0"] < o["t1"] and o["gpu"] == s["gpu"]
                and o["gpu"] is not None]
        assert sum(master.UNITS[o["size"]] for o in live) <= master.GPU_UNITS
    p0_starts = [starts[t] for t in ("a", "b", "c", "big", "cpu") if t in starts]
    assert starts["p1"] >= min(p0_starts)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert {"task_id", "status", "config_hash", "gpu"} <= manifest[0].keys()


def test_master_respects_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "V3", tmp_path)
    m = master.Master([master.task("long", "x", {}, est=600)], ["0"], 4, deadline=time.time() + 60,
                      launch=lambda s, g: pytest.fail("must not launch"), poll=0.01)
    m.run(aggregate_every=float("inf"))
    assert m.state["long"] == "pending"


def test_gate_skips_without_launching_and_then_runs_p1(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "V3", tmp_path)
    monkeypatch.setattr(master, "gate_open", lambda gate: False)
    launched = []

    def launch(spec, gpu):
        launched.append(spec["id"])
        status = master.status_path(spec["id"])
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({"status": "done"}))
        return subprocess.Popen([sys.executable, "-c", "pass"])

    m = master.Master([master.task("g", "x", {}, gate="freebase_extension"),
                       master.task("p1", "x", {}, priority=1)], ["0"], 4, None,
                      launch=launch, poll=0.01)
    m.run(aggregate_every=float("inf"))
    assert m.state["g"] == "skipped" and launched == ["p1"] and m.state["p1"] == "done"


# --------------------------------------------------------------------------- statistics
def rec(v):
    return {"dataset": "acm", "test": {"micro_f1": v, "macro_f1": v}}


def test_paired_statistics():
    a = {s: rec(0.9 + 0.001 * s) for s in range(5)}
    b = {s: rec(0.88) for s in range(5)}
    comp = aggregate.paired(a, b)
    assert comp["n"] == 5 and comp["wins"] == 5 and comp["ci95"][0] > 0
    assert comp["p_perm"] == pytest.approx(2 / 32)
    assert aggregate.verdict(comp) == "supported"
    assert aggregate.verdict(aggregate.paired(b, a)) == "contradicted"
    mixed = aggregate.paired({s: rec(0.9 + (0.01 if s % 2 else -0.01)) for s in range(4)},
                             {s: rec(0.9) for s in range(4)})
    assert aggregate.verdict(mixed) == "inconclusive"
    assert aggregate.abbr(["paper", "cite", "paper", "to", "term", "to", "paper"]) == "P-cite-P-T-P"


def test_smoke_settings_are_isolated():
    out = subprocess.run([sys.executable, "-c",
                          "from experiments.residual_selection import plan; print(plan.V3, plan.MAG['epochs'], "
                          "plan.SEEDS['acm'])"], env={**__import__("os").environ, "V3_SMOKE": "1"},
                         capture_output=True, text=True, check=True)
    assert out.stdout.split()[0] == "artifacts/v3_smoke" and out.stdout.split()[1] == "1"
    assert str(plan.V3) == "artifacts/v3"


def mag_like_graph():
    from tests.test_sampler_equivalence import random_hetero_graph
    g = random_hetero_graph(seed=3)
    n = g["paper"].num_nodes
    gen = torch.Generator().manual_seed(0)
    g["paper"].x = torch.randn(n, 8, generator=gen)
    g["paper"].y = torch.randint(0, 4, (n,), generator=gen)
    order = torch.randperm(n, generator=gen)
    for name, part in (("train_mask", order[:2000]), ("val_mask", order[2000:2500]),
                       ("test_mask", order[2500:])):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[part] = True
        g["paper"][name] = mask
    return g


def test_mag_search_and_final_glue(tmp_path, monkeypatch):
    import auto_phgt.data
    from experiments import protocol, run_mag
    g = mag_like_graph()
    monkeypatch.setattr(tasks, "V3", tmp_path)
    monkeypatch.setattr(auto_phgt.data, "load_mag", lambda root: g)
    monkeypatch.setattr(run_mag, "load_mag", lambda root: g)
    small = {**protocol.hyperparameters("ogbn-mag"), "d_model": 16, "hgt_heads": 2, "fusion_heads": 2,
             "batch_size": 256, "num_neighbors": [3, 2], "instances_per_path": 2}
    monkeypatch.setattr(protocol, "hyperparameters", lambda ds: dict(small))
    monkeypatch.setattr(run_mag, "protocol_hyperparameters", lambda ds: dict(small))
    monkeypatch.setitem(tasks.MAG, "search_epochs", 1)
    monkeypatch.setitem(tasks.MAG, "search_subset", 300)
    monkeypatch.setitem(tasks.RCMS, "probe_steps", 20)
    out = tasks.mag_search(0)
    rec = json.loads((tmp_path / "searches" / "ogbn-mag_L4_seed0.json").read_text())
    assert len(out["rcms_k5"]) == min(5, rec["candidates"]["kept"]) and rec["search_nodes"] <= 300
    real_run = run_mag.run
    monkeypatch.setattr(run_mag, "run", lambda **kw: real_run(**{**kw, "device": "cpu"}))
    res = tasks.mag_final("rcms_k5", 0, epochs=1)
    saved = json.loads((tmp_path / "results" / "ogbn-mag" / "mag__rcms_k5__seed0__e1.json").read_text())
    assert [p["path"] for p in saved["discovery"]["paths"]] == rec["selected"]["rcms"]["5"]
    assert res["epochs"] == 1 and saved["experiment_id"] == "mag__rcms_k5__seed0__e1"
