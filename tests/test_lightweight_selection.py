import json

import numpy as np
import pytest
import torch

from experiments import lightweight_campaign as master
from experiments import lightweight_recovery as recovery
from experiments.lightweight_selection import plan, selectors as sel, stats, transitions as tr

PTP = ["paper", "to", "term", "to", "paper"]
PAP = ["paper", "to", "author", "to", "paper"]
PCTP = ["paper", "cite", "paper", "to", "term", "to", "paper"]


def dense_ops(graph):
    ops = {}
    for et in graph.edge_types:
        src, _, dst = et
        a = np.zeros((graph[src].num_nodes, graph[dst].num_nodes))
        ei = graph[et].edge_index.numpy()
        np.add.at(a, (ei[0], ei[1]), 1.0)
        deg = a.sum(1, keepdims=True)
        ops[et] = np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)
    return ops


def brute_ti(graph, schema):
    ops = dense_ops(graph)
    t = np.eye(graph[schema[0]].num_nodes)
    for et in tr.edges_of(schema):
        t = t @ ops[et]
    rows, covered, any_ = [], [], 0
    for u in range(t.shape[0]):
        row = t[u].copy()
        any_ += row.sum() > 0
        row[u] = 0.0
        if row.sum() > 0:
            rows.append(row / row.sum())
            covered.append(u)
    if not rows:
        return {"C": 0.0, "I": 0.0, "C_any": any_ / t.shape[0]}, t, [], None
    p = np.stack(rows)
    q = p.mean(0)
    kl = [sum(x * np.log(x / q[v]) for v, x in enumerate(r) if x > 0) for r in p]
    return {"C": len(rows) / t.shape[0], "I": float(np.mean(kl)),
            "C_any": any_ / t.shape[0]}, t, covered, (p, q)


# --------------------------------------------------------------------------- transitions
def test_relation_operators_are_row_stochastic(toy_graph):
    ops = tr.relation_operators(toy_graph)
    for et, op in ops.items():
        sums = np.asarray(op.sum(1)).ravel()
        assert np.all((np.abs(sums - 1) < 1e-6) | (sums == 0)), et


@pytest.mark.parametrize("schema", [PTP, PAP, PCTP])
@pytest.mark.parametrize("dense_fraction", [0.0, 1.1])
def test_transition_info_matches_brute_force(toy_graph, schema, dense_fraction):
    ops = tr.relation_operators(toy_graph)
    proj = tr.gaussian_projection(6, 4, 0)
    got, fp = tr.transition_info(ops, schema, 6, chunk=4, dense_fraction=dense_fraction,
                                 projection=proj)
    ref, _, covered, pq = brute_ti(toy_graph, schema)
    assert got["C"] == pytest.approx(ref["C"]) and got["C_any"] == pytest.approx(ref["C_any"])
    assert got["I"] == pytest.approx(ref["I"], abs=1e-5)
    assert got["TI_cov"] == pytest.approx(got["C"] * got["I"])
    assert got["TI_norm"] == pytest.approx(got["C"] * got["I"] / (got["H_q"] + 1e-12))
    if pq is not None:
        p, q = pq
        expect = np.zeros((6, 4))
        expect[covered] = (p - q) @ proj
        assert np.allclose(fp, expect, atol=1e-5)


def test_self_returns_carry_no_information():
    from torch_geometric.data import HeteroData
    g = HeteroData()
    g["paper"].num_nodes = 3
    g["author"].num_nodes = 3
    pa = torch.tensor([[0, 1, 2], [0, 1, 2]])  # every author writes one paper
    g["paper", "to", "author"].edge_index = pa
    g["author", "to", "paper"].edge_index = pa.flip(0)
    got, _ = tr.transition_info(tr.relation_operators(g), PAP, 3)
    assert got["C"] == 0.0 and got["I"] == 0.0 and got["C_any"] == 1.0
    assert got["I_incl_self"] == pytest.approx(np.log(3))  # identity map: maximal, useless


def test_train_block_removes_the_diagonal(toy_graph):
    ops = tr.relation_operators(toy_graph)
    ids = np.array([0, 2, 3, 4])
    block, completion = tr.train_block(ops, PAP, ids, chunk=2)
    _, t, _, _ = brute_ti(toy_graph, PAP)
    tn = t / np.where(t.sum(1, keepdims=True) > 0, t.sum(1, keepdims=True), 1)
    expect = tn[np.ix_(ids, ids)]
    np.fill_diagonal(expect, 0)
    assert np.allclose(block, expect, atol=1e-6)
    assert np.allclose(completion, t[ids].sum(1), atol=1e-6)


def test_beam_search_discovers_valid_paths(toy_graph):
    ops = tr.relation_operators(toy_graph)
    res = tr.beam_search(toy_graph, ops, "paper", variant="TI_cov", width=3, lmax=4,
                         sample_sources=6, sample_seed=0, min_coverage=1e-3,
                         min_effective_endpoints=1.0, duplicate_cosine=0.9999,
                         fingerprint_dim=4)
    assert res["discovered"]
    for row in res["discovered"]:
        p = row["path"]
        assert p[0] == p[-1] == "paper" and 2 <= len(p) // 2 <= 4
        for et in tr.edges_of(p):
            assert et in toy_graph.edge_types
    assert any(entry["action"] == "keep" for entry in res["log"])


# --------------------------------------------------------------------------- FastPath
def test_propagation_is_echo_free_and_masks_the_fold():
    rng = np.random.default_rng(0)
    n, c = 12, 3
    block = rng.random((n, n)).astype(np.float32)
    np.fill_diagonal(block, 0)
    y = np.arange(n) % c
    folds = sel.stratified_folds(y, 3, 0)
    L, W, _ = sel.propagate(block, y, folds, c)
    for u in range(n):
        y2 = y.copy()
        y2[u] = (y[u] + 1) % c  # a node's own label never changes its out-of-fold evidence
        L2, _, _ = sel.propagate(block, y2, folds, c)
        assert np.allclose(L2[folds[u], u], L[folds[u], u])
    j = 0
    y3 = y.copy()
    y3[folds == j] = (y3[folds == j] + 1) % c  # fold-j labels are masked for fold j
    L3, _, _ = sel.propagate(block, y3, folds, c)
    assert np.allclose(L3[j], L[j])


def test_fastpath_scores_prefer_the_informative_path():
    rng = np.random.default_rng(1)
    n, c = 90, 3
    y = np.repeat(np.arange(c), n // c)
    same = (y[:, None] == y[None, :]).astype(np.float32)
    np.fill_diagonal(same, 0)
    good = same / same.sum(1, keepdims=True)
    noise = rng.random((n, n)).astype(np.float32)
    np.fill_diagonal(noise, 0)
    noise /= noise.sum(1, keepdims=True)
    folds = sel.stratified_folds(y, 3, 0)
    scores, feats = [], []
    for b in (noise, good):
        L, W, pri = sel.propagate(b, y, folds, c)
        scores.append(sel.fastpath_scores(L, W, pri, y, folds, 0.1))
        feats.append(sel.fold_features(L, W))
    assert scores[1]["Q_FP"] > scores[0]["Q_FP"] and scores[1]["accuracy"] > 0.9
    chosen, steps, single, empty = sel.fastpath_set_greedy(np.stack(feats), y, folds, c, 2)
    assert chosen[0] == 1 and steps[0]["gain"] > 0 and single[1] < single[0] < empty + 0.2


def test_ti_set_avoids_duplicates():
    fps = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    chosen, steps = sel.ti_set_greedy([3.0, 2.9, 1.0], sel.cosine_matrix(fps), 2)
    assert chosen == [0, 2] and steps[1]["max_similarity"] == 0.0
    assert sel.rank_by([3.0, 2.9, 1.0])[:2] == [0, 1]


def test_stratified_folds_are_balanced():
    y = np.array([0] * 9 + [1] * 6)
    folds = sel.stratified_folds(y, 3, 5)
    for f in range(3):
        assert (y[folds == f] == 0).sum() == 3 and (y[folds == f] == 1).sum() == 2


# --------------------------------------------------------------------------- stats
def test_paired_statistics():
    a = {s: 0.80 + 0.01 * s for s in range(5)}
    b = {s: 0.79 + 0.01 * s for s in range(5)}
    res = stats.paired(a, b, 1000, 0)
    assert res["mean"] == pytest.approx(1.0) and res["ci95"] == pytest.approx([1.0, 1.0])
    assert res["p_perm"] == pytest.approx(2 / 32) and stats.verdict(res["ci95"]) == "above"
    d = stats.spearman_diff_ci([1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1], 200, 0)
    assert d["diff"] == pytest.approx(0.0)
    assert stats.rank_of([0.5, None, 2.0]) == [2, 3, 1]


# --------------------------------------------------------------------------- plan and master
def test_plan_is_consistent():
    ids = [c["id"] for c in plan.conditions() + plan.bridge_conditions()]
    assert len(ids) == len(set(ids))
    for ds, ref in plan.HGB_REFERENCE.items():
        grid = plan.GRID[ds]
        assert all(ref[k] in grid[k] for k in grid)
    p0 = {c["name"] for c in plan.conditions() if c["dataset"] == "freebase" and c["priority"] == 0}
    assert {"hgt", "canonical_k5", "random_k5", "hybrid_k5", "ti_k5", "fastpath_k5",
            "fastpath_set_k5"} <= p0
    assert len(plan.protocol_hash()) == 16


def test_task_graph_dependencies():
    tasks = master.build_tasks()
    by = {t["id"]: t for t in tasks}
    for t in tasks:
        if t["fn"] == "tune":
            assert "acm__diagnostic" in t["after"]
        if t["fn"] == "final" and t["args"]["cond_id"].split("__")[1].startswith("fastpath"):
            assert f"{t['meta']['dataset']}__fastpath" in t["deps"]
        if t["fn"] == "final" and t["args"]["seed"] >= 5:
            assert t["gate"] == "freebase_extension" and "freebase__decision" in t["deps"]
            assert t["priority"] == 1
    assert by["acm__ti_decision"]["after"] == ["acm__diagnostic"]
    assert all(by[d]["size"] == "medium" for d in by if d.startswith("freebase__tune"))


class FakeProc:
    def __init__(self, code=0):
        self.code, self.pid, self.returncode = code, 0, code

    def poll(self):
        return self.code

    def wait(self):
        return self.code


def run_fake_master(tmp_path, monkeypatch, tasks, fail=()):
    monkeypatch.setattr(master, "V4", tmp_path)
    order = []

    def launch(spec, gpu):
        order.append(spec["id"])
        status = "failed" if spec["id"] in fail else "done"
        master.write_status(spec["id"], {"id": spec["id"], "status": status})
        return FakeProc(1 if spec["id"] in fail else 0)

    m = master.Master(tasks, ["0"], 8, None, launch=launch, poll=0.0, aggregate=lambda: None)
    m.run()
    return m, order


def test_master_orders_tiers_and_after(tmp_path, monkeypatch):
    T = master.task
    tasks = [T("a", "x", {}, est=1), T("diag", "x", {}, deps=["a"], est=1),
             T("tune", "x", {}, after=["diag"], size="small", est=1),
             T("p1", "x", {}, priority=1, size="small", est=1),
             T("p2", "x", {}, priority=2, size="small", est=1)]
    m, order = run_fake_master(tmp_path, monkeypatch, tasks, fail=("diag",))
    assert order.index("diag") < order.index("tune")  # after: runs even though diag failed
    assert m.state["tune"] == "done" and m.state["diag"] == "failed"
    assert order.index("p1") < order.index("tune")  # P1 fills resources P0 cannot use yet
    assert order.index("tune") < order.index("p2")  # P2 waits for every P0/P1 task


class HeldProc(FakeProc):
    def poll(self):
        return None


def test_lower_tier_never_takes_resources_from_ready_task(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "V4", tmp_path)
    T = master.task
    tasks = [T(f"fb{i}", "x", {}, size="medium", est=1) for i in range(3)]
    tasks += [T("p1", "x", {}, priority=1, size="small", est=1),
              T("p1cpu", "x", {}, priority=1, est=1)]
    started = []

    def launch(spec, gpu):
        started.append(spec["id"])
        return HeldProc()

    m = master.Master(tasks, ["0"], 8, None, launch=launch, poll=0.0, aggregate=lambda: None)
    m.step()
    assert started == ["fb0", "fb1"]  # fb2 is ready but waits for GPU units: P1 must not run


def test_master_gates_skip(tmp_path, monkeypatch):
    T = master.task
    tasks = [T("g", "x", {}, gate="closed", est=1), T("after_g", "x", {}, deps=["g"], est=1)]
    monkeypatch.setattr(master, "gate_open", lambda gate, args: gate != "closed")
    m, order = run_fake_master(tmp_path, monkeypatch, tasks)
    assert m.state["g"] == "skipped" and m.state["after_g"] == "done" and order == ["after_g"]
    assert json.loads((tmp_path / "status" / "g.json").read_text())["status"] == "skipped"


def test_recovery_selects_only_failed_freebase_finals():
    specs = [
        {"id": "failed_fb", "fn": "final", "meta": {"dataset": "freebase"}, "priority": 1},
        {"id": "done_fb", "fn": "final", "meta": {"dataset": "freebase"}, "priority": 0},
        {"id": "failed_acm", "fn": "final", "meta": {"dataset": "acm"}, "priority": 0},
        {"id": "failed_search", "fn": "ti", "meta": {"dataset": "freebase"}, "priority": 0},
        {"id": "failed_fb_p0", "fn": "final", "meta": {"dataset": "freebase"}, "priority": 0},
    ]
    states = {
        "failed_fb": {"status": "failed"},
        "done_fb": {"status": "done"},
        "failed_acm": {"status": "failed"},
        "failed_search": {"status": "failed"},
        "failed_fb_p0": {"status": "failed"},
    }
    chosen = recovery.select_targets(specs, states.__getitem__)
    assert [spec["id"] for spec in chosen] == ["failed_fb_p0", "failed_fb"]


class HoldingProc:
    def __init__(self):
        self.pid = 0

    def poll(self):
        return None


def test_recovery_runs_exactly_one_worker_per_gpu(monkeypatch):
    specs = [
        {"id": f"t{i}", "priority": 0, "est_minutes": 1}
        for i in range(4)
    ]
    monkeypatch.setattr(recovery, "read_status", lambda _task_id: {"status": "failed"})
    launched = []

    def launch(spec, gpu):
        launched.append((spec["id"], gpu))
        return HoldingProc()

    pool = recovery.Recovery(specs, ["0", "1"], None, launch=launch)
    assert pool.step() == 2
    assert launched == [("t0", "0"), ("t1", "1")]
    assert len(pool.running) == 2
    assert len({run.gpu for run in pool.running.values()}) == len(pool.running)
    assert pool.step() == 0


def test_abbreviation_matches_v2(toy_graph):
    from experiments.selector_controls_summary import _abbr
    from experiments.lightweight_selection.tasks import abbr
    for schema in (PTP, PAP, PCTP):
        assert abbr(schema) == _abbr(schema)


def fake_result(ds, name, seed, micro, mode="auto_phgt"):
    path = ["paper", "to", "author", "to", "paper"]
    return {"status": "completed", "dataset": ds, "name": name, "seed": seed, "mode": mode,
            "k": 5, "condition": f"{ds}__{name}",
            "test": {"micro_f1": micro, "macro_f1": micro - 0.01},
            "validation": {"micro_f1": micro + 0.01, "macro_f1": micro},
            "fit": {"train_runtime_s": 30.0, "best_epoch": 10, "gpu_memory": {}},
            "parameters": 1000, "selection_runtime_s": 0.1,
            "templates": [path] if name != "fastpath_set_k5" else [path, path],
            "discovery": {"paths": [{"path": path, "canonical_rank": 1}]}}


def test_aggregate_on_synthetic_results(tmp_path, monkeypatch):
    from experiments.lightweight_selection import aggregate
    monkeypatch.setattr(aggregate, "V4", tmp_path)
    monkeypatch.setattr(aggregate, "SUMMARY", tmp_path / "summary")
    monkeypatch.setattr(aggregate, "PLOTS", tmp_path / "plots")
    monkeypatch.setattr(plan, "V2_ACM_RESULTS", tmp_path / "v2")
    base = {"hgt": 0.80, "canonical_k5": 0.81, "random_k5": 0.82, "hybrid_k5": 0.82,
            "ti_k5": 0.83, "ti_set_k5": 0.83, "fastpath_k5": 0.82, "fastpath_set_k5": 0.85}
    rng = np.random.default_rng(0)
    for ds in plan.DATASETS:
        for name, value in base.items():
            for seed in range(5):
                rec = fake_result(ds, name, seed, value + 0.002 * rng.standard_normal(),
                                  mode="hgt" if name == "hgt" else "auto_phgt")
                out = tmp_path / "results" / ds / f"{ds}__{name}__seed{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec))
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "ti_primary.json").write_text(json.dumps({"primary": "TI_cov"}))
    out = aggregate.aggregate()
    assert out["hypotheses"]["H4"]["verdict"] == "supported"
    assert out["hypotheses"]["H3"]["verdict"] == "supported"
    assert out["hypotheses"]["H5"]["verdict"] == "supported"
    assert out["outcomes"]["C"] is True
    for name in ("main_results.csv", "statistics.csv", "efficiency.csv", "final_summary.md",
                 "paper_claims.md", "selected_paths.csv", "correlations.csv"):
        assert (tmp_path / "summary" / name).exists(), name
    assert (tmp_path / "plots" / "selected_paths.png").exists()
