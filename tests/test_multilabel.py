import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import HeteroData

from experiments.hgb_training import build_model, extractor_for, predict
from experiments.path_ranking import rank_by
from experiments import multilabel as ml

CFG = {"layers": 1, "lr": 1e-2, "wd": 1e-4, "dropout": 0.0, "heads": 2,
       "d_model": 16, "max_epochs": 6, "patience": 3, "pct_start": 0.5}
TEMPLATES = [["movie", "to", "actor", "to", "movie"],
             ["movie", "to", "director", "to", "movie"]]


@pytest.fixture
def multilabel_graph():
    generator = torch.Generator().manual_seed(17)
    graph = HeteroData()
    graph["movie"].x = torch.randn(8, 6, generator=generator)
    graph["actor"].x = torch.randn(5, 4, generator=generator)
    graph["director"].num_nodes = 3
    graph["movie"].y = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0],
                                     [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    for kind, endpoints in (("actor", [0, 1, 2, 3, 4, 0, 1, 2]),
                             ("director", [0, 1, 2, 0, 1, 2, 0, 1])):
        edges = torch.tensor([list(range(8)), endpoints])
        graph["movie", "to", kind].edge_index = edges
        graph[kind, "to", "movie"].edge_index = edges.flip(0)
    return graph


class BiasModel(torch.nn.Module):
    mode = "hgt"
    target_node_type = "movie"

    def __init__(self, classes=3):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(classes))
        self.register_buffer("training_forwards", torch.tensor(0))

    def forward(self, x_dict, edges, *, target_ids):
        if self.training:
            self.training_forwards.add_(1)
        return self.bias.expand(target_ids.numel(), -1)


def test_metrics_use_binary_indicators_and_fixed_inclusive_sigmoid_threshold():
    labels = torch.tensor([[1, 1, 0], [0, 1, 1]])
    logits = torch.tensor([[0.0, 2.0, -2.0], [-2.0, -0.25, 0.25]])
    scores = ml.multilabel_metrics(labels, logits)
    assert scores == pytest.approx({"micro_f1": 6 / 7, "macro_f1": 8 / 9})
    expected = (logits.sigmoid() >= 0.5).numpy()
    for average in ("micro", "macro"):
        assert scores[f"{average}_f1"] == pytest.approx(
            f1_score(labels.numpy(), expected, average=average, zero_division=0))
    assert set(scores) == {"micro_f1", "macro_f1"}


def test_single_label_column_f1_does_not_count_true_negatives_as_positives():
    scores = ml.multilabel_metrics(torch.tensor([[1], [0], [0]]),
                                    torch.tensor([[-2.0], [-2.0], [-2.0]]))
    assert scores == {"micro_f1": 0.0, "macro_f1": 0.0}


@pytest.mark.parametrize("labels,logits", [
    (torch.ones(3), torch.ones(3, 1)),
    (torch.empty(0, 2), torch.empty(0, 2)),
    (torch.empty(3, 0), torch.empty(3, 0)),
    (torch.tensor([[0.5, 1.0]]), torch.zeros(1, 2)),
    (torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 2)),
    (torch.ones(2, 3), torch.ones(2, 2)),
    (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long)),
    (torch.ones(1, 2), torch.tensor([[float("inf"), 0.0]])),
])
def test_invalid_multilabel_shapes_and_values_are_rejected(labels, logits):
    with pytest.raises(ValueError):
        ml.multilabel_metrics(labels, logits)


@pytest.mark.parametrize("mode", ["hgt", "auto_phgt", "tokens"])
@pytest.mark.parametrize("training", [True, False])
def test_actual_backbone_modes_train_and_evaluate_multilabel(multilabel_graph, mode, training):
    graph = multilabel_graph
    x = graph.x_dict
    model = build_model(graph, TEMPLATES, 3, x, CFG, mode)
    model.train(training)
    extractor = extractor_for(graph, TEMPLATES, 9) if mode != "hgt" else None
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    train_ids, val_ids, eval_ids = torch.arange(4), torch.tensor([4, 5]), torch.tensor([6, 7])
    result = ml.fit_multilabel(model, graph, x, extractor, train_ids, val_ids, CFG, 9)
    assert model.training is training
    assert set(result) == {"best_epoch", "best_val_loss", "val", "epochs_completed",
                           "early_stopped", "train_runtime_s", "gpu_memory", "parameters", "history"}
    assert 1 <= result["best_epoch"] <= result["epochs_completed"] <= CFG["max_epochs"]
    assert result["best_val_loss"] == min(row["val_loss"] for row in result["history"])
    assert result["parameters"] == sum(p.numel() for p in model.parameters())
    assert result["train_runtime_s"] >= 0
    assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(bool(g.abs().sum() > 0) for g in grads)
    logits = predict(model, graph, x, extractor, val_ids)
    assert logits.shape == (2, 3)
    assert float(F.binary_cross_entropy_with_logits(logits, graph["movie"].y[val_ids].float())) \
        == pytest.approx(result["best_val_loss"])
    assert ml.multilabel_metrics(graph["movie"].y[val_ids], logits) == result["val"]
    scores = ml.evaluate_multilabel(model, graph, x, extractor, eval_ids)
    assert scores == ml.evaluate_multilabel(model, graph, x, extractor, eval_ids)
    assert set(scores) == {"micro_f1", "macro_f1", "n"}
    assert scores["n"] == 2 and 0 <= scores["micro_f1"] <= 1 and 0 <= scores["macro_f1"] <= 1
    assert model.training is training


@pytest.mark.parametrize("training", [True, False])
def test_bce_early_stopping_restores_best_weights_buffers_and_mode(multilabel_graph, training):
    graph = multilabel_graph
    graph["movie"].y[:4] = 1
    graph["movie"].y[4:6] = 0
    model = BiasModel()
    model.train(training)
    result = ml.fit_multilabel(model, graph, graph.x_dict, None, torch.arange(4),
                               torch.tensor([4, 5]), {**CFG, "patience": 2}, 0)
    assert result["history"][0]["train_loss"] == pytest.approx(math.log(2))
    assert result["best_epoch"] == 1
    assert result["early_stopped"] and result["epochs_completed"] == 3
    assert result["history"][-1]["val_loss"] > result["best_val_loss"]
    assert model.training_forwards.item() == 1
    val_logits = predict(model, graph, graph.x_dict, None, torch.tensor([4, 5]))
    restored_loss = F.binary_cross_entropy_with_logits(val_logits, torch.zeros(2, 3))
    assert restored_loss.item() == pytest.approx(result["best_val_loss"])
    assert model.training is training


def test_optimizer_and_scheduler_use_the_same_configuration(monkeypatch, multilabel_graph):
    calls = {}
    adamw, onecycle = torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR

    def optimizer(*args, **kwargs):
        calls["optimizer"] = kwargs
        return adamw(*args, **kwargs)

    def scheduler(*args, **kwargs):
        calls["scheduler"] = kwargs
        return onecycle(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "OneCycleLR", scheduler)
    ml.fit_multilabel(BiasModel(), multilabel_graph, multilabel_graph.x_dict, None,
                      torch.arange(4), torch.tensor([4, 5]), CFG, 0)
    assert calls["optimizer"] == {"lr": CFG["lr"], "weight_decay": CFG["wd"]}
    assert calls["scheduler"] == {"max_lr": CFG["lr"], "total_steps": CFG["max_epochs"],
                                   "pct_start": CFG["pct_start"]}


@pytest.mark.parametrize("training", [True, False])
def test_forward_failure_restores_model_mode(monkeypatch, multilabel_graph, training):
    model = BiasModel()
    model.train(training)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        ml.fit_multilabel(model, multilabel_graph, {}, None, torch.arange(4),
                          torch.tensor([4, 5]), CFG, 0)
    assert model.training is training
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        ml.evaluate_multilabel(model, multilabel_graph, {}, None, torch.tensor([4, 5]))
    assert model.training is training


def test_fit_reads_only_requested_train_and_validation_labels(multilabel_graph):
    graph = multilabel_graph
    graph["movie"].y = graph["movie"].y.float()
    graph["movie"].y[6:] = float("nan")
    result = ml.fit_multilabel(BiasModel(), graph, graph.x_dict, None, torch.arange(4),
                               torch.tensor([4, 5]), CFG, 0)
    assert math.isfinite(result["best_val_loss"])
    with pytest.raises(ValueError, match="binary"):
        ml.evaluate_multilabel(BiasModel(), graph, {}, None, torch.tensor([6, 7]))


@pytest.mark.parametrize("train,val,cfg", [
    (torch.tensor([], dtype=torch.long), torch.tensor([4, 5]), CFG),
    (torch.tensor([0, 1, 1]), torch.tensor([4, 5]), CFG),
    (torch.tensor([-1, 0]), torch.tensor([4, 5]), CFG),
    (torch.tensor([0.0, 1.0]), torch.tensor([4, 5]), CFG),
    (torch.tensor([0, 1]), torch.tensor([1, 2]), CFG),
    (torch.tensor([0, 1]), torch.tensor([4, 5]), {**CFG, "max_epochs": 0}),
])
def test_invalid_training_splits_and_epoch_counts_are_rejected(multilabel_graph, train, val, cfg):
    with pytest.raises(ValueError):
        ml.fit_multilabel(BiasModel(), multilabel_graph, {}, None, train, val, cfg, 0)


def test_propagation_masks_all_heldout_labels_and_their_prior():
    rng = np.random.default_rng(13)
    block = rng.random((12, 12))
    labels = rng.integers(0, 2, size=(12, 4))
    folds = np.arange(12) % 3
    probabilities, mass, priors = ml.propagate_multilabel(block, labels, folds)
    for heldout in range(3):
        changed = labels.copy()
        changed[folds == heldout] = 1 - changed[folds == heldout]
        got, got_mass, got_priors = ml.propagate_multilabel(block, changed, folds)
        np.testing.assert_array_equal(got[heldout], probabilities[heldout])
        np.testing.assert_array_equal(got_priors[heldout], priors[heldout])
        np.testing.assert_array_equal(got_mass, mass)


def test_diagonal_is_removed_even_for_nonheldout_rows_without_mutating_input():
    rng = np.random.default_rng(19)
    block = rng.random((9, 9))
    labels = rng.integers(0, 2, size=(9, 3))
    folds = np.arange(9) % 3
    diagonal_free = block.copy()
    np.fill_diagonal(diagonal_free, 0)
    original = block.copy()
    actual = ml.propagate_multilabel(block, labels, folds)
    expected = ml.propagate_multilabel(diagonal_free, labels, folds)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(block, original)


def test_denominator_is_endpoint_mass_not_positive_label_mass():
    labels = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0]])
    block = np.zeros((4, 4))
    block[0] = [100.0, 10.0, 0.25, 0.75]
    probabilities, mass, _ = ml.propagate_multilabel(block, labels, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(probabilities[0, 0], [0.25, 1.0, 0.25])
    assert mass[0, 0] == 1.0
    labels[3] = 0
    probabilities, mass, _ = ml.propagate_multilabel(block, labels, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(probabilities[0, 0], [0.25, 0.25, 0.25])
    assert mass[0, 0] == 1.0


def test_zero_evidence_uses_only_known_fold_prevalence():
    labels = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    folds = np.array([0, 0, 1, 1])
    probabilities, mass, priors = ml.propagate_multilabel(np.eye(4), labels, folds)
    assert not mass.any()
    for j in range(2):
        expected = labels[folds != j].mean(axis=0)
        np.testing.assert_array_equal(priors[j], expected)
        np.testing.assert_array_equal(probabilities[j], np.broadcast_to(expected, (4, 3)))
    scores = ml.multilabel_fastpath_scores(np.eye(4), labels, folds=2)
    assert scores["coverage"] == 0 and scores["labelled_mass"] == 0
    assert math.isfinite(scores["BCE"])


def test_folds_are_reproducible_balanced_and_label_independent():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=(17, 4))
    block = np.zeros((17, 17))
    first = ml.multilabel_fastpath_scores(block, labels, folds=3, seed=12)
    assert first == ml.multilabel_fastpath_scores(block, labels, folds=3, seed=12)
    second = ml.multilabel_fastpath_scores(block, 1 - labels, folds=3, seed=12)
    assert first["fold_ids"] == second["fold_ids"]
    assert first["fold_ids"] != ml.multilabel_fastpath_scores(block, labels, seed=13)["fold_ids"]
    assert max(first["fold_sizes"]) - min(first["fold_sizes"]) <= 1
    assert sum(first["fold_sizes"]) == 17
    assert first["known_sizes"] == [17 - size for size in first["fold_sizes"]]
    assert first["folds"] == 3 and first["fold_seed"] == 12
    assert "label-independent" in first["fold_rule"]


@pytest.mark.parametrize("eps", [0.0, 0.1, 1.0])
def test_fastpath_is_negative_mean_oof_bce_with_prior_smoothing(eps):
    rng = np.random.default_rng(27)
    labels = rng.integers(0, 2, size=(15, 3))
    block = rng.random((15, 15))
    scores = ml.multilabel_fastpath_scores(block, labels, folds=3, seed=4, smoothing_eps=eps)
    folds = np.array(scores["fold_ids"])
    probabilities, mass, priors = ml.propagate_multilabel(block, labels, folds)
    idx = np.arange(15)
    oof = probabilities[folds, idx]
    smoothed = (1 - eps) * oof + eps * priors[folds]
    clipped = np.clip(smoothed, scores["bce_clip"], 1 - scores["bce_clip"])
    expected = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()
    assert scores["Q_FP"] == pytest.approx(-expected)
    assert scores["CE"] == scores["BCE"] == pytest.approx(expected)
    assert scores["coverage"] == pytest.approx((mass[folds, idx] > 0).mean())
    for average in ("micro", "macro"):
        assert scores[f"{average}_f1"] == pytest.approx(
            f1_score(labels, oof >= 0.5, average=average, zero_division=0))
    assert "accuracy" not in scores


def test_fastpath_prefers_informative_independent_candidates_and_is_mass_scale_invariant():
    labels = np.repeat(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0]]), 15, axis=0)
    good = (labels[:, None, :] == labels[None, :, :]).all(axis=-1).astype(float)
    noise = np.ones_like(good)
    informative = ml.multilabel_fastpath_scores(good, labels, seed=3)
    uninformative = ml.multilabel_fastpath_scores(noise, labels, seed=3)
    scaled = ml.multilabel_fastpath_scores(good * 0.1, labels, seed=3)
    assert rank_by([uninformative["Q_FP"], informative["Q_FP"]]) == [1, 0]
    assert informative["micro_f1"] == informative["macro_f1"] == 1.0
    assert scaled["Q_FP"] == pytest.approx(informative["Q_FP"])
    assert scaled["labelled_mass"] == pytest.approx(informative["labelled_mass"] * 0.1)


def test_all_zero_labels_still_have_known_endpoint_coverage():
    labels = np.zeros((6, 2))
    probabilities, mass, priors = ml.propagate_multilabel(
        np.ones((6, 6)), labels, np.arange(6) % 3)
    assert not probabilities.any() and not priors.any() and (mass > 0).all()
    scores = ml.multilabel_fastpath_scores(np.ones((6, 6)), labels)
    assert scores["coverage"] == 1.0 and math.isfinite(scores["BCE"])
    assert scores["micro_f1"] == scores["macro_f1"] == 0.0


@pytest.mark.parametrize("block,labels,kwargs", [
    (np.zeros((3, 2)), np.zeros((3, 2)), {}),
    (-np.eye(3), np.zeros((3, 2)), {}),
    (np.eye(3) * np.nan, np.zeros((3, 2)), {}),
    (np.eye(3, dtype=complex), np.zeros((3, 2)), {}),
    (np.eye(3), np.zeros(3), {}),
    (np.eye(3), np.full((3, 2), 0.5), {}),
    (np.eye(3), np.full((3, 2), np.inf), {}),
    (np.eye(3), np.empty((3, 0)), {}),
    (np.eye(1), np.zeros((1, 2)), {}),
    (np.eye(3), np.zeros((3, 2)), {"folds": 1}),
    (np.eye(3), np.zeros((3, 2)), {"folds": 4}),
    (np.eye(3), np.zeros((3, 2)), {"folds": 2.0}),
    (np.eye(3), np.zeros((3, 2)), {"folds": True}),
    (np.eye(3), np.zeros((3, 2)), {"seed": -1}),
    (np.eye(3), np.zeros((3, 2)), {"seed": 0.5}),
    (np.eye(3), np.zeros((3, 2)), {"smoothing_eps": -0.1}),
    (np.eye(3), np.zeros((3, 2)), {"smoothing_eps": 1.1}),
    (np.eye(3), np.zeros((3, 2)), {"smoothing_eps": np.nan}),
])
def test_invalid_fastpath_inputs_are_rejected(block, labels, kwargs):
    with pytest.raises(ValueError):
        ml.multilabel_fastpath_scores(block, labels, **kwargs)


@pytest.mark.parametrize("folds", [
    np.array([0, 0, 0]), np.array([0, 2, 2]), np.array([-1, 0, 1]),
    np.array([0.0, 1.0, 2.0]), np.array([[0, 1, 2]]), np.array([0, 1]),
])
def test_invalid_fold_assignments_are_rejected(folds):
    with pytest.raises(ValueError):
        ml.propagate_multilabel(np.eye(3), np.zeros((3, 2)), folds)
