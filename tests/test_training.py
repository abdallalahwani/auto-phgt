import torch

from auto_phgt.evaluation import (evaluate_full_graph, evaluate_sampled_graph,
                                  parameter_count)
from auto_phgt.model import AutoPHGT
from auto_phgt.sampling import HeteroNeighborSampler
from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features
from auto_phgt.training import fit_full_graph, fit_sampled_graph, split_ids


def make_model(graph, templates):
    features = impute_missing_features(graph)
    extractor = MetaPathInstanceExtractor(graph, templates, instances_per_path=2, seed=0)
    model = AutoPHGT.build(graph, templates, 3, x_dict=features, d_model=16,
                           hgt_layers=1, hgt_heads=2, fusion_layers=1, fusion_heads=2,
                           dropout=0.0)
    return model, features, extractor


def test_split_and_full_graph_training(toy_graph, toy_templates):
    toy_graph["paper"].train_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    toy_graph["paper"].test_mask = ~toy_graph["paper"].train_mask
    train, val, test = split_ids(toy_graph, seed=3, acm_validation_size=1)
    assert train.numel() == 4 and val.numel() == 1 and test.tolist() == [5]
    assert not (set(train.tolist()) & set(val.tolist()))

    model, features, extractor = make_model(toy_graph, toy_templates)
    result = fit_full_graph(model, toy_graph, features, extractor, train, val,
                            epochs=2, patience=1, seed=0)
    assert 1 <= result["best_epoch"] <= 2
    assert len(result["history"]) == result["best_epoch"] or len(result["history"]) == 2
    assert result["parameters"] == parameter_count(model)
    assert result["runtime_s"] >= 0
    metrics = evaluate_full_graph(model, features, toy_graph.edge_index_dict,
                                  extractor, toy_graph["paper"].y, test)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_sampled_training_and_evaluation(toy_graph, toy_templates):
    model, features, extractor = make_model(toy_graph, toy_templates)
    train_sampler = HeteroNeighborSampler(toy_graph, [4], seed=0)
    eval_sampler = HeteroNeighborSampler(toy_graph, [4], seed=1)
    train, val = torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5])
    result = fit_sampled_graph(model, toy_graph, features, extractor,
                               train_sampler, eval_sampler, train, val,
                               epochs=1, patience=1, batch_size=2, seed=0)
    assert result["best_epoch"] == 1
    first = evaluate_sampled_graph(model, features, extractor, eval_sampler,
                                   toy_graph["paper"].y, val, batch_size=2, seed=4)
    second = evaluate_sampled_graph(model, features, extractor, eval_sampler,
                                    toy_graph["paper"].y, val, batch_size=2, seed=4)
    assert first["accuracy"] == second["accuracy"]
    assert first["macro_f1"] == second["macro_f1"]
