import pytest
import torch

from auto_phgt.discovery import save_templates
from auto_phgt.tokenization import (PAD, MetaPathInstanceExtractor, MetaPathTemplate, SemanticTokenizer,
                          impute_missing_features, instance_statistics,
                          discover_or_load_templates, load_metapath_templates,
                          make_token_dataloader)


def edge_set(g, et):
    return set(map(tuple, g[et].edge_index.t().tolist()))


# --- templates ---------------------------------------------------------------
def test_template_parsing(toy_templates):
    t = MetaPathTemplate.from_list(toy_templates[2])
    assert t.node_types == ("paper", "paper", "term", "paper")
    assert t.edge_types[0] == ("paper", "cite", "paper")
    assert t.num_hops == 3 and t.length == 4
    with pytest.raises(ValueError):
        MetaPathTemplate.from_list(["paper", "to"])


def test_template_validation(toy_graph):
    with pytest.raises(ValueError):
        MetaPathTemplate.from_list(["paper", "writes", "author"]).validate(toy_graph)
    with pytest.raises(ValueError):
        MetaPathTemplate.from_list(["term", "to", "paper"]).validate(toy_graph, "paper")


def test_load_discovery_record(tmp_path, toy_graph, toy_templates):
    p = tmp_path / "acm_top_k_metapaths.json"
    save_templates(p, toy_templates, dataset="acm", target_node_type="paper",
                   max_hops=4, k=3)
    assert [t.schema for t in load_metapath_templates(toy_templates)] == [tuple(t) for t in toy_templates]
    loaded = discover_or_load_templates(toy_graph, dataset="acm", path=p, k=3)
    assert [t.schema for t in loaded] == [tuple(t) for t in toy_templates]


# --- features ----------------------------------------------------------------
def test_impute_missing_features(toy_graph):
    x = impute_missing_features(toy_graph)
    assert set(x) == {"paper", "author", "term"}
    # term 0 is linked to papers 0 and 3 (the largest incoming relation from a featured type)
    expected = toy_graph["paper"].x[[0, 3]].mean(0)
    assert torch.allclose(x["term"][0], expected, atol=1e-6)


def test_impute_small_chunks_match(toy_graph):
    a = impute_missing_features(toy_graph)
    b = impute_missing_features(toy_graph, chunk_elements=8)  # forces 1-edge chunks
    assert torch.allclose(a["term"], b["term"])


# --- extraction --------------------------------------------------------------
def test_extractor_shapes_and_type_consistency(toy_graph, toy_templates):
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=8, seed=1)
    ids = torch.arange(6)
    inst = ex.sample(ids)
    B, K, I, L = inst.node_ids.shape
    assert (B, K, I, L) == (6, 3, 8, 4)
    assert torch.equal(inst.node_ids[:, :, :, 0], ids.view(6, 1, 1).expand(6, 3, 8))

    for k, raw in enumerate(toy_templates):
        tpl = MetaPathTemplate.from_list(raw)
        assert (inst.node_ids[:, k, :, tpl.length:] == PAD).all()  # past template end
        for h, et in enumerate(tpl.edge_types):
            edges = edge_set(toy_graph, et)
            src, dst = inst.node_ids[:, k, :, h].reshape(-1), inst.node_ids[:, k, :, h + 1].reshape(-1)
            for s, d in zip(src.tolist(), dst.tolist()):
                if s == PAD:
                    assert d == PAD  # PAD is absorbing
                elif d != PAD:
                    assert (s, d) in edges  # each hop is a real edge of the right relation


def test_dead_end_is_padded(toy_graph, toy_templates):
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=4)
    inst = ex.sample([5])
    assert (inst.node_ids[0, 0, :, 1:] == PAD).all()        # paper 5 has no terms
    assert not inst.complete_mask[0, 0].any()
    assert inst.complete_mask[0, 1].all()                    # but has an author


def test_seeded_sampling_is_reproducible(toy_graph, toy_templates):
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=16)
    a, b = ex.sample(torch.arange(6), seed=7), ex.sample(torch.arange(6), seed=7)
    assert torch.equal(a.node_ids, b.node_ids)
    c, d = ex.sample(torch.arange(6)), ex.sample(torch.arange(6))
    assert not torch.equal(c.node_ids, d.node_ids)           # generator advances


def test_uniform_neighbour_sampling(toy_graph):
    ex = MetaPathInstanceExtractor(toy_graph, [["paper", "to", "author"]], instances_per_path=4000)
    first_hop = ex.sample([0], seed=0).node_ids[0, 0, :, 1]
    frac = (first_hop == 0).float().mean().item()            # paper 0 -> authors {0, 1}
    assert abs(frac - 0.5) < 0.05


def test_out_of_range_targets(toy_graph, toy_templates):
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates)
    with pytest.raises(IndexError):
        ex.sample([6])


def test_statistics(toy_graph, toy_templates):
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=4)
    stats = instance_statistics(ex.sample(torch.arange(6)), toy_templates)
    assert len(stats) == 3
    assert stats[0]["targets_with_any_instance"] == pytest.approx(5 / 6)


# --- tokenizer ---------------------------------------------------------------
@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_tokenizer_shapes_and_masks(toy_graph, toy_templates, pooling):
    x = impute_missing_features(toy_graph)
    tok = SemanticTokenizer.from_graph(toy_graph, toy_templates, x_dict=x, d_model=16,
                                       pooling=pooling, dropout=0.0)
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=2)
    inst = ex.sample(torch.arange(6))
    out = tok(x, inst)
    assert out.node_tokens.shape == (6, 16)
    assert out.semantic_tokens.shape == (6, 6, 16)
    assert out.path_tokens.shape == (6, 3, 2, 4, 16)
    assert torch.equal(out.semantic_mask, inst.complete_mask.reshape(6, 6))
    assert (out.semantic_tokens[~out.semantic_mask] == 0).all()
    assert (out.path_tokens[~out.path_mask] == 0).all()
    assert torch.equal(out.semantic_template_ids, torch.tensor([0, 0, 1, 1, 2, 2]))
    assert torch.isfinite(out.semantic_tokens).all()

    seq, kpm = out.sequence()
    assert seq.shape == (6, 7, 16) and kpm.shape == (6, 7)
    assert not kpm[:, 0].any()
    assert torch.equal(kpm[:, 1:], ~out.semantic_mask)


def test_featureless_type_uses_embedding(toy_graph, toy_templates):
    tok = SemanticTokenizer.from_graph(toy_graph, toy_templates,
                                       x_dict={"paper": toy_graph["paper"].x,
                                               "author": toy_graph["author"].x}, d_model=8)
    assert "term" in tok.projection.embeddings and "paper" in tok.projection.linears


def test_gradients_reach_projection(toy_graph, toy_templates):
    x = impute_missing_features(toy_graph)
    x.pop("term")                                            # exercise the embedding path too
    tok = SemanticTokenizer.from_graph(toy_graph, toy_templates, x_dict=x, d_model=8)
    inst = MetaPathInstanceExtractor(toy_graph, toy_templates).sample(torch.arange(5))
    out = tok(x, inst)
    (out.semantic_tokens.sum() + out.node_tokens.sum()).backward()
    for name in ("paper", "author"):
        assert tok.projection.linears[name].weight.grad.abs().sum() > 0
    assert tok.projection.embeddings["term"].weight.grad.abs().sum() > 0


def test_tokens_depend_only_on_path_nodes(toy_graph, toy_templates):
    x = impute_missing_features(toy_graph)
    inst = MetaPathInstanceExtractor(toy_graph, [toy_templates[1]]).sample([0], seed=0)
    tok1 = SemanticTokenizer.from_graph(toy_graph, [toy_templates[1]], x_dict=x, d_model=8, dropout=0.0)
    tok1.eval()
    base = tok1(x, inst).semantic_tokens
    used = set(inst.node_ids[0, 0, :, :3:2].reshape(-1).tolist())   # paper positions
    unused = [p for p in range(6) if p not in used]
    x2 = dict(x)
    x2["paper"] = x["paper"].clone()
    x2["paper"][unused] += 100.0
    assert torch.allclose(tok1(x2, inst).semantic_tokens, base)


def test_vocabulary_mismatch_raises(toy_graph, toy_templates):
    x = impute_missing_features(toy_graph)
    tok = SemanticTokenizer.from_graph(toy_graph, toy_templates, x_dict=x, d_model=8)
    inst = MetaPathInstanceExtractor(toy_graph, toy_templates).sample([0])
    inst.node_types = tuple(reversed(inst.node_types))
    with pytest.raises(ValueError):
        tok(x, inst)


def test_template_order_mismatch_raises(toy_graph, toy_templates):
    x = impute_missing_features(toy_graph)
    tok = SemanticTokenizer.from_graph(toy_graph, toy_templates, x_dict=x, d_model=8)
    reversed_templates = list(reversed(toy_templates))
    inst = MetaPathInstanceExtractor(toy_graph, reversed_templates).sample([0])
    with pytest.raises(ValueError, match="different ordered templates"):
        tok(x, inst)


# --- data pipeline -----------------------------------------------------------
def test_dataloader(toy_graph, toy_templates):
    ex = MetaPathInstanceExtractor(toy_graph, toy_templates, instances_per_path=2)
    y = toy_graph["paper"].y
    dl = make_token_dataloader(torch.arange(6), ex, batch_size=4, labels=y, shuffle=True, seed=0)
    batches = list(dl)
    assert [b["target_ids"].numel() for b in batches] == [4, 2]
    for b in batches:
        assert torch.equal(b["y"], y[b["target_ids"]])
        assert torch.equal(b["instances"].target_ids, b["target_ids"])

    # labels are the full per-node vector even when only a subset of targets is batched
    sub = make_token_dataloader(torch.tensor([4, 1, 3]), ex, batch_size=2, labels=y)
    for b in sub:
        assert torch.equal(b["y"], y[b["target_ids"]])

    fixed = make_token_dataloader(torch.arange(6), ex, batch_size=6, fixed_instances=True, seed=3)
    a, b = next(iter(fixed)), next(iter(fixed))
    assert torch.equal(a["instances"].node_ids, b["instances"].node_ids)
