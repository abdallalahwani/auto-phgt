import pytest
import torch

from auto_phgt.model import AutoPHGT, MaskedMultiHeadSelfAttention
from auto_phgt.tokenization import MetaPathInstanceExtractor, impute_missing_features


def build(graph, templates, mode="auto_phgt", d_model=16):
    torch.manual_seed(0)
    x = impute_missing_features(graph)
    model = AutoPHGT.build(graph, templates, num_classes=3, x_dict=x, d_model=d_model,
                           hgt_layers=1, hgt_heads=2, fusion_layers=2, fusion_heads=2, mode=mode)
    ex = MetaPathInstanceExtractor(graph, templates, instances_per_path=2)
    return model, x, ex


# --- attention masking ---------------------------------------------------------
def test_masked_keys_receive_zero_attention():
    torch.manual_seed(0)
    attn = MaskedMultiHeadSelfAttention(8, 2, dropout=0.0)
    x = torch.randn(2, 5, 8)
    kpm = torch.tensor([[False, False, True, True, False], [False, True, True, True, True]])
    _, w = attn(x, key_padding_mask=kpm)
    assert torch.all(w[kpm.view(2, 1, 1, 5).expand_as(w)] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 2, 5))


def test_padded_token_content_is_ignored():
    torch.manual_seed(0)
    attn = MaskedMultiHeadSelfAttention(8, 2, dropout=0.0)
    x = torch.randn(1, 4, 8)
    kpm = torch.tensor([[False, False, True, True]])
    out1, _ = attn(x, kpm)
    x2 = x.clone()
    x2[:, 2:] = torch.randn(1, 2, 8) * 50
    out2, _ = attn(x2, kpm)
    assert torch.allclose(out1[:, :2], out2[:, :2], atol=1e-6)


def test_fully_masked_row_is_finite():
    attn = MaskedMultiHeadSelfAttention(8, 2, dropout=0.0)
    out, w = attn(torch.randn(1, 3, 8), key_padding_mask=torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(out).all() and (w == 0).all()


def test_structural_attn_mask():
    attn = MaskedMultiHeadSelfAttention(8, 2, dropout=0.0)
    causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    _, w = attn(torch.randn(1, 4, 8), attn_mask=causal)
    assert torch.all(w[..., causal] == 0)


# --- full model ----------------------------------------------------------------
@pytest.mark.parametrize("mode", ["auto_phgt", "hgt", "tokens"])
def test_forward_backward_all_modes(toy_graph, toy_templates, mode):
    model, x, ex = build(toy_graph, toy_templates, mode)
    inst = ex.sample(torch.arange(6))
    logits = model(x, toy_graph.edge_index_dict, instances=inst)
    assert logits.shape == (6, 3)
    loss = torch.nn.functional.cross_entropy(logits, toy_graph["paper"].y)
    loss.backward()
    grads = [p.grad for p in model.head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    proj_grad = model.projection.linears["paper"].weight.grad
    assert proj_grad is not None and proj_grad.abs().sum() > 0


def test_predict_proba(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates)
    model.train()
    inst = ex.sample(torch.arange(6), seed=1)
    p = model.predict_proba(x, toy_graph.edge_index_dict, instances=inst)
    assert p.shape == (6, 3)
    assert torch.allclose(p.sum(-1), torch.ones(6), atol=1e-6) and (p >= 0).all()
    assert model.training
    assert torch.allclose(p, model.predict_proba(x, toy_graph.edge_index_dict, instances=inst))


def test_hgt_mode_accepts_target_ids_only(toy_graph, toy_templates):
    model, x, _ = build(toy_graph, toy_templates, mode="hgt")
    assert model(x, toy_graph.edge_index_dict, target_ids=torch.tensor([1, 4])).shape == (2, 3)
    assert model.fusion is None
    assert model.tokenizer is None
    assert not any(name.startswith("tokenizer.") for name, _ in model.named_parameters())
    parts = model.num_parameters()
    assert parts["tokenizer (excl. projection)"] == 0
    assert parts["total"] == sum(value for key, value in parts.items() if key != "total")


def test_subgraph_interface_matches_full_graph(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates)
    model.eval()
    inst = ex.sample(torch.tensor([2, 4]), seed=0)
    full = model(x, toy_graph.edge_index_dict, instances=inst)
    n_id = {t: torch.arange(toy_graph[t].num_nodes) for t in toy_graph.node_types}
    sub = model(x, toy_graph.edge_index_dict, instances=inst, n_id_dict=n_id,
                target_local=torch.tensor([2, 4]))
    assert torch.allclose(full, sub, atol=1e-6)


def test_padded_semantic_tokens_do_not_change_prediction(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates, mode="tokens")
    model.eval()
    inst = ex.sample(torch.tensor([5]), seed=0)             # paper 5: template 0 and 2 dead-end
    logits, det = model(x, instances=inst, return_details=True)
    assert det["key_padding_mask"][0, 1:3].all()
    tokens = det["tokens"]
    seq, kpm = tokens.sequence()
    seq2 = seq.clone()
    seq2[kpm] = 123.0
    fused1, _ = model.fusion(seq, kpm)
    fused2, _ = model.fusion(seq2, kpm)
    assert torch.allclose(fused1[:, 0], fused2[:, 0], atol=1e-5)


def test_template_attention_sums_to_one(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates)
    model.eval()
    _, det = model(x, toy_graph.edge_index_dict, instances=ex.sample(torch.arange(6)),
                   return_details=True)
    share = AutoPHGT.template_attention(det, num_templates=3)
    assert share.shape == (6, 4)
    assert torch.allclose(share.sum(-1), torch.ones(6), atol=1e-5)


def test_missing_inputs_raise(toy_graph, toy_templates):
    model, x, ex = build(toy_graph, toy_templates)
    with pytest.raises(ValueError):
        model(x, toy_graph.edge_index_dict, target_ids=torch.tensor([0]))
    with pytest.raises(ValueError):
        model(x, None, instances=ex.sample([0]))


def test_can_overfit_toy_graph(toy_graph, toy_templates):
    """End-to-end sanity check: the model can memorise 6 labels."""
    model, x, ex = build(toy_graph, toy_templates, d_model=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    y = toy_graph["paper"].y
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(x, toy_graph.edge_index_dict, instances=ex.sample(torch.arange(6))), y)
        loss.backward()
        opt.step()
    model.eval()
    pred = model(x, toy_graph.edge_index_dict, instances=ex.sample(torch.arange(6), seed=0)).argmax(-1)
    assert (pred == y).float().mean() >= 5 / 6
