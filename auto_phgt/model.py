"""
Module 3: Model Architecture (Auto-PHGT)
=========================================

A PyTorch ``nn.Module`` that takes node tokens and semantic tokens and outputs
class logits for target nodes (or probabilities through ``predict_proba``).

    x_dict, edge_index_dict ──> TypeAwareProjection (shared with Module 2)
                               └─> BaseHGTEncoder (L x HGTConv)  ──> node token h_v
    x_dict, PathInstances   ──> SemanticTokenizer (Module 2)     ──> semantic tokens s_1..s_{K*I}
    [h_v ; s_1 .. s_{K*I}]  ──> AttentionFusionTransformer (W_Q, W_K, W_V + key-padding mask)
                               └─> ClassificationHead on the node-token position ──> logits / probabilities

The model has three modes so the ablation in the proposal runs on one code path:
  * ``"auto_phgt"`` : base HGT node token + automatically discovered semantic tokens (full model)
  * ``"hgt"``       : base HGT only, no semantic tokens (vanilla HGT baseline)
  * ``"tokens"``    : projected raw node token + semantic tokens, no HGT message passing

Two optional controls (``token_control``) remove the path information while keeping the
fusion machinery: ``"dummy"`` replaces the semantic tokens with the same number of learned,
input-independent tokens; ``"shuffle"`` hands every target the semantic tokens of another
target in the batch (a fixed permutation at evaluation time).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import HGTConv

from .tokenization import PathInstances, SemanticTokenizer, TokenBatch, impute_missing_features

__all__ = [
    "BaseHGTEncoder",
    "MaskedMultiHeadSelfAttention",
    "FusionTransformerBlock",
    "AttentionFusionTransformer",
    "ClassificationHead",
    "AutoPHGT",
]

MODES = ("auto_phgt", "hgt", "tokens")
TOKEN_CONTROLS = (None, "dummy", "shuffle")


# ---------------------------------------------------------------------------
# 1. Base HGT layer
# ---------------------------------------------------------------------------
class BaseHGTEncoder(nn.Module):
    """A stack of HGT layers (Hu et al., 2020). Each layer is PyG ``HGTConv``, which
    applies type-specific Q/K/V, relation-specific attention and message matrices, and
    a gated skip. We add dropout and a per-type LayerNorm after every layer."""

    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]], d_model: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by the number of HGT heads")
        node_types, _ = metadata
        self.convs = nn.ModuleList([HGTConv(d_model, d_model, metadata, heads=heads)
                                    for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(d_model) for t in node_types})
                                    for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
        for conv, norms in zip(self.convs, self.norms):
            out = conv(h_dict, edge_index_dict)
            # Types that receive no messages (absent from `out` or None) keep their state.
            h_dict = {t: (norms[t](self.dropout(out[t])) if out.get(t) is not None else h)
                      for t, h in h_dict.items()}
        return h_dict


# ---------------------------------------------------------------------------
# 2. Attention Fusion Transformer
# ---------------------------------------------------------------------------
class MaskedMultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention with explicit W_Q, W_K, W_V, W_O projections.

    ``key_padding_mask [B, N]`` (True = ignore) stops every query from attending to
    padded semantic tokens. An optional ``attn_mask`` ([N, N] or [B, N, N], True = block)
    adds structural constraints. Fully masked rows have zero attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        for lin in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _split(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, dh]

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, N, D = x.shape
        q, k, v = self._split(self.W_Q(x)), self._split(self.W_K(x)), self._split(self.W_V(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]

        blocked = torch.zeros(B, 1, N, N, dtype=torch.bool, device=x.device)
        if key_padding_mask is not None:
            blocked = blocked | key_padding_mask.to(torch.bool).view(B, 1, 1, N)
        if attn_mask is not None:
            am = attn_mask.to(torch.bool)
            blocked = blocked | (am.view(1, 1, N, N) if am.dim() == 2 else am.view(B, 1, N, N))

        scores = scores.masked_fill(blocked, float("-inf"))
        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out = self.attn_dropout(attn) @ v                                # [B, H, N, dh]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_O(out), attn


class FusionTransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: x + MHA(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MaskedMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        a, attn = self.attn(self.norm1(x), key_padding_mask, attn_mask)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn


class AttentionFusionTransformer(nn.Module):
    """Self-attention fusion over the ``[node token ; semantic tokens]`` sequence."""

    def __init__(self, d_model: int, num_layers: int = 2, num_heads: int = 4,
                 ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([FusionTransformerBlock(d_model, num_heads, ffn_mult, dropout)
                                     for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        attentions = []
        x = tokens
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask, attn_mask)
            attentions.append(attn)
        return self.final_norm(x), attentions


# ---------------------------------------------------------------------------
# 3. Classification head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear, which outputs logits."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. Full model
# ---------------------------------------------------------------------------
class AutoPHGT(nn.Module):
    """Auto-PHGT: base HGT + automatically discovered semantic tokens + attention fusion.

    ``forward`` returns logits ``[B, C]`` (use with ``nn.CrossEntropyLoss``);
    ``predict_proba`` returns softmax probabilities.

    Full-graph use (e.g. ACM)::

        logits = model(x_dict, edge_index_dict, instances=instances)

    Mini-batch use (e.g. ogbn-mag with PyG ``HGTLoader`` / ``NeighborLoader``): pass the
    sampled subgraph's ``edge_index_dict``, its ``n_id_dict`` (global IDs of every local
    node), and ``target_local`` (local positions of the targets, usually
    ``arange(batch_size)``). ``x_dict`` always holds the *global* features.
    """

    def __init__(self, tokenizer: SemanticTokenizer,
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]], num_classes: int,
                 hgt_layers: int = 2, hgt_heads: int = 4, fusion_layers: int = 2,
                 fusion_heads: int = 4, ffn_mult: int = 2, dropout: float = 0.2,
                 mode: str = "auto_phgt", token_control: Optional[str] = None,
                 num_dummy_tokens: Optional[int] = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if token_control not in TOKEN_CONTROLS:
            raise ValueError(f"token_control must be one of {TOKEN_CONTROLS}")
        if token_control is not None and mode == "hgt":
            raise ValueError("token controls need semantic tokens; mode 'hgt' has none")
        if token_control == "dummy" and not num_dummy_tokens:
            raise ValueError("token_control='dummy' needs num_dummy_tokens (K * I)")
        self.mode = mode
        self.token_control = token_control
        # The HGT baseline retains only the projection and type embedding it uses.
        # In the two token modes the complete tokenizer remains registered.
        self.tokenizer = tokenizer if mode != "hgt" else None
        self.projection = tokenizer.projection  # shared embedding space for HGT and tokens
        self.hgt_type_embedding = tokenizer.type_embedding if mode == "hgt" else None
        self.node_types = tokenizer.node_types
        self.num_nodes = tokenizer.num_nodes
        self.target_node_type = tokenizer.target_node_type
        self.num_classes = num_classes
        d_model = tokenizer.d_model

        self.hgt = (BaseHGTEncoder(metadata, d_model, hgt_layers, hgt_heads, dropout)
                    if mode != "tokens" else None)
        self.fusion = (AttentionFusionTransformer(d_model, fusion_layers, fusion_heads, ffn_mult, dropout)
                       if mode != "hgt" else None)
        self.head = ClassificationHead(d_model, num_classes, dropout)
        self.dummy_tokens = None
        if token_control == "dummy":
            self.dummy_tokens = nn.Parameter(torch.randn(num_dummy_tokens, d_model) * 0.02)

    @classmethod
    def build(cls, graph, templates, num_classes: int, x_dict: Optional[Dict[str, Tensor]] = None,
              d_model: int = 128, pooling: str = "mean", token_dropout: float = 0.1,
              **kwargs) -> "AutoPHGT":
        """Builds tokenizer + model from a PyG ``HeteroData`` and Module 1 templates."""
        if x_dict is None:
            x_dict = impute_missing_features(graph)
        tokenizer = SemanticTokenizer.from_graph(graph, templates, x_dict=x_dict, d_model=d_model,
                                                 pooling=pooling, dropout=token_dropout)
        return cls(tokenizer, graph.metadata(), num_classes, **kwargs)

    # -- graph branch ---------------------------------------------------------
    def encode_graph(self, x_dict: Dict[str, Tensor],
                     edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                     n_id_dict: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """Projects every (sub)graph node with the shared projection and runs the base HGT."""
        type_embedding = (self.hgt_type_embedding if self.mode == "hgt"
                          else self.tokenizer.type_embedding)
        device = type_embedding.weight.device
        h_dict = {}
        for i, t in enumerate(self.node_types):
            ids = (n_id_dict[t] if n_id_dict is not None and t in n_id_dict
                   else torch.arange(self.num_nodes[t], device=device))
            h_dict[t] = self.projection(t, ids, x_dict) + type_embedding.weight[i]
        edge_index_dict = {et: ei.to(device) for et, ei in edge_index_dict.items()}
        return self.hgt(h_dict, edge_index_dict)

    # -- forward --------------------------------------------------------------
    def forward(self, x_dict: Dict[str, Tensor],
                edge_index_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
                instances: Optional[PathInstances] = None, target_ids: Optional[Tensor] = None,
                target_local: Optional[Tensor] = None,
                n_id_dict: Optional[Dict[str, Tensor]] = None,
                return_details: bool = False):
        device = next(self.projection.parameters()).device
        if instances is not None:
            target_ids = instances.target_ids
        if target_ids is None:
            raise ValueError("Provide `instances` (auto_phgt / tokens) or `target_ids` (hgt)")
        if self.mode != "hgt" and instances is None:
            raise ValueError(f"mode '{self.mode}' needs `instances` from MetaPathInstanceExtractor")
        target_ids = target_ids.to(device)

        node_repr = None
        if self.mode != "tokens":
            if edge_index_dict is None:
                raise ValueError(f"mode '{self.mode}' needs `edge_index_dict` for the HGT branch")
            if n_id_dict is not None and target_local is None:
                raise ValueError("`target_local` is required when running on a sampled subgraph")
            h_dict = self.encode_graph(x_dict, edge_index_dict, n_id_dict)
            idx = target_local.to(device) if target_local is not None else target_ids
            node_repr = h_dict[self.target_node_type][idx]

        details: Dict[str, object] = {}
        if self.mode == "hgt":
            logits = self.head(node_repr)
        else:
            tokens: TokenBatch = self._apply_token_control(self.tokenizer(x_dict, instances))
            if node_repr is not None:
                node_repr = node_repr + self.tokenizer.segment_embedding.weight[0]
            seq, key_padding_mask = tokens.sequence(node_repr)
            fused, attentions = self.fusion(seq, key_padding_mask)
            logits = self.head(fused[:, 0])  # read-out at the node-token position
            details = {"tokens": tokens, "attentions": attentions,
                       "key_padding_mask": key_padding_mask}

        if return_details:
            details["logits"] = logits
            return logits, details
        return logits

    def _apply_token_control(self, tokens: TokenBatch) -> TokenBatch:
        if self.token_control is None:
            return tokens
        sem = tokens.semantic_tokens
        B, N, _ = sem.shape
        if self.token_control == "dummy":
            if self.dummy_tokens.size(0) != N:
                raise ValueError(f"{self.dummy_tokens.size(0)} dummy tokens for {N} semantic slots")
            dummy = self.dummy_tokens + self.tokenizer.segment_embedding.weight[1]
            dummy = self.tokenizer.dropout(self.tokenizer.semantic_norm(dummy))
            return replace(tokens, semantic_tokens=dummy.unsqueeze(0).expand(B, -1, -1),
                           semantic_mask=torch.ones(B, N, dtype=torch.bool, device=sem.device))
        # Training uses the global RNG (so checkpoints restore it); evaluation is fixed.
        perm = (torch.randperm(B) if self.training else
                torch.randperm(B, generator=torch.Generator().manual_seed(1))).to(sem.device)
        return replace(tokens, semantic_tokens=sem[perm],
                       semantic_mask=tokens.semantic_mask[perm])

    def predict_proba(self, *args, **kwargs) -> Tensor:
        """Evaluate class probabilities and restore the prior training mode."""
        kwargs.pop("return_details", None)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return F.softmax(self.forward(*args, **kwargs), dim=-1)
        finally:
            self.train(was_training)

    @staticmethod
    def template_attention(details: Dict[str, object], num_templates: int, layer: int = -1) -> Tensor:
        """Share of the node token's attention that goes to each meta-path template.

        Returns ``[B, K + 1]``: column 0 is self-attention to the node token and column
        k + 1 is template k (summed over its instances, averaged over heads). This
        shows which discovered meta-paths the model relies on.
        """
        attn = details["attentions"][layer].mean(1)[:, 0]          # [B, N]
        tokens: TokenBatch = details["tokens"]
        tid = tokens.semantic_template_ids
        out = attn.new_zeros(attn.size(0), num_templates + 1)
        out[:, 0] = attn[:, 0]
        out[:, 1:].index_add_(1, tid, attn[:, 1:])
        return out

    def num_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0
        tok_total = count(self.tokenizer)
        return {"projection": count(self.projection),
                 "tokenizer (excl. projection)": (tok_total - count(self.projection)
                                                if self.tokenizer is not None else 0),
                "hgt type embedding": count(self.hgt_type_embedding),
                "hgt": count(self.hgt), "fusion": count(self.fusion), "head": count(self.head),
                "dummy tokens": (self.dummy_tokens.numel()
                                 if self.dummy_tokens is not None else 0),
                "total": count(self)}
