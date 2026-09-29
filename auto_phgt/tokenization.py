"""
Module 2: Semantic Tokenization (Auto-PHGT)
============================================

Turns the top-k meta-path templates discovered by Module 1
(`StatisticalDiscoveryModule`) into fixed-dimension tensor tokens.

    raw graph + top-k templates
      -> MetaPathInstanceExtractor  local path-instance extraction (vectorised CSR walks)
      -> TypeAwareProjection        per-node-type linear projection matrix W_t : R^{d_t} -> R^{d}
      -> SemanticTokenizer          1 node token + 1 semantic token per path instance
      -> TokenBatch                 tensors + boolean masks consumed by Module 3

Shape glossary
--------------
B  batch size (number of target nodes)
K  number of meta-path templates (top-k)
I  sampled instances per template
L  max number of node positions in a template (= max hops + 1)
D  shared transformer embedding dimension (d_model)

Padding convention (same as Module 1): a missing node is the ID ``-1`` (``PAD``).
Boolean masks in this module are ``True`` for VALID entries. `TokenBatch.sequence()`
additionally returns a ``key_padding_mask`` in the PyTorch convention (``True`` = ignore).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

PAD = -1
EdgeType = Tuple[str, str, str]

__all__ = [
    "PAD",
    "MetaPathTemplate",
    "load_metapath_templates",
    "discover_or_load_templates",
    "impute_missing_features",
    "PathInstances",
    "MetaPathInstanceExtractor",
    "instance_statistics",
    "TypeAwareProjection",
    "TokenBatch",
    "SemanticTokenizer",
    "make_token_dataloader",
]


# ---------------------------------------------------------------------------
# 1. Meta-path templates (Module 1 hand-off format)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaPathTemplate:
    """A meta-path in Module 1's flat format: [type, rel, type, rel, ..., type]."""

    schema: Tuple[str, ...]

    @classmethod
    def from_list(cls, schema: Sequence[str]) -> "MetaPathTemplate":
        schema = tuple(schema)
        if len(schema) < 3 or len(schema) % 2 == 0:
            raise ValueError(
                f"Invalid meta-path {list(schema)}: expected [type, rel, type, ..., type]"
            )
        return cls(schema)

    @property
    def node_types(self) -> Tuple[str, ...]:
        return self.schema[0::2]

    @property
    def edge_types(self) -> List[EdgeType]:
        s = self.schema
        return [(s[i], s[i + 1], s[i + 2]) for i in range(0, len(s) - 2, 2)]

    @property
    def num_hops(self) -> int:
        return len(self.schema) // 2

    @property
    def length(self) -> int:
        """Number of node positions (hops + 1)."""
        return self.num_hops + 1

    @property
    def name(self) -> str:
        return " -> ".join(self.schema)

    def validate(self, graph, target_node_type: Optional[str] = None) -> None:
        if target_node_type is not None and self.node_types[0] != target_node_type:
            raise ValueError(f"Template '{self.name}' does not start at '{target_node_type}'")
        known = set(graph.edge_types)
        for et in self.edge_types:
            if et not in known:
                raise ValueError(f"Template '{self.name}' uses unknown edge type {et}")


TemplateLike = Union[MetaPathTemplate, Sequence[str]]


def _as_templates(templates: Sequence[TemplateLike]) -> List[MetaPathTemplate]:
    out = [t if isinstance(t, MetaPathTemplate) else MetaPathTemplate.from_list(t) for t in templates]
    if not out:
        raise ValueError("At least one meta-path template is required")
    return out


def load_metapath_templates(source: Sequence[TemplateLike]) -> List[MetaPathTemplate]:
    """Convert already validated schema lists into typed templates."""
    return _as_templates(source)


def discover_or_load_templates(graph, *, dataset: str, path=None, k: int = 5,
                               target_node_type: str = "paper", max_hops: int = 4,
                               overwrite: bool = False) -> List[MetaPathTemplate]:
    """Load dataset/config-specific discovery output or run canonical discovery."""
    from .discovery import discover_or_load_paths

    raw = discover_or_load_paths(
        graph, dataset=dataset, target_node_type=target_node_type,
        max_hops=max_hops, k=k, path=path, overwrite=overwrite,
    )
    templates = load_metapath_templates(raw)
    for t in templates:
        t.validate(graph, target_node_type)
        if t.node_types[-1] != target_node_type:
            raise ValueError(f"Discovered template '{t.name}' does not end at '{target_node_type}'")
    return templates


# ---------------------------------------------------------------------------
# 2. Feature preparation for feature-less node types
# ---------------------------------------------------------------------------
@torch.no_grad()
def impute_missing_features(graph, max_rounds: int = 4,
                            chunk_elements: int = 1 << 26) -> Dict[str, Tensor]:
    """Returns an ``x_dict`` in which feature-less node types get the mean of their
    featured neighbours (e.g. ACM ``term`` <- mean of its papers).

    For each missing type the incoming relation with the most edges from an already
    featured type is used. Types that stay unreachable are left out of the dict and
    are handled by a learnable ``nn.Embedding`` in `TypeAwareProjection`.
    Aggregation is chunked so memory stays bounded on large graphs such as ogbn-mag.
    """
    x_dict: Dict[str, Tensor] = {}
    for t in graph.node_types:
        x = getattr(graph[t], "x", None)
        if x is not None:
            x_dict[t] = x

    for _ in range(max_rounds):
        missing = [t for t in graph.node_types if t not in x_dict]
        if not missing:
            break
        progress = False
        for t in missing:
            cands = [et for et in graph.edge_types
                     if et[2] == t and et[0] != t and et[0] in x_dict and graph[et].num_edges > 0]
            if not cands:
                continue
            et = max(cands, key=lambda e: graph[e].num_edges)
            src, dst = graph[et].edge_index
            xs = x_dict[et[0]]
            n = graph[t].num_nodes
            out = xs.new_zeros(n, xs.size(1))
            step = max(1, chunk_elements // max(1, xs.size(1)))
            for i in range(0, src.numel(), step):
                out.index_add_(0, dst[i:i + step], xs[src[i:i + step]])
            deg = torch.bincount(dst, minlength=n).clamp(min=1).unsqueeze(1).to(xs.dtype)
            x_dict[t] = out / deg
            progress = True
        if not progress:
            break
    return x_dict


# ---------------------------------------------------------------------------
# 3. Local path-instance extraction
# ---------------------------------------------------------------------------
@dataclass
class PathInstances:
    """Sampled meta-path instances for a batch of target nodes.

    node_ids : [B, K, I, L] long   global node IDs per position, ``PAD`` (-1) if missing
    type_ids : [K, L] long         node-type index (into ``node_types``) per position, ``PAD`` past the template end
    lengths  : [K] long            number of node positions of each template
    """

    target_ids: Tensor
    node_ids: Tensor
    type_ids: Tensor
    lengths: Tensor
    node_types: Tuple[str, ...]
    schemas: Tuple[Tuple[str, ...], ...]

    @property
    def valid_mask(self) -> Tensor:
        """[B, K, I, L] True where a real node is present."""
        return self.node_ids != PAD

    @property
    def complete_mask(self) -> Tensor:
        """[B, K, I] True where the walk reached the template's final node."""
        B, K, I, _ = self.node_ids.shape
        last = (self.lengths.to(self.node_ids.device) - 1).view(1, K, 1, 1).expand(B, K, I, 1)
        return self.node_ids.gather(-1, last).squeeze(-1) != PAD

    def to(self, device) -> "PathInstances":
        return PathInstances(self.target_ids.to(device), self.node_ids.to(device),
                             self.type_ids.to(device), self.lengths.to(device),
                             self.node_types, self.schemas)

    def __repr__(self) -> str:
        B, K, I, L = self.node_ids.shape
        return (f"PathInstances(B={B}, K={K}, I={I}, L={L}, "
                f"complete={self.complete_mask.float().mean().item():.3f})")


def _to_csr(edge_index: Tensor, num_src: int) -> Tuple[Tensor, Tensor]:
    src, dst = edge_index[0].long(), edge_index[1].long()
    perm = torch.argsort(src, stable=True)
    col = dst[perm].contiguous()
    rowptr = torch.zeros(num_src + 1, dtype=torch.long)
    rowptr[1:] = torch.bincount(src, minlength=num_src).cumsum(0)
    return rowptr, col


def _template_type_ids(templates: List[MetaPathTemplate], node_types: Sequence[str]) -> Tensor:
    index = {t: i for i, t in enumerate(node_types)}
    L = max(t.length for t in templates)
    type_ids = torch.full((len(templates), L), PAD, dtype=torch.long)
    for k, tpl in enumerate(templates):
        for l, nt in enumerate(tpl.node_types):
            type_ids[k, l] = index[nt]
    return type_ids


class MetaPathInstanceExtractor:
    """Vectorised, type-consistent random walks along each meta-path template.

    Uses CSR adjacency per relation, like the standalone Module 1 sampler, but
    samples whole batches with tensor operations. Dead ends remain ``PAD`` for
    every later position. A seeded ``torch.Generator`` makes draws repeatable.

    Sampling is uniform over neighbours, as in standalone Module 1.
    """

    def __init__(self, graph, templates: Sequence[TemplateLike], instances_per_path: int = 4,
                 target_node_type: Optional[str] = None, seed: int = 0):
        self.templates = _as_templates(templates)
        self.target_node_type = target_node_type or self.templates[0].node_types[0]
        for t in self.templates:
            t.validate(graph, self.target_node_type)
        if instances_per_path < 1:
            raise ValueError("instances_per_path must be >= 1")

        self.instances_per_path = instances_per_path
        self.node_types: Tuple[str, ...] = tuple(graph.node_types)
        self.num_target_nodes = graph[self.target_node_type].num_nodes
        self.type_ids = _template_type_ids(self.templates, self.node_types)
        self.lengths = torch.tensor([t.length for t in self.templates], dtype=torch.long)
        self.max_len = int(self.lengths.max())
        self.csr: Dict[EdgeType, Tuple[Tensor, Tensor]] = {}
        for et in {et for t in self.templates for et in t.edge_types}:
            self.csr[et] = _to_csr(graph[et].edge_index, graph[et[0]].num_nodes)
        self.generator = torch.Generator().manual_seed(seed)

    @property
    def num_templates(self) -> int:
        return len(self.templates)

    @torch.no_grad()
    def sample(self, target_ids, seed: Optional[int] = None) -> PathInstances:
        """Samples ``instances_per_path`` instances of every template for each target.

        Pass ``seed`` for a fixed draw (e.g. evaluation); otherwise the extractor's
        internal generator advances, giving fresh instances every epoch.
        """
        target_ids = torch.as_tensor(target_ids, dtype=torch.long).view(-1).cpu()
        if target_ids.numel() and (target_ids.min() < 0 or target_ids.max() >= self.num_target_nodes):
            raise IndexError(f"target ids must lie in [0, {self.num_target_nodes})")
        gen = torch.Generator().manual_seed(seed) if seed is not None else self.generator

        B, K, I, L = target_ids.numel(), self.num_templates, self.instances_per_path, self.max_len
        node_ids = torch.full((B, K, I, L), PAD, dtype=torch.long)
        start = target_ids.repeat_interleave(I)  # [B*I], b-major
        for k, tpl in enumerate(self.templates):
            cur = start.clone()
            node_ids[:, k, :, 0] = cur.view(B, I)
            for h, et in enumerate(tpl.edge_types):
                rowptr, col = self.csr[et]
                if col.numel() == 0:
                    cur = torch.full_like(cur, PAD)
                else:
                    alive = cur != PAD
                    c = cur.clamp(min=0)
                    begin = rowptr[c]
                    deg = rowptr[c + 1] - begin
                    ok = alive & (deg > 0)
                    offset = (torch.rand(cur.numel(), generator=gen) * deg).long()
                    offset = torch.minimum(offset, (deg - 1).clamp(min=0))
                    nxt = col[torch.where(ok, begin + offset, torch.zeros_like(begin))]
                    cur = torch.where(ok, nxt, torch.full_like(cur, PAD))
                node_ids[:, k, :, h + 1] = cur.view(B, I)
        return PathInstances(target_ids, node_ids, self.type_ids.clone(), self.lengths.clone(),
                             self.node_types, tuple(t.schema for t in self.templates))


def instance_statistics(instances: PathInstances,
                        templates: Sequence[TemplateLike]) -> List[Dict[str, float]]:
    """Per-template diagnostics: completion rate, mean valid length, end-node diversity."""
    templates = _as_templates(templates)
    complete = instances.complete_mask  # [B, K, I]
    valid = instances.valid_mask
    stats = []
    for k, tpl in enumerate(templates):
        ends = instances.node_ids[:, k, :, tpl.length - 1]
        ends = ends[ends != PAD]
        stats.append({
            "template": tpl.name,
            "complete_rate": complete[:, k].float().mean().item(),
            "mean_valid_positions": valid[:, k, :, :tpl.length].sum(-1).float().mean().item(),
            "targets_with_any_instance": complete[:, k].any(-1).float().mean().item(),
            "unique_end_node_ratio": (ends.unique().numel() / max(1, ends.numel())),
        })
    return stats


# ---------------------------------------------------------------------------
# 4. Linear projection matrix layer
# ---------------------------------------------------------------------------
class TypeAwareProjection(nn.Module):
    """Per-node-type projection into the shared embedding space.

    ``h = W_t x + b_t`` for types with features, and a learnable ``nn.Embedding``
    for types without features. It only projects the rows it is asked for, so
    ogbn-mag's full feature matrices never pass through the layer, and the feature
    matrices may stay on CPU while the layer runs on GPU.
    """

    def __init__(self, in_dims: Dict[str, Optional[int]], num_nodes: Dict[str, int],
                 d_model: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.node_types = list(in_dims.keys())
        self.d_model = d_model
        self.linears = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        for t, d in in_dims.items():
            if d is None:
                emb = nn.Embedding(num_nodes[t], d_model)
                nn.init.normal_(emb.weight, std=0.02)
                self.embeddings[t] = emb
            else:
                lin = nn.Linear(d, d_model, bias=bias)
                nn.init.xavier_uniform_(lin.weight)
                if bias:
                    nn.init.zeros_(lin.bias)
                self.linears[t] = lin
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_type: str, ids: Tensor, x_dict: Dict[str, Tensor]) -> Tensor:
        """ids: any-shaped tensor of non-negative global IDs -> [*ids.shape, d_model]."""
        device = next(self.parameters()).device
        if node_type in self.linears:
            x = x_dict[node_type]
            feats = x[ids.to(x.device)].to(device=device, dtype=torch.float32, non_blocking=True)
            out = self.linears[node_type](feats)
        elif node_type in self.embeddings:
            out = self.embeddings[node_type](ids.to(device))
        else:
            raise KeyError(f"Unknown node type '{node_type}'")
        return self.dropout(out)


# ---------------------------------------------------------------------------
# 5. Semantic tokenizer
# ---------------------------------------------------------------------------
@dataclass
class TokenBatch:
    """Output of `SemanticTokenizer`: the hand-off to the Model Lead (Module 3).

    node_tokens           [B, D]          target-node token
    semantic_tokens       [B, K*I, D]     one token per sampled meta-path instance
    semantic_mask         [B, K*I] bool   True = valid instance (padded tokens are zero)
    semantic_template_ids [K*I] long      template index of every semantic token
    path_tokens           [B, K, I, L, D] per-position tokens (for intra-path attention)
    path_mask             [B, K, I, L]    True = real node
    """

    node_tokens: Tensor
    semantic_tokens: Tensor
    semantic_mask: Tensor
    semantic_template_ids: Tensor
    path_tokens: Tensor
    path_mask: Tensor

    def sequence(self, node_tokens: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Concatenates [node token ; semantic tokens] for the fusion transformer.

        Returns ``tokens [B, 1+K*I, D]`` and ``key_padding_mask [B, 1+K*I]``
        (True = ignore, PyTorch convention). Position 0 is always the node token and is
        never masked, so no attention row is fully masked.
        """
        node = self.node_tokens if node_tokens is None else node_tokens
        tokens = torch.cat([node.unsqueeze(1), self.semantic_tokens], dim=1)
        keep = torch.cat([torch.ones_like(self.semantic_mask[:, :1]), self.semantic_mask], dim=1)
        return tokens, ~keep


class SemanticTokenizer(nn.Module):
    """Projects target nodes and their meta-path instances into ``d_model`` tokens.

    Each path position gets ``W_{type} x_node + E_type[type] + E_hop[position]``.
    The positions of one instance are pooled (masked mean / max / attention) and
    ``E_template[k]`` is added, giving one semantic token per instance, as PHGT does.
    """

    POOLINGS = ("mean", "max", "attention")

    def __init__(self, node_types: Sequence[str], in_dims: Dict[str, Optional[int]],
                 num_nodes: Dict[str, int], templates: Sequence[TemplateLike],
                 d_model: int = 128, target_node_type: Optional[str] = None,
                 pooling: str = "mean", dropout: float = 0.1, require_complete: bool = True,
                 projection: Optional[TypeAwareProjection] = None):
        super().__init__()
        if pooling not in self.POOLINGS:
            raise ValueError(f"pooling must be one of {self.POOLINGS}")
        self.templates = _as_templates(templates)
        self.node_types: Tuple[str, ...] = tuple(node_types)
        self.target_node_type = target_node_type or self.templates[0].node_types[0]
        self.d_model = d_model
        self.pooling = pooling
        self.require_complete = require_complete
        self.num_nodes = dict(num_nodes)

        self.projection = projection or TypeAwareProjection(
            {t: in_dims.get(t) for t in self.node_types}, num_nodes, d_model)
        max_len = max(t.length for t in self.templates)
        self.type_embedding = nn.Embedding(len(self.node_types), d_model)
        self.hop_embedding = nn.Embedding(max_len, d_model)
        self.template_embedding = nn.Embedding(len(self.templates), d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 0 = node token, 1 = semantic token
        for emb in (self.type_embedding, self.hop_embedding, self.template_embedding,
                    self.segment_embedding):
            nn.init.normal_(emb.weight, std=0.02)
        if pooling == "attention":
            self.pool_score = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(),
                                            nn.Linear(d_model, 1, bias=False))
        self.node_norm = nn.LayerNorm(d_model)
        self.semantic_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("type_ids", _template_type_ids(self.templates, self.node_types),
                             persistent=False)

    @classmethod
    def from_graph(cls, graph, templates: Sequence[TemplateLike],
                   x_dict: Optional[Dict[str, Tensor]] = None, **kwargs) -> "SemanticTokenizer":
        """Infers node types, feature dims and node counts from a PyG ``HeteroData``."""
        if x_dict is None:
            x_dict = impute_missing_features(graph)
        in_dims = {t: (x_dict[t].size(-1) if t in x_dict else None) for t in graph.node_types}
        num_nodes = {t: graph[t].num_nodes for t in graph.node_types}
        return cls(graph.node_types, in_dims, num_nodes, templates, **kwargs)

    @property
    def target_type_index(self) -> int:
        return self.node_types.index(self.target_node_type)

    def _check(self, instances: PathInstances) -> None:
        if tuple(instances.node_types) != self.node_types:
            raise ValueError("PathInstances were built with a different node-type vocabulary")
        if instances.node_ids.size(1) != len(self.templates):
            raise ValueError("PathInstances and tokenizer disagree on the number of templates")
        if instances.schemas != tuple(t.schema for t in self.templates):
            raise ValueError("PathInstances and tokenizer use different ordered templates")

    def embed_node(self, x_dict: Dict[str, Tensor], target_ids: Tensor) -> Tensor:
        """[B] -> [B, D] target-node tokens."""
        h = self.projection(self.target_node_type, target_ids, x_dict)
        h = h + self.type_embedding.weight[self.target_type_index] + self.segment_embedding.weight[0]
        return self.dropout(self.node_norm(h))

    def embed_paths(self, x_dict: Dict[str, Tensor], instances: PathInstances) -> Tuple[Tensor, Tensor]:
        """Returns ``path_tokens [B, K, I, L, D]`` and ``path_mask [B, K, I, L]``."""
        device = self.type_embedding.weight.device
        node_ids = instances.node_ids.to(device)
        B, K, I, L = node_ids.shape
        mask = node_ids != PAD
        safe = node_ids.clamp(min=0)
        type_ids = self.type_ids[:, :L]

        tokens = torch.zeros(B, K, I, L, self.d_model, device=device,
                             dtype=self.type_embedding.weight.dtype)
        for ti, t in enumerate(self.node_types):
            kk, ll = (type_ids == ti).nonzero(as_tuple=True)
            if kk.numel() == 0:
                continue
            ids = safe[:, kk, :, ll]                                # [P, B, I]
            tokens[:, kk, :, ll] = self.projection(t, ids, x_dict)  # [P, B, I, D]

        pos = self.type_embedding(type_ids.clamp(min=0)) + self.hop_embedding.weight[:L].unsqueeze(0)
        tokens = (tokens + pos[None, :, None]) * mask.unsqueeze(-1)
        return tokens, mask

    def pool(self, path_tokens: Tensor, path_mask: Tensor) -> Tensor:
        """Masked pooling over the L positions: [B, K, I, L, D] -> [B, K, I, D]."""
        m = path_mask.unsqueeze(-1)
        if self.pooling == "mean":
            return (path_tokens * m).sum(-2) / m.sum(-2).clamp(min=1)
        if self.pooling == "max":
            pooled = path_tokens.masked_fill(~m, float("-inf")).amax(-2)
            return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        scores = self.pool_score(path_tokens).squeeze(-1).masked_fill(~path_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        return (weights.unsqueeze(-1) * path_tokens).sum(-2)

    def forward(self, x_dict: Dict[str, Tensor], instances: PathInstances) -> TokenBatch:
        self._check(instances)
        device = self.type_embedding.weight.device
        path_tokens, path_mask = self.embed_paths(x_dict, instances)
        B, K, I, L, D = path_tokens.shape

        if self.require_complete:
            sem_mask = instances.complete_mask.to(device)
        else:
            sem_mask = path_mask[..., 1:].any(-1)  # at least one hop beyond the target
        sem = self.pool(path_tokens, path_mask)
        sem = sem + self.template_embedding.weight.view(1, K, 1, D) + self.segment_embedding.weight[1]
        sem = self.dropout(self.semantic_norm(sem)) * sem_mask.unsqueeze(-1)

        template_ids = torch.arange(K, device=device).repeat_interleave(I)
        return TokenBatch(
            node_tokens=self.embed_node(x_dict, instances.target_ids.to(device)),
            semantic_tokens=sem.reshape(B, K * I, D),
            semantic_mask=sem_mask.reshape(B, K * I),
            semantic_template_ids=template_ids,
            path_tokens=path_tokens,
            path_mask=path_mask,
        )


# ---------------------------------------------------------------------------
# 6. Batch data pipeline
# ---------------------------------------------------------------------------
def make_token_dataloader(target_ids, extractor: MetaPathInstanceExtractor, batch_size: int = 256,
                          labels: Optional[Tensor] = None, shuffle: bool = False,
                          seed: Optional[int] = None, fixed_instances: bool = False):
    """Batches target nodes and samples their meta-path instances in ``collate_fn``.

    ``labels`` is the full label vector of the target node type, indexed by global ID
    (e.g. ``graph['paper'].y``). Each batch is a dict with ``target_ids [B]``,
    ``instances`` (`PathInstances`) and, when labels are given, ``y = labels[target_ids]``. Training draws fresh instances every epoch
    (stochastic augmentation). With ``fixed_instances=True`` each batch reuses the
    same seeded draw, which gives deterministic evaluation.
    """
    from torch.utils.data import DataLoader, TensorDataset

    target_ids = torch.as_tensor(target_ids, dtype=torch.long).view(-1)
    if labels is not None:
        labels = torch.as_tensor(labels)
        if target_ids.numel() and int(target_ids.max()) >= labels.size(0):
            raise IndexError("labels must be indexed by global node ID (e.g. graph['paper'].y)")
    base_seed = 0 if seed is None else seed

    def collate(items):
        ids = torch.stack([it[0] for it in items])
        # Deterministic per-batch seed derived from the ids, so the draw is independent of batch order.
        inst_seed = (base_seed * 1_000_003 + int(ids.sum()) * 31 + ids.numel()) if fixed_instances else None
        batch = {"target_ids": ids, "instances": extractor.sample(ids, seed=inst_seed)}
        if labels is not None:
            batch["y"] = labels[ids]
        return batch

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(TensorDataset(target_ids), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=gen)
