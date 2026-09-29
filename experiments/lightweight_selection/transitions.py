"""Sparse, memory-safe path transition statistics for V4-Hedge.

For a meta-path p = (r_1, ..., r_h), P_r is the row-normalised adjacency of relation r and
T_p = P_r1 @ ... @ P_rh. Rows of T_p are computed chunk-wise from sparse products (a chunk is
densified only while more than ``dense_fraction`` of it is filled), so no dense N x N matrix
is ever built. Row u of T_p, renormalised, is the endpoint distribution of complete uniform
random walks from u, i.e. of the Auto-PHGT instance sampler (``MetaPathInstanceExtractor``).

All statistics come from one pass over the raw rows: with c = sum_v x_v, the self entry s,
m = c - s and S = sum_v x_v log x_v, the self-excluded row entropy is
log m - (S - s log s) / m and the self-inclusive one log c - S / c.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

DTYPE = np.float32


def edges_of(schema) -> list[tuple[str, str, str]]:
    return [tuple(schema[i:i + 3]) for i in range(0, len(schema) - 2, 2)]


def relation_operators(graph, edge_types=None) -> dict:
    """Row-normalised CSR operator per relation (duplicate edges counted, empty rows zero)."""
    ops = {}
    for et in edge_types or graph.edge_types:
        src, _, dst = et
        ei = graph[et].edge_index.cpu().numpy()
        a = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])),
                          shape=(graph[src].num_nodes, graph[dst].num_nodes))
        deg = np.asarray(a.sum(1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
        ops[tuple(et)] = (sp.diags(inv) @ a).tocsr().astype(DTYPE)
    return ops


# --------------------------------------------------------------------------- row algebra
def _density_switch(x, dense_fraction: float):
    size = x.shape[0] * x.shape[1]
    if sp.issparse(x):
        return x.toarray() if x.nnz > dense_fraction * size else x
    if np.count_nonzero(x) < 0.5 * dense_fraction * size:
        return sp.csr_matrix(x)
    return x


def step(x, op, dense_fraction: float):
    """x @ op for a sparse or dense row block, switching representation by fill."""
    out = (x @ op).tocsr() if sp.issparse(x) else np.asarray((op.T @ x.T).T, dtype=DTYPE)
    return _density_switch(out, dense_fraction)


def chain_rows(ops, schema, rows, dense_fraction: float = 0.1):
    """Raw (sub-stochastic) rows ``rows`` of T_p for the path ``schema``."""
    edges = edges_of(schema)
    x = _density_switch(ops[edges[0]][np.asarray(rows)], dense_fraction)
    for et in edges[1:]:
        x = step(x, ops[et], dense_fraction)
    return x


def row_sums(x) -> np.ndarray:
    if sp.issparse(x):
        rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        return np.bincount(rows, weights=x.data.astype(np.float64), minlength=x.shape[0])
    return x.sum(1, dtype=np.float64)


def entries(x, cols) -> np.ndarray:
    """x[i, cols[i]] for every row i (float values of the block's dtype)."""
    idx = np.arange(x.shape[0])
    if sp.issparse(x):
        return np.asarray(x[idx, cols]).ravel().astype(x.dtype)
    return x[idx, cols]


def xlogx(v):
    v = np.asarray(v)
    safe = np.where(v > 0, v, np.ones_like(v))
    return np.where(v > 0, v * np.log(safe), np.zeros_like(v))


def row_xlogx(x) -> np.ndarray:
    if sp.issparse(x):
        rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        return np.bincount(rows, weights=xlogx(x.data).astype(np.float64),
                           minlength=x.shape[0])
    return xlogx(x).sum(1, dtype=np.float64)


def weighted_col_sum(x, w) -> np.ndarray:
    """sum_i w_i x_i (float64)."""
    if sp.issparse(x):
        return np.asarray(x.T.astype(np.float64) @ w).ravel()
    return w @ x.astype(np.float64, copy=False)


def entropy(p) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def gaussian_projection(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)) / np.sqrt(d)


def block_stats(x, sources, *, exclude_self: bool, projection=None):
    """One-pass statistics of a raw row block whose row i starts at node ``sources[i]``.

    Returns per-row mass c, non-self mass m, self entry s, self-excluded and self-inclusive
    entropies, the (weighted) column sums of the normalised rows, and optionally the rows
    of normalised(x) @ projection (self entry excluded when ``exclude_self``).
    """
    c = row_sums(x)
    s = entries(x, sources).astype(np.float64) if exclude_self else np.zeros(x.shape[0])
    total = row_xlogx(x)
    e_s = xlogx(entries(x, sources)).astype(np.float64) if exclude_self else 0.0
    m = c - s
    any_, cov = c > 0, m > 0
    safe_c, safe_m = np.where(any_, c, 1.0), np.where(cov, m, 1.0)
    h_in = np.where(any_, np.log(safe_c) - total / safe_c, 0.0)
    h_ex = np.where(cov, np.log(safe_m) - (total - e_s) / safe_m, 0.0)
    w_in = np.where(any_, 1.0 / safe_c, 0.0)
    w_ex = np.where(cov, 1.0 / safe_m, 0.0)
    q_in = weighted_col_sum(x, w_in)
    q_ex = weighted_col_sum(x, w_ex)
    if exclude_self:
        np.subtract.at(q_ex, sources[cov], s[cov] * w_ex[cov])
    out = {"c": c, "m": m, "s": s, "any": any_, "cov": cov, "h_in": np.clip(h_in, 0, None),
           "h_ex": np.clip(h_ex, 0, None), "q_in": q_in, "q_ex": q_ex, "w_ex": w_ex}
    if projection is not None:
        xr = np.asarray(x @ projection, dtype=np.float64)
        if exclude_self:
            xr -= s[:, None] * projection[sources]
        out["proj"] = xr * w_ex[:, None]
    return out


# --------------------------------------------------------------------------- Transition-Info
def ti_scores(c: float, info: float, hq: float, eps: float = 1e-12) -> dict:
    return {"TI_raw": info, "TI_cov": c * info, "TI_norm": c * info / (hq + eps)}


def transition_info(ops, schema, n_target: int, *, chunk: int = 512,
                    dense_fraction: float = 0.1, projection=None, eps: float = 1e-12):
    """Exact Transition-Info statistics of one target-to-target path over ALL source nodes.

    Self-excluded (primary): T(.|u) without the source's own entry, renormalised; C = share of
    sources with any non-self endpoint; q = mean covered row; I = H(q) - mean_u H(T(.|u)),
    which equals mean_u KL(T(.|u) || q). Self-inclusive values are returned as diagnostics.
    With ``projection`` [n_target, d] the centred source-conditioned fingerprint
    (T(.|u) - q) @ projection of every covered source (zero rows elsewhere) is returned too.
    """
    t0 = time.perf_counter()
    q_ex, q_in = np.zeros(n_target), np.zeros(n_target)
    h_ex = h_in = self_share = completion = 0.0
    n_cov = n_any = max_nnz = 0
    covered = np.zeros(n_target, dtype=bool)
    fp = np.zeros((n_target, projection.shape[1])) if projection is not None else None
    for start in range(0, n_target, chunk):
        rows = np.arange(start, min(start + chunk, n_target))
        x = chain_rows(ops, schema, rows, dense_fraction)
        max_nnz = max(max_nnz, x.nnz if sp.issparse(x) else int(np.count_nonzero(x)))
        st = block_stats(x, rows, exclude_self=True, projection=projection)
        completion += float(st["c"].sum())
        any_, cov = st["any"], st["cov"]
        self_share += float((st["s"][any_] / st["c"][any_]).sum())
        h_in += float(st["h_in"][any_].sum())
        h_ex += float(st["h_ex"][cov].sum())
        q_in += st["q_in"]
        q_ex += st["q_ex"]
        n_any += int(any_.sum())
        n_cov += int(cov.sum())
        covered[rows[cov]] = True
        if fp is not None:
            fp[rows[cov]] = st["proj"][cov]
    out = {"C": n_cov / n_target, "C_any": n_any / n_target,
           "completion_rate": completion / n_target,
           "self_return_share": self_share / n_any if n_any else 0.0,
           "covered_sources": n_cov, "max_chunk_nnz": max_nnz}
    if n_cov:
        q = q_ex / n_cov
        hq = entropy(q)
        out.update(H_q=hq, mean_row_entropy=h_ex / n_cov, I=max(hq - h_ex / n_cov, 0.0),
                   effective_endpoints=float(np.exp(hq)))
        if fp is not None:
            fp[covered] -= q @ projection
    else:
        out.update(H_q=0.0, mean_row_entropy=0.0, I=0.0, effective_endpoints=0.0)
    if n_any:
        qi = q_in / n_any
        out.update(I_incl_self=max(entropy(qi) - h_in / n_any, 0.0), H_q_incl_self=entropy(qi))
    else:
        out.update(I_incl_self=0.0, H_q_incl_self=0.0)
    out.update(ti_scores(out["C"], out["I"], out["H_q"], eps))
    out["runtime_s"] = time.perf_counter() - t0
    return out, (fp.astype(np.float32) if fp is not None else None)


# --------------------------------------------------------------------------- FastPath blocks
def train_block(ops, schema, train_ids, *, chunk: int = 512, dense_fraction: float = 0.1):
    """T_p restricted to training rows and columns (rows normalised over ALL endpoints of
    complete walks before the restriction), with the diagonal (self-return) removed.

    Returns a dense float32 [n, n] block in the order of ``train_ids`` and the completion
    probability of every training row.
    """
    train_ids = np.asarray(train_ids)
    n = train_ids.size
    block = np.zeros((n, n), dtype=np.float32)
    completion = np.zeros(n)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        x = chain_rows(ops, schema, train_ids[sl], dense_fraction)
        c = row_sums(x)
        completion[sl] = c
        inv = np.where(c > 0, 1.0 / np.where(c > 0, c, 1.0), 0.0).astype(np.float32)
        sub = x[:, train_ids]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        block[sl] = sub * inv[:, None]
    np.fill_diagonal(block, 0.0)
    return block, completion


# --------------------------------------------------------------------------- TI-Beam
def schema_out(graph) -> dict:
    out = {}
    for src, rel, dst in graph.edge_types:
        out.setdefault(src, []).append((rel, dst))
    return out


def reach_within(graph, target: str, lmax: int) -> dict:
    """reach[t][r]: can node type t reach ``target`` in exactly r more hops?"""
    types = list(graph.node_types)
    nxt = schema_out(graph)
    reach = {t: {0: t == target} for t in types}
    for r in range(1, lmax + 1):
        for t in types:
            reach[t][r] = any(reach[d][r - 1] for _, d in nxt.get(t, []))
    return reach


def can_complete(reach, end_type: str, remaining: int) -> bool:
    return any(reach[end_type].get(r, False) for r in range(1, remaining + 1))


def prefix_stats(x, sources, is_target_end: bool, projection, variant: str, eps: float):
    """Transition statistics of a prefix block on sampled source rows."""
    st = block_stats(x, sources, exclude_self=is_target_end, projection=projection)
    cov = st["cov"]
    n = x.shape[0]
    res = {"coverage": float(cov.mean()) if n else 0.0}
    if not cov.any():
        res.update(I=0.0, H_q=0.0, effective_endpoints=0.0, score=0.0)
        return res, np.zeros(n * projection.shape[1])
    q = st["q_ex"] / cov.sum()
    hq = entropy(q)
    info = max(hq - float(st["h_ex"][cov].sum()) / cov.sum(), 0.0)
    res.update(I=info, H_q=hq, effective_endpoints=float(np.exp(hq)))
    res["score"] = ti_scores(res["coverage"], info, hq, eps)[variant]
    fp = np.where(cov[:, None], st["proj"] - q @ projection, 0.0)
    return res, fp.ravel()


def beam_search(graph, ops, target: str, *, variant: str, width: int, lmax: int,
                sample_sources: int, sample_seed: int, min_coverage: float,
                min_effective_endpoints: float, duplicate_cosine: float, fingerprint_dim: int,
                min_hops: int = 2, eps: float = 1e-12, dense_fraction: float = 0.1, **_):
    """Transition-guided discovery of target-to-target paths up to ``lmax`` hops."""
    t0 = time.perf_counter()
    n_target = graph[target].num_nodes
    rng = np.random.default_rng(sample_seed)
    size = min(sample_sources, n_target)
    sources = np.sort(rng.choice(n_target, size=size, replace=False))
    proj = {t: gaussian_projection(graph[t].num_nodes, fingerprint_dim, sample_seed + 1 + i)
            for i, t in enumerate(graph.node_types)}
    nxt = schema_out(graph)
    reach = reach_within(graph, target, lmax)
    frontier = [([target], None)]
    discovered, log, generated = {}, [], 0
    for depth in range(1, lmax + 1):
        scored = []
        for prefix, x in frontier:
            for rel, dst in nxt.get(prefix[-1], []):
                schema = prefix + [rel, dst]
                generated += 1
                if dst != target and not can_complete(reach, dst, lmax - depth):
                    log.append({"depth": depth, "path": schema, "action": "drop",
                                "reason": "cannot reach the target type"})
                    continue
                op = ops[(prefix[-1], rel, dst)]
                x2 = (_density_switch(op[sources], dense_fraction) if x is None
                      else step(x, op, dense_fraction))
                stats, fp = prefix_stats(x2, sources, dst == target, proj[dst], variant, eps)
                del x2
                reason = None
                if stats["coverage"] < min_coverage:
                    reason = "coverage < min_coverage"
                elif stats["effective_endpoints"] < min_effective_endpoints:
                    reason = "endpoint collapse"
                if reason:
                    log.append({"depth": depth, "path": schema, "action": "drop",
                                "reason": reason, **stats})
                    continue
                scored.append((schema, x, stats, fp))
        scored.sort(key=lambda z: (-z[2]["score"], "-".join(z[0])))
        survivors = []
        for item in scored:
            dup = None
            for other in survivors:
                if other[0][-1] != item[0][-1]:
                    continue
                a, b = item[3], other[3]
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                if den > 0 and float(a @ b) / den > duplicate_cosine:
                    dup = other[0]
                    break
            if dup is not None:
                log.append({"depth": depth, "path": item[0], "action": "drop",
                            "reason": "duplicate", "of": dup, **item[2]})
                continue
            survivors.append(item)
        for rank, (schema, _, stats, _) in enumerate(survivors):
            if schema[-1] == target and depth >= min_hops:
                discovered[tuple(schema)] = {"depth": depth, **stats}
            log.append({"depth": depth, "path": schema,
                        "action": "keep" if rank < width else "prune", "rank": rank + 1,
                        **stats})
        frontier = []
        for schema, x, _, _ in survivors[:width]:
            op = ops[tuple(schema[-3:])]
            frontier.append((schema, _density_switch(op[sources], dense_fraction) if x is None
                             else step(x, op, dense_fraction)))
    return {"discovered": [{"path": list(p), **s} for p, s in discovered.items()],
            "log": log, "generated": generated, "sources": int(size),
            "runtime_s": time.perf_counter() - t0}
