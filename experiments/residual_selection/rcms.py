"""RCMS: Residual-Conditional Meta-Path Selection.

U(p | H, S) = CE(H, Z_S) - CE(H, Z_S, Z_p), estimated with a fixed-capacity L2-regularised
multinomial logistic probe under cross-fitting inside the official training set:

* H is the HGT representation entering the probe as OUT-OF-FOLD class log-probabilities:
  for fold j an HGT with the strong configuration is trained on the other folds (official
  validation only for early stopping) and predicts fold j. Log-probabilities are aligned
  across the fold models; raw embeddings of separately trained HGTs are not.
* Z_p is a label-free path representation: for each target node, the mean over sampled
  complete instances of the mean input representation of the path nodes after the target
  (the Auto-PHGT token content without learned embeddings), plus the completion rate.
* Greedy selection adds argmax_p U(p | H, S) until |S| = k. Ablations: independent
  residual ranking U(p | H, {}) and conditional selection without H, U(p | S).
The official test split is never read.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from auto_phgt.discovery import StatisticalDiscoveryModule, random_valid_paths
from auto_phgt.selection import homophily_table, hybrid_top_k, path_families
from auto_phgt.tokenization import PAD, MetaPathInstanceExtractor

from .plan import RCMS


# --------------------------------------------------------------------------- candidates
def candidate_space(graph, target_type: str, lmax: int):
    """All schema-valid target-to-target paths with 2..lmax hops in canonical score order."""
    engine = StatisticalDiscoveryModule(graph, target_type, lmax)
    cands = engine.discover_candidate_paths()
    scores = engine.calculate_path_frequencies(cands)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [list(p) for p, _ in ranked], {tuple(p): s for p, s in ranked}


def coverage(graph, schemas, ids, seed: int, instances: int = 4) -> list[float]:
    out = []
    for start in range(0, len(schemas), 64):
        chunk = schemas[start:start + 64]
        ex = MetaPathInstanceExtractor(graph, chunk, instances_per_path=instances, seed=seed)
        out.extend(ex.sample(ids, seed=seed).complete_mask.float().mean((0, 2)).tolist())
    return out


def screen(graph, schemas, ids, seed: int):
    """Drop only zero/negligible-coverage candidates; every decision is recorded."""
    cov = coverage(graph, schemas, ids, seed)
    kept, dropped = [], []
    for schema, c in zip(schemas, cov):
        (kept if c >= RCMS["min_coverage"] else dropped).append((schema, c))
    return kept, [{"path": s, "coverage": c, "reason": "coverage < min_coverage"}
                  for s, c in dropped]


# --------------------------------------------------------------------------- representations
@torch.no_grad()
def path_reprs(graph, schemas, node_repr: dict, ids, seed: int,
               instances: int = RCMS["search_instances"]) -> torch.Tensor:
    """[M, N, d + 1]: mean input vector of the non-target path positions over complete
    instances, and the completion rate."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    d = next(iter(node_repr.values())).size(1)
    out = torch.zeros(len(schemas), ids.numel(), d + 1)
    for start in range(0, len(schemas), 32):
        chunk = schemas[start:start + 32]
        ex = MetaPathInstanceExtractor(graph, chunk, instances_per_path=instances, seed=seed)
        inst = ex.sample(ids, seed=seed)
        complete = inst.complete_mask  # [N, K, I]
        for k, template in enumerate(ex.templates):
            types = template.node_types
            acc = torch.zeros(ids.numel(), instances, d)
            for pos in range(1, len(types)):
                node = inst.node_ids[:, k, :, pos].clamp(min=0)
                acc += node_repr[types[pos]][node]
            acc /= max(1, len(types) - 1)
            mask = complete[:, k].unsqueeze(-1).float()
            count = mask.sum(1)
            out[start + k, :, :d] = (acc * mask).sum(1) / count.clamp(min=1)
            out[start + k, :, d] = complete[:, k].float().mean(1)
    return out


def standardize(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Column z-scores over nodes (label-free statistics)."""
    mean = x.mean(dim, keepdim=True)
    std = x.std(dim, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std


def stratified_folds(labels: torch.Tensor, folds: int, seed: int) -> torch.Tensor:
    """Fold id per position; each class is spread round-robin after a seeded shuffle."""
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(labels.numel(), dtype=torch.long)
    offset = 0
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero().view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        out[idx] = (torch.arange(idx.numel()) + offset) % folds
        offset += idx.numel()
    return out


# --------------------------------------------------------------------------- probe
def probe_ce(xb, xc, y, folds, num_classes: int, device, *, l2=RCMS["probe_l2"],
             lr=RCMS["probe_lr"], steps=RCMS["probe_steps"], chunk: int = 64):
    """Held-out CE of independent logistic probes, one per candidate block in ``xc``.

    ``xb`` [N, Fb] (or None) is shared; ``xc`` [M, N, Fc] (or None for the base probe).
    Every probe has the same architecture, L2 penalty, optimiser, and step budget; Adam is
    per-coordinate, so batching candidates does not couple them. Returns [M] (or [1]).
    """
    y = y.to(device)
    xb = xb.to(device) if xb is not None else None
    n_cand = 1 if xc is None else xc.size(0)
    total = torch.zeros(n_cand, device=device)
    for fold in folds.unique().tolist():
        tr, ev = (folds != fold).to(device), (folds == fold).to(device)
        for start in range(0, n_cand, chunk):
            m = min(chunk, n_cand - start)
            params = [torch.zeros(m, num_classes, device=device, requires_grad=True)]
            if xb is not None:
                params.append(torch.zeros(m, xb.size(1), num_classes, device=device,
                                          requires_grad=True))
            block = None
            if xc is not None:
                block = xc[start:start + m].to(device)
                params.append(torch.zeros(m, block.size(2), num_classes, device=device,
                                          requires_grad=True))

            def logits(mask):
                out = params[0][:, None, :].expand(m, int(mask.sum()), num_classes)
                i = 1
                if xb is not None:
                    out = out + torch.einsum("nf,mfc->mnc", xb[mask], params[i])
                    i += 1
                if block is not None:
                    out = out + torch.einsum("mnf,mfc->mnc", block[:, mask], params[i])
                return out

            opt = torch.optim.Adam(params, lr=lr)
            y_tr = y[tr]
            for _ in range(steps):
                opt.zero_grad()
                out = logits(tr)
                loss = F.cross_entropy(out.reshape(-1, num_classes), y_tr.repeat(m),
                                       reduction="sum") / y_tr.numel()
                loss = loss + l2 * sum((p ** 2).sum() for p in params[1:])
                loss.backward()
                opt.step()
            with torch.no_grad():
                out = logits(ev)
                ce = F.cross_entropy(out.reshape(-1, num_classes), y[ev].repeat(m),
                                     reduction="none").view(m, -1).mean(1)
            total[start:start + m] += ce * ev.sum() / y.numel()
    return total.cpu()


def greedy(h, z, y, folds, num_classes: int, k: int, device, *, use_h: bool = True,
           pool_after_step1: int = RCMS["pool_after_step1"]):
    """Greedy conditional selection. Returns chosen indices, per-step records, and the
    step-1 utilities of every candidate."""
    selected, steps, pool, first = [], [], list(range(z.size(0))), None
    for step in range(min(k, z.size(0))):
        t0 = time.perf_counter()
        blocks = ([h] if use_h and h is not None else []) + [z[i] for i in selected]
        xb = torch.cat(blocks, 1) if blocks else None
        base = float(probe_ce(xb, None, y, folds, num_classes, device)[0])
        cand = [i for i in pool if i not in selected]
        ce = probe_ce(xb, z[cand], y, folds, num_classes, device)
        util = base - ce
        order = sorted(range(len(cand)), key=lambda j: (-float(util[j]), cand[j]))
        best = cand[order[0]]
        if step == 0:
            first = {cand[j]: float(util[j]) for j in range(len(cand))}
            if len(pool) > pool_after_step1:
                pool = [cand[j] for j in order[:pool_after_step1]]
        steps.append({"step": step + 1, "selected": best, "utility": float(util[order[0]]),
                      "ce_base": base, "ce_with": float(ce[order[0]]),
                      "next_best": [{"candidate": cand[j], "utility": float(util[j])}
                                    for j in order[1:6]],
                      "evaluated": len(cand), "runtime_s": time.perf_counter() - t0})
        selected.append(best)
    return selected, steps, first


# --------------------------------------------------------------------------- analysis
def residual_cka(h, z, sets: dict, ridge: float = 1e-2) -> dict:
    """Mean pairwise linear CKA of HGT-residualised path representations within each set."""
    zs = z[:, :, :-1]
    if h is not None:
        a = torch.cat([h, torch.ones(h.size(0), 1)], 1)
        proj = torch.linalg.solve(a.T @ a + ridge * torch.eye(a.size(1)), a.T)
        zs = zs - a @ (proj @ zs)
    zs = zs - zs.mean(1, keepdim=True)

    def cka(x, w):
        num = (x.T @ w).pow(2).sum()
        den = (x.T @ x).pow(2).sum().sqrt() * (w.T @ w).pow(2).sum().sqrt()
        return float(num / den.clamp(min=1e-12))

    out = {}
    for name, members in sets.items():
        pairs = [(a_, b_) for i, a_ in enumerate(members) for b_ in members[i + 1:]]
        values = [cka(zs[a_], zs[b_]) for a_, b_ in pairs]
        out[name] = {"members": members, "mean_pairwise_cka": (sum(values) / len(values)
                                                                if values else None),
                     "pairs": [{"a": a_, "b": b_, "cka": v} for (a_, b_), v in zip(pairs, values)]}
    return out


def selections(steps_full, first_full, steps_nohgt, ks, m: int) -> dict:
    sel = {"rcms": {}, "rcms_indep": {}, "rcms_nohgt": {}}
    ranked = sorted(first_full, key=lambda i: (-first_full[i], i))
    for k in ks:
        kk = min(k, m)
        sel["rcms"][str(k)] = [s["selected"] for s in steps_full[:kk]]
        sel["rcms_nohgt"][str(k)] = [s["selected"] for s in steps_nohgt[:kk]]
        sel["rcms_indep"][str(k)] = ranked[:kk]
    return sel


def comparison_sets(graph, target_type, ranked, scores, train_ids, labels, seed, k, lmax):
    """Index sets (into ``ranked``) of the heuristic selectors, computed exactly as the
    final-run selector code does (v2 select_templates semantics)."""
    table = homophily_table(graph, ranked, train_ids, labels, target_node_type=target_type,
                            seed=seed)
    kk = min(k, len(ranked))
    index = {tuple(p): i for i, p in enumerate(ranked)}
    sets = {"canonical": list(range(kk)),
            "hybrid": [index[tuple(p)] for p in hybrid_top_k(ranked, scores, table, kk)]}
    if kk <= len(ranked):
        rand, _, _ = random_valid_paths(graph, target_node_type=target_type, max_hops=lmax,
                                        k=kk, seed=seed)
        sets["random"] = [index[tuple(p)] for p in rand]
    return table, sets


def describe(ranked, scores, idx_list):
    return [{"path": ranked[i], "canonical_rank": i + 1, "structural_score": scores[tuple(ranked[i])],
             "families": sorted(path_families(ranked[i]))} for i in idx_list]
