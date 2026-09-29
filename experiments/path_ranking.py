"""Selectors: Transition-Info (independent and set-aware) and FastPath (individual and
set-aware). Every function is deterministic; ties are broken by canonical structural rank."""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# --------------------------------------------------------------------------- Transition-Info
def rank_by(scores) -> list[int]:
    """Indices sorted by descending score; ties by position (= canonical rank)."""
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def cosine_matrix(fps: np.ndarray) -> np.ndarray:
    flat = fps.reshape(fps.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    gram = flat @ flat.T
    den = np.outer(norms, norms)
    return np.divide(gram, den, out=np.zeros_like(gram), where=den > 0)


def ti_set_greedy(scores, cos: np.ndarray, k: int):
    """First pick argmax TI; then argmax TI(p) * (1 - max_{s in S} max(0, cos(p, s)))."""
    scores = np.asarray(scores, dtype=float)
    selected, steps = [], []
    for _ in range(min(k, len(scores))):
        best, best_val, rows = None, None, []
        for i in range(len(scores)):
            if i in selected:
                continue
            sim = max((max(0.0, float(cos[i, s])) for s in selected), default=0.0)
            value = scores[i] * (1.0 - sim)
            rows.append((i, value, sim))
            if best is None or value > best_val:
                best, best_val = i, value
        rows.sort(key=lambda r: (-r[1], r[0]))
        steps.append({"selected": best, "score": best_val, "ti": float(scores[best]),
                      "max_similarity": next(r[2] for r in rows if r[0] == best),
                      "next_best": [{"candidate": r[0], "score": r[1], "max_similarity": r[2]}
                                    for r in rows[1:6]]})
        selected.append(best)
    return selected, steps


# --------------------------------------------------------------------------- FastPath
def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Fold id per position: each class shuffled (seeded) and dealt round-robin."""
    rng = np.random.default_rng(seed)
    out = np.empty(labels.size, dtype=int)
    offset = 0
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(idx.size)]
        out[idx] = (np.arange(idx.size) + offset) % folds
        offset += idx.size
    return out


def class_prior(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    return counts / counts.sum()


def propagate(block: np.ndarray, y: np.ndarray, folds: np.ndarray, num_classes: int):
    """Echo-free cross-fitted label propagation along one path.

    ``block`` [n, n]: T_p between the run's training nodes, diagonal already removed. For fold
    j the labels of fold j are masked; every node gets m(u) = sum_{v known, v != u} T(v|u)
    onehot(y_v). Returns fold-specific distributions L [F, n, C], labelled mass w [F, n] and
    priors [F, C]; out-of-fold rows are L[fold(u), u].
    """
    n_folds = int(folds.max()) + 1
    onehot = np.eye(num_classes)[y]
    L = np.zeros((n_folds, y.size, num_classes))
    W = np.zeros((n_folds, y.size))
    priors = np.zeros((n_folds, num_classes))
    for j in range(n_folds):
        known = folds != j
        prior = class_prior(y[known], num_classes)
        m = block[:, known].astype(np.float64) @ onehot[known]
        w = m.sum(1)
        L[j] = np.where(w[:, None] > 0, m / np.where(w > 0, w, 1.0)[:, None], prior[None, :])
        W[j] = w
        priors[j] = prior
    return L, W, priors


def fastpath_scores(L, W, priors, y, folds, eps: float) -> dict:
    """Out-of-fold FastPath metrics; Q_FP = -CE with eps-smoothing toward the fold prior."""
    idx = np.arange(y.size)
    oof = L[folds, idx]
    w = W[folds, idx]
    prior = priors[folds]
    smoothed = (1.0 - eps) * oof + eps * prior
    ce = float(-np.log(smoothed[idx, y]).mean())
    pred = oof.argmax(1)
    covered = w > 0
    ent = -(np.where(oof > 0, oof * np.log(np.where(oof > 0, oof, 1.0)), 0.0)).sum(1)
    return {"Q_FP": -ce, "CE": ce, "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "coverage": float(covered.mean()), "labelled_mass": float(w.mean()),
            "mean_entropy": float(ent[covered].mean()) if covered.any() else None,
            "mean_confidence": float(oof.max(1)[covered].mean()) if covered.any() else None}


def fold_features(L, W) -> np.ndarray:
    """[F, n, C + 1]: fold-specific class distribution plus an evidence indicator."""
    return np.concatenate([L, (W > 0).astype(float)[..., None]], axis=-1)


def probe_ce(blocks, y, folds, num_classes: int, *, C: float = 1.0, max_iter: int = 1000):
    """Held-out CE of a multinomial logistic probe on the concatenated feature blocks.

    For fold j the probe is fitted on the known rows (folds != j) and evaluated on fold j;
    the result is averaged with fold-size weights. With no block, the known-label class prior.
    """
    total = 0.0
    n_folds = int(folds.max()) + 1
    for j in range(n_folds):
        tr, ev = folds != j, folds == j
        if not blocks:
            prior = np.clip(class_prior(y[tr], num_classes), 1e-12, None)
            ce = float(-np.log(prior[y[ev]]).mean())
        else:
            x = np.concatenate([b[j] for b in blocks], axis=1)
            mu = x[tr].mean(0)
            sd = x[tr].std(0)
            sd = np.where(sd > 1e-6, sd, 1.0)
            x = (x - mu) / sd
            model = LogisticRegression(C=C, max_iter=max_iter)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x[tr], y[tr])
            proba = np.full((int(ev.sum()), num_classes), 1e-12)
            proba[:, model.classes_] = np.clip(model.predict_proba(x[ev]), 1e-12, None)
            proba /= proba.sum(1, keepdims=True)
            ce = float(-np.log(proba[np.arange(proba.shape[0]), y[ev]]).mean())
        total += ce * ev.sum() / y.size
    return total


def fastpath_set_greedy(features, y, folds, num_classes: int, k: int, *, C: float = 1.0,
                        max_iter: int = 1000):
    """Greedy conditional selection: add argmax_p CE(S) - CE(S + p) until |S| = k.

    ``features`` [M, F, n, C + 1] are cached per candidate, so no model is retrained except
    the tiny probe. Returns the chosen indices, per-step records and the single-path probe CE
    of every candidate (step 1).
    """
    selected, steps, single = [], [], None
    base = probe_ce([], y, folds, num_classes)
    empty = base
    for _ in range(min(k, features.shape[0])):
        t0 = time.perf_counter()
        blocks = [features[i] for i in selected]
        rows = []
        for i in range(features.shape[0]):
            if i in selected:
                continue
            ce = probe_ce(blocks + [features[i]], y, folds, num_classes, C=C, max_iter=max_iter)
            rows.append((i, base - ce, ce))
        rows.sort(key=lambda r: (-r[1], r[0]))
        if single is None:
            single = {i: ce for i, _, ce in rows}
        best, gain, ce = rows[0]
        steps.append({"selected": best, "gain": gain, "ce_before": base, "ce_after": ce,
                      "evaluated": len(rows), "runtime_s": time.perf_counter() - t0,
                      "next_best": [{"candidate": r[0], "gain": r[1]} for r in rows[1:6]]})
        selected.append(best)
        base = ce
    return selected, steps, single, empty
