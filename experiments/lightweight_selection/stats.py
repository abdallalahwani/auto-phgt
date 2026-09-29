"""Statistics for V4-Hedge: Spearman correlations with path-bootstrap intervals and paired
seed-matched comparisons (bootstrap CI, exact sign-flip permutation p, paired t p)."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats as sps


def rank_of(values) -> list[int]:
    """1 = highest value; missing values rank last (ties by position)."""
    order = sorted(range(len(values)),
                   key=lambda i: (values[i] is None, -(values[i] or 0.0) if values[i] is not None
                                  else 0.0, i))
    ranks = [0] * len(values)
    for r, i in enumerate(order, 1):
        ranks[i] = r
    return ranks


def _pairs(*columns):
    keep = [i for i in range(len(columns[0])) if all(c[i] is not None for c in columns)]
    return [np.asarray([c[i] for i in keep], dtype=float) for c in columns]


def _rho(x, y) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def _rho_batch(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Row-wise Spearman rho of [B, n] resamples (average ranks for ties; NaN if constant)."""
    rx = sps.rankdata(xs, axis=1)
    ry = sps.rankdata(ys, axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def spearman(x, y, n_boot: int = 10000, seed: int = 0) -> dict:
    x, y = _pairs(x, y)
    if x.size < 3:
        return {"n": int(x.size), "rho": None, "p": None, "ci95": None}
    res = sps.spearmanr(x, y)
    idx = np.random.default_rng(seed).integers(0, x.size, (n_boot, x.size))
    boots = _rho_batch(x[idx], y[idx])
    boots = boots[~np.isnan(boots)]
    ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
          if boots.size else None)
    return {"n": int(x.size), "rho": float(res.statistic), "p": float(res.pvalue), "ci95": ci}


def spearman_diff_ci(a, b, y, n_boot: int = 10000, seed: int = 0) -> dict:
    """rho(a, y) - rho(b, y) with a paired bootstrap over items (same resamples for both)."""
    a, b, y = _pairs(a, b, y)
    if a.size < 3:
        return {"n": int(a.size), "diff": None, "ci95": None}
    diff = _rho(a, y) - _rho(b, y)
    idx = np.random.default_rng(seed).integers(0, a.size, (n_boot, a.size))
    boots = _rho_batch(a[idx], y[idx]) - _rho_batch(b[idx], y[idx])
    boots = boots[~np.isnan(boots)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"n": int(a.size), "diff": float(diff), "ci95": ci,
            "share_boot_le_0": float(np.mean(boots <= 0))}


def sign_flip_p(d: np.ndarray, max_exact: int = 16, draws: int = 100000, seed: int = 0) -> float:
    obs = abs(d.mean())
    if d.size <= max_exact:
        means = [abs((d * np.asarray(signs)).mean())
                 for signs in itertools.product((1, -1), repeat=d.size)]
        return float(np.mean(np.asarray(means) >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice((1, -1), size=(draws, d.size))
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))


def paired(a: dict, b: dict, n_boot: int = 10000, seed: int = 0, scale: float = 100.0) -> dict:
    """a - b over matched seeds (values scaled to points)."""
    seeds = sorted(set(a) & set(b))
    d = np.asarray([(a[s] - b[s]) * scale for s in seeds], dtype=float)
    out = {"n": int(d.size), "seeds": seeds, "per_seed": d.tolist(), "mean": None, "sd": None,
           "ci95": None, "p_perm": None, "p_t": None, "wins": int((d > 0).sum()),
           "losses": int((d < 0).sum())}
    if d.size == 0:
        return out
    out["mean"] = float(d.mean())
    if d.size >= 2:
        out["sd"] = float(d.std(ddof=1))
        rng = np.random.default_rng(seed)
        boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
        out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["p_perm"] = sign_flip_p(d)
        out["p_t"] = (float(sps.ttest_1samp(d, 0.0).pvalue) if out["sd"] > 0
                      else (1.0 if out["mean"] == 0 else 0.0))
    return out


def verdict(ci) -> str:
    """'above' if the CI lies above 0, 'below' if below 0, else 'overlaps'."""
    if not ci:
        return "n/a"
    if ci[0] > 0:
        return "above"
    if ci[1] < 0:
        return "below"
    return "overlaps"


def holm(pvalues: dict) -> dict:
    items = sorted((p, k) for k, p in pvalues.items() if p is not None)
    adjusted, running = {}, 0.0
    for i, (p, key) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        adjusted[key] = running
    return adjusted


def kendall_tau(x, y) -> float | None:
    x, y = _pairs(x, y)
    if x.size < 3:
        return None
    return float(sps.kendalltau(x, y).statistic)
