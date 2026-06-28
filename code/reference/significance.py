# src/causalbr/eval/significance.py
"""Paired significance utilities for the Phase-6 multi-seed verdict (Gate-G6).

Exact paired Wilcoxon signed-rank (one/two-sided), bootstrap effect-size CIs, and
Bonferroni family correction. Pure; delegates to SciPy when available but carries a
self-contained exact / normal-approx fallback. No ranking-era imports.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

try:  # optional; the fallback path is exercised by monkeypatching _SPS to None
    import scipy.stats as _SPS
except ImportError:  # pragma: no cover
    _SPS = None


def _avg_ranks(values: list[float]) -> list[float]:
    """1-based ranks with ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _exact_p(ranks: list[float], w_plus: float, alternative: str) -> float:
    n = len(ranks)
    total = 2 ** n
    ge = le = 0
    for signs in itertools.product((0, 1), repeat=n):
        wp = sum(r for r, s in zip(ranks, signs, strict=True) if s)
        if wp >= w_plus:
            ge += 1
        if wp <= w_plus:
            le += 1
    if alternative == "greater":
        return ge / total
    if alternative == "less":
        return le / total
    return min(1.0, 2.0 * min(ge, le) / total)


def _normal_p(ranks: list[float], w_plus: float, alternative: str) -> float:
    n = len(ranks)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0.0:
        return 1.0
    z = (w_plus - mu) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if alternative == "greater":
        return 1.0 - cdf
    if alternative == "less":
        return cdf
    return min(1.0, 2.0 * min(cdf, 1.0 - cdf))


def paired_wilcoxon(
    a: list[float], b: list[float], *, alternative: str = "greater"
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test on diffs a-b. Zeros dropped (Pratt)."""
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    median_diff = float(np.median(diffs)) if diffs else 0.0
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "median_diff": median_diff, "n": 0}
    ranks = _avg_ranks([abs(d) for d in nz])
    w_plus = float(sum(r for r, d in zip(ranks, nz, strict=True) if d > 0))
    if _SPS is not None:
        p = float(_SPS.wilcoxon(nz, alternative=alternative, zero_method="wilcox").pvalue)
    elif n <= 20:
        p = _exact_p(ranks, w_plus, alternative)
    else:
        p = _normal_p(ranks, w_plus, alternative)
    return {"statistic": w_plus, "p_value": p, "median_diff": median_diff, "n": n}


def bootstrap_ci(
    diffs: list[float], *, alpha: float = 0.05, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `diffs`."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bonferroni(p_values: list[float], m: int | None = None) -> list[float]:
    """Family-wise Bonferroni correction; caps each corrected p at 1.0."""
    k = m if m is not None else len(p_values)
    return [min(1.0, float(p) * k) for p in p_values]
