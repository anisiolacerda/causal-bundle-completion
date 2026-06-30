from __future__ import annotations
from typing import Callable
import numpy as np


def item_rank(scores: np.ndarray, item: int) -> float:
    """1-based average-tie rank of `item` (higher score = better)."""
    s = np.asarray(scores, dtype=float)
    sv = s[item]
    greater = int(np.sum(s > sv))
    ties = int(np.sum(s == sv))  # includes item itself
    return float(greater + (ties + 1) / 2.0)


def per_bin_paired_stats(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
    sig_fn: Callable[[list, list], dict] | None = None,
) -> list[dict]:
    """Per-bin mean(a), mean(b), delta=mean_a-mean_b, n, and p_value via injected sig_fn."""
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    bi = np.asarray(bin_idx)
    out = []
    for bn in range(n_bins):
        mask = bi == bn
        n = int(mask.sum())
        if n == 0:
            out.append({"bin": bn, "n": 0, "mean_a": float("nan"),
                        "mean_b": float("nan"), "delta": float("nan"), "p_value": float("nan")})
            continue
        ma = float(np.nanmean(a[mask]))
        mb = float(np.nanmean(b[mask]))
        p = float("nan")
        if sig_fn is not None:
            p = float(sig_fn(list(a[mask]), list(b[mask])).get("p_value", float("nan")))
        out.append({"bin": bn, "n": n, "mean_a": ma, "mean_b": mb, "delta": ma - mb, "p_value": p})
    return out
