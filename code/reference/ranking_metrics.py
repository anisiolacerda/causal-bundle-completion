"""Full-catalog ranking metrics for bundle completion (PIC-Diff recovery, Stage 3).

No negative sampling: each method scores ALL catalog items; we report the held-out gt's rank.
Ties use the average-rank convention (fair to ties — important for sparse co-occurrence scores
where many items tie at 0). Pure numpy.
"""
from __future__ import annotations

import numpy as np


def gt_ranks(scores: np.ndarray, gt_ids: np.ndarray) -> np.ndarray:
    """1-based average-tie rank of each row's gt item. scores (N, M), gt_ids (N,) -> (N,)."""
    n = scores.shape[0]
    gt_s = scores[np.arange(n), gt_ids][:, None]  # (N,1)
    greater = (scores > gt_s).sum(axis=1)
    ties = (scores == gt_s).sum(axis=1)  # includes the gt itself (>=1)
    return greater + (ties + 1) / 2.0


def rank_metrics_multigt(
    scores: np.ndarray, gt_lists: list, ks: tuple[int, ...] = (1, 5, 10, 20)
) -> dict[str, float]:
    """Multi-gt full-catalog metrics: recall@k = |top-k ∩ gt| / |gt|, hit@1, MRR (first gt hit).

    scores (N, M); gt_lists is a length-N list of held-out item-id arrays (|gt| >= 1). Ranks are
    descending by score, ties broken arbitrarily (the gt items typically have nonzero scores).
    """
    n = scores.shape[0]
    out: dict[str, list[float]] = {"hit@1": [], "mrr": [], **{f"recall@{k}": [] for k in ks}}
    for i in range(n):
        gt = np.asarray(gt_lists[i], dtype=np.int64)
        if gt.size == 0:
            continue
        ranks = (-scores[i]).argsort().argsort()[gt] + 1  # 1-based rank of each gt item
        best = int(ranks.min())
        out["hit@1"].append(1.0 if best == 1 else 0.0)
        out["mrr"].append(1.0 / best)
        for k in ks:
            out[f"recall@{k}"].append(float((ranks <= k).sum()) / gt.size)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


def rank_metrics_multigt_per_example(
    scores: np.ndarray, gt_lists: list, ks: tuple[int, ...] = (1, 5, 10, 20)
) -> dict[str, np.ndarray]:
    """Per-example multi-gt metrics (same definitions as `rank_metrics_multigt`, not averaged).

    Returns a dict of length-N arrays (one entry per row of `scores`). Rows with empty gt get NaN
    so callers can drop them. Enables a per-example paired test (n = #test bundles) between two
    methods scored on the SAME bundles — far more powerful than a seed-level test for a
    deterministic baseline like co-occurrence.
    """
    n = scores.shape[0]
    keys = ["hit@1", "mrr", *[f"recall@{k}" for k in ks]]
    out = {key: np.full(n, np.nan, dtype=np.float64) for key in keys}
    for i in range(n):
        gt = np.asarray(gt_lists[i], dtype=np.int64)
        if gt.size == 0:
            continue
        ranks = (-scores[i]).argsort().argsort()[gt] + 1  # 1-based rank of each gt item
        best = int(ranks.min())
        out["hit@1"][i] = 1.0 if best == 1 else 0.0
        out["mrr"][i] = 1.0 / best
        for k in ks:
            out[f"recall@{k}"][i] = float((ranks <= k).sum()) / gt.size
    return out


def reciprocal_rank_fusion(score_mats: list[np.ndarray], k0: float = 60.0) -> np.ndarray:
    """Combine several (N, M) score matrices by reciprocal-rank fusion: for each, rank items
    descending per row, then sum 1/(k0 + rank). Scale-free way to fuse content + CF signals."""
    fused = np.zeros_like(score_mats[0], dtype=np.float64)
    for s in score_mats:
        order = np.argsort(-s, axis=1)  # indices sorted by descending score
        rank = np.empty_like(order)
        rows = np.arange(s.shape[0])[:, None]
        rank[rows, order] = np.arange(s.shape[1])[None, :]  # 0-based rank per item
        fused += 1.0 / (k0 + rank)
    return fused


def rank_metrics(
    scores: np.ndarray, gt_ids: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 20)
) -> dict[str, float]:
    """hit@1, recall@k, MRR over the full catalog (gt ranked among ALL items)."""
    ranks = gt_ranks(scores, gt_ids)
    out: dict[str, float] = {
        "hit@1": float(np.mean(ranks <= 1.0)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks <= k))
    return out
