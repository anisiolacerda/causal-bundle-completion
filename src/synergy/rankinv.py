from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, kendalltau


def item_rank(scores: np.ndarray, item: int) -> float:
    """1-based average-tie rank of `item` (higher score = better). Copied from cec.stratify."""
    s = np.asarray(scores, dtype=float)
    sv = s[item]
    greater = int(np.sum(s > sv))
    ties = int(np.sum(s == sv))
    return float(greater + (ties + 1) / 2.0)


def hit_at_k(scores: np.ndarray, item: int, k: int) -> float:
    return 1.0 if item_rank(scores, item) <= k else 0.0


def rank_agreement(scores_a: np.ndarray, scores_b: np.ndarray,
                   candidates: np.ndarray, topk: int) -> dict:
    """Agreement between two score orderings restricted to `candidates`.

    Returns spearman, kendall (rank correlation of the scores over candidates) and
    topk_jaccard (Jaccard overlap of the two top-`topk` candidate sets). Non-finite
    scores (-inf masked) are dropped from the candidate set.
    """
    cand = np.asarray(candidates)
    a = np.asarray(scores_a, dtype=float)[cand]
    b = np.asarray(scores_b, dtype=float)[cand]
    finite = np.isfinite(a) & np.isfinite(b)
    cand, a, b = cand[finite], a[finite], b[finite]
    if a.size < 3:
        return {"spearman": float("nan"), "kendall": float("nan"), "topk_jaccard": float("nan")}
    sp = float(spearmanr(a, b).statistic)
    kt = float(kendalltau(a, b).statistic)
    kk = min(topk, a.size)
    top_a = set(cand[np.argsort(-a)[:kk]].tolist())
    top_b = set(cand[np.argsort(-b)[:kk]].tolist())
    union = top_a | top_b
    jac = len(top_a & top_b) / len(union) if union else float("nan")
    return {"spearman": sp, "kendall": kt, "topk_jaccard": float(jac)}
