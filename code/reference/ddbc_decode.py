"""Full-catalog scoring for DDBC's generative decode (PIC-Diff recovery, Stage 6 headline).

DDBC samples a completion tail, then reconstructs each generated slot's embedding from its RVQ
codes and retrieves the nearest catalog item by cosine. The shipped evaluator scores that retrieval
against a *rho-pool shortlist that already contains the ground truth* — an over-claiming protocol
(see CLAUDE.md / the shortlist-vs-full-catalog artifact). To compare DDBC head-to-head with
co-occurrence and the reranker we instead turn its decode into a **full-catalog** score vector:
for each catalog item, its score is the max cosine similarity to ANY reconstructed slot. This
recovers DDBC's own argmax retrieval as the top of the ranking and extends it to recall@k with no
negative sampling. Pure numpy; the diffusion sampling that produces `recon` lives in the runner.
"""
from __future__ import annotations

import numpy as np


def unit_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero (eps guard)."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def slot_pool_scores(
    recon: np.ndarray,
    item_unit: np.ndarray,
    *,
    observed: list[int] | None = None,
    pool: str = "max",
) -> np.ndarray:
    """Full-catalog score vector from one example's reconstructed slot embeddings.

    `recon` (n_slots, D) are the decoded-slot embeddings (un-normalized is fine; normalized here).
    `item_unit` (n_items, D) MUST be row-unit-normalized by the caller (done once for the catalog).
    Score(item) = pool over slots of cosine(slot, item); `pool` is "max" (default) or "mean".
    `observed` item ids are masked to -inf (standard completion protocol — never re-predict a seen
    item). Returns (n_items,).
    """
    rn = unit_normalize(recon)
    sims = rn @ item_unit.T  # (n_slots, n_items)
    if pool == "max":
        scores = sims.max(axis=0)
    elif pool == "mean":
        scores = sims.mean(axis=0)
    else:
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
    scores = scores.astype(np.float32)
    if observed:
        obs = np.asarray(observed, dtype=np.int64)
        obs = obs[obs < scores.shape[0]]
        scores[obs] = -np.inf
    return scores
