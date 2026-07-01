from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix


def item_bundle_incidence(bundles: list[list[int]], n_items: int) -> csr_matrix:
    """Item x bundle incidence (csr): B[i, b] = 1 iff item i in bundle b.

    Memory-safe: only sum(|bundle|) nonzeros, never the n_items^2 co-occ matrix.
    (Copied from scripts/cec_m05_exposure_premise.py.)
    """
    rows, cols = [], []
    for b_idx, b in enumerate(bundles):
        for it in b:
            if 0 <= it < n_items:
                rows.append(it); cols.append(b_idx)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_items, len(bundles)))


def bundles_with(B: csr_matrix, items: list[int]) -> np.ndarray:
    """Sorted bundle indices containing ALL of `items` (intersection). [] -> all bundles."""
    if not items:
        return np.arange(B.shape[1], dtype=int)
    acc = set(B[items[0]].indices.tolist())
    for it in items[1:]:
        acc &= set(B[it].indices.tolist())
        if not acc:
            break
    return np.array(sorted(acc), dtype=int)


def counts_over_j(B: csr_matrix, bundle_idx: np.ndarray) -> np.ndarray:
    """Length-n_items vector: #(given bundles) containing each item j."""
    if len(bundle_idx) == 0:
        return np.zeros(B.shape[0], dtype=float)
    return np.asarray(B[:, bundle_idx].sum(axis=1)).ravel().astype(float)


def cond_prob(B: csr_matrix, cond_items: list[int], n_bundles: int) -> np.ndarray:
    """P(j | all cond_items present) per item j, Laplace-smoothed (+1 / +n_items).

    cond_items == [] -> marginal P(j) over all bundles.
    Note: n_bundles is used ONLY for the marginal case (cond_items == []); when
    cond_items is non-empty the denominator uses the count of qualifying bundles
    (i.e. len(bundles_with(B, cond_items))), not n_bundles.
    """
    n_items = B.shape[0]
    if not cond_items:
        idx = np.arange(n_bundles, dtype=int)
    else:
        idx = bundles_with(B, cond_items)
    counts = counts_over_j(B, idx)
    return (counts + 1.0) / (len(idx) + n_items)


def cooc_score(B: csr_matrix, observed: list[int], n_items: int) -> np.ndarray:
    """Additive co-occ completion score cs[j] = sum_{o in observed} cooc(o, j); observed -> -inf.

    Via B @ (B[observed].sum(0)) (copied from cec_m05_exposure_premise.cooc_scores).
    """
    if not observed:
        return np.full(n_items, -np.inf)
    overlap = np.asarray(B[observed].sum(axis=0)).ravel()
    cs = np.asarray(B @ overlap).ravel().astype(float)
    cs[observed] = -np.inf
    return cs


def ppmi_score(B: csr_matrix, observed: list[int], n_items: int) -> np.ndarray:
    """PPMI-weighted additive co-occ score; observed -> -inf.

    PPMI(o, j) = max(0, log( (n(o,j) * N) / (n(o) * n(j)) )). Score[j] = sum_o PPMI(o, j).
    Computed only over items that co-occur with the observed set (sparse).
    """
    N = B.shape[1]
    n_item = np.asarray(B.sum(axis=1)).ravel().astype(float)  # per-item bundle freq
    score = np.zeros(n_items, dtype=float)
    for o in observed:
        b_o = B[o].indices
        n_o = float(len(b_o))
        if n_o == 0:
            continue
        co = counts_over_j(B, b_o)  # n(o, j) for all j
        nz = co > 0
        pmi = np.zeros(n_items, dtype=float)
        pmi[nz] = np.log((co[nz] * N) / (n_o * np.maximum(n_item[nz], 1.0)))
        score += np.maximum(pmi, 0.0)
    score[observed] = -np.inf
    return score
