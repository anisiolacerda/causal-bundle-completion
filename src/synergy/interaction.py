# src/synergy/interaction.py
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from synergy.counts import cond_prob


def synergy_vectors(B: csr_matrix, obs_pair: tuple[int, int], n_bundles: int) -> dict[str, np.ndarray]:
    """Per-candidate-j vectors for the pairwise synergy contrast given observed (i, k).

    tau_syn(j) = P(j|{i,k}) - P(j|i) - P(j|k) + P(j)   (2x2 factorial interaction contrast)
    additive(j) = P(j|i) + P(j|k) - P(j)               (inclusion-exclusion additive prediction)
    joint(j)    = P(j|{i,k})
    Observed i,k entries of joint/additive/tau set to -inf.
    """
    i, k = obs_pair
    p_marg = cond_prob(B, [], n_bundles)
    p_i = cond_prob(B, [i], n_bundles)
    p_k = cond_prob(B, [k], n_bundles)
    joint = cond_prob(B, [i, k], n_bundles)
    additive = p_i + p_k - p_marg
    tau = joint - p_i - p_k + p_marg
    for arr in (joint, additive, tau):
        arr[i] = -np.inf
        arr[k] = -np.inf
    return {"joint": joint, "additive": additive, "tau": tau,
            "p_i": p_i, "p_k": p_k, "p_marg": p_marg}


def null_bundles_product_of_marginals(bundles, n_items, sizes, rng):
    """Placebo: redraw each bundle's items independently from the item marginal, preserving
    per-bundle size. Breaks any i-k-j interaction while keeping marginals ~fixed -> the
    additivity null for the synergy estimator.

    Design note — ``replace=False`` is deliberate.
    Bundles are *sets* of distinct items, so drawing a bundle's items without replacement
    preserves (a) per-bundle size, (b) the set structure (no self-co-occurrence), and
    (c) the item marginal distribution up to the within-bundle exclusion constraint.
    This is a **size-conditioned marginal null**: each bundle is drawn from the item
    marginal p conditioned on producing a set of exactly ``sz`` distinct items.
    It is *not* i.i.d.-with-replacement; using replace=True would allow an item to appear
    more than once in a single bundle, violating the set assumption and inflating variance.
    The null breaks structured i–k–j interaction (synergy) while retaining realistic
    bundle-size heterogeneity."""
    freq = np.zeros(n_items, dtype=float)
    for b in bundles:
        for it in b:
            if 0 <= it < n_items:
                freq[it] += 1.0
    p = freq / freq.sum()
    items = np.arange(n_items)
    out = []
    for sz in sizes:
        sz = min(sz, n_items)
        chosen = rng.choice(items, size=sz, replace=False, p=p)
        out.append(list(int(x) for x in chosen))
    return out
