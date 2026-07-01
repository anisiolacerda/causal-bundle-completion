# Synergy Falsification Gate (Design 1, slice-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cheaply (CPU-minutes) decide whether a within-set **causal synergy** estimand `τ_syn(j|S)` carries non-additive completion signal that co-occurrence cannot represent — kill-or-proceed BEFORE any reranker training / GPU spend, mirroring the M0.5 pattern.

**Architecture:** A small pure-numpy/scipy package `src/synergy/` (conditional co-occurrence counts → the pairwise synergy contrast → rank-agreement metrics), plus one runner `scripts/synergy_m1_gate.py` that measures the pre-registered gate metrics on the dense MealRec+ splits and writes a verdict JSON. No model training, no content embeddings, no shortlist — full-catalog counts over the item×bundle incidence matrix. The gate reuses this repo's proven memory-safe incidence mat-vec and the predecessor's paired-Wilcoxon.

**Tech Stack:** Python 3.10+, numpy 2.5, scipy 1.18 (pytest with `pythonpath=["src","."]`). Runs under the predecessor venv `/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python` (has numpy/scipy; torch present but unused by the gate).

## Global Constraints

- **Full-catalog only.** No shortlist, no negative sampling. All scores range over every item; observed items masked to `-inf`. (Copied from spec: "FULL-CATALOG or it does not count.")
- **CPU-minutes, no GPU, no training.** The gate loads `.npy` bundles + builds a sparse incidence matrix only. If it needs a model it is out of scope for this plan.
- **Pre-registered thresholds (fixed BEFORE running).** The exact GO/KILL numbers in Task 6 are frozen; do not tune them after seeing results. (Copied from spec §4.8 / honesty guards.)
- **Reuse, do not rebuild.** Incidence mat-vec is copied from `scripts/cec_m05_exposure_premise.py`; paired Wilcoxon from predecessor `causalbr.eval.significance` (added to `sys.path`, same pattern as the cec scripts). Predecessor code is UNPUBLISHED — reuse, never cite as prior art.
- **Datasets:** `/Users/anisiomlacerda/code/Bundle_construction/data/mealrec_l` and `.../mealrec_h`. Format: `train_examples.npy` rows `(id:int, items:list)`, `test_examples.npy` rows `(id:int, observed:list, held_out:list)`. MealRec+L meals are size 3 (observe 2 → predict 1).
- **Run command prefix** (all tests and the runner):
  ```
  PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
  /Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m ...
  ```
- **Package layout:** new code in `src/synergy/` (package name `synergy`, importable as `from synergy.X import Y` because pyproject sets `pythonpath=["src","."]`). Tests in `tests/`.

---

## File Structure

- Create `src/synergy/__init__.py` — package marker.
- Create `src/synergy/counts.py` — incidence matrix, subset bundle-index intersection, per-candidate conditional-probability & co-occ score vectors, PPMI. One responsibility: turn bundles into count/probability vectors.
- Create `src/synergy/interaction.py` — the pairwise synergy contrast `τ_syn(j|{i,k})` and the additive baseline, built on `counts.py`. One responsibility: the estimand.
- Create `src/synergy/rankinv.py` — rank-agreement (Spearman, Kendall, top-k Jaccard), `hit_at_k`, `item_rank`. One responsibility: rank-inversion / completion metrics.
- Create `tests/test_synergy_counts.py`, `tests/test_synergy_interaction.py`, `tests/test_synergy_rankinv.py` — unit tests with hand-built synthetic bundles where the answer is known.
- Create `scripts/synergy_m1_gate.py` — the runner: load MealRec+, compute all gate metrics + placebo null, write verdict JSON.
- Output: `results/synergy/m1/<dataset>.json`.
- Create `docs/findings/2026-07-01-synergy-gate-verdict.md` — the recorded GO/KILL verdict (Task 7).

---

### Task 1: Package scaffold + conditional-count primitives

**Files:**
- Create: `src/synergy/__init__.py`
- Create: `src/synergy/counts.py`
- Test: `tests/test_synergy_counts.py`

**Interfaces:**
- Produces:
  - `item_bundle_incidence(bundles: list[list[int]], n_items: int) -> scipy.sparse.csr_matrix` — item×bundle incidence `B[i,b]=1 iff item i in bundle b`.
  - `bundles_with(B, items: list[int]) -> np.ndarray` — sorted bundle indices containing ALL of `items` (all-items intersection); empty list → all bundles.
  - `counts_over_j(B, bundle_idx: np.ndarray) -> np.ndarray` — length-`n_items` vector: number of the given bundles containing each item `j`.
  - `cond_prob(B, cond_items: list[int], n_bundles: int) -> np.ndarray` — `P(j | all cond_items present)` per item `j` (Laplace-smoothed; `[]` → marginal `P(j)`).
  - `cooc_score(B, observed: list[int], n_items: int) -> np.ndarray` — additive co-occ completion score (the proven mat-vec), observed masked to `-inf`.
  - `ppmi_score(B, observed: list[int], n_items: int) -> np.ndarray` — PPMI-weighted additive co-occ score, observed masked to `-inf`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synergy_counts.py
import numpy as np
from synergy.counts import (
    item_bundle_incidence, bundles_with, counts_over_j, cond_prob, cooc_score,
)

# 4 items, 4 bundles. item 3 appears ONLY with {0,1} together.
BUNDLES = [[0, 1, 3], [0, 1, 3], [0, 2], [1, 2]]
N_ITEMS = 4

def test_incidence_shape_and_entries():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    assert B.shape == (4, 4)
    assert np.array_equal(np.asarray(B[3].todense()).ravel(), [1, 1, 0, 0])  # item 3 in bundles 0,1

def test_bundles_with_intersection():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    assert list(bundles_with(B, [0, 1])) == [0, 1]     # bundles containing BOTH 0 and 1
    assert list(bundles_with(B, [2])) == [2, 3]
    assert list(bundles_with(B, [])) == [0, 1, 2, 3]   # empty -> all bundles

def test_counts_over_j():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    # over bundles {0,1} (which contain items 0,1,3), item counts:
    c = counts_over_j(B, np.array([0, 1]))
    assert list(c) == [2, 2, 0, 2]

def test_cond_prob_laplace():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    # P(3 | {0,1}) : bundles with {0,1} = 2, both contain 3 -> (2+1)/(2+4*1)? use +1/+n_items smoothing
    p = cond_prob(B, [0, 1], n_bundles=4)
    # smoothing: (count_j + 1) / (n_cond + n_items)
    assert abs(p[3] - (2 + 1) / (2 + 4)) < 1e-9
    # marginal P(j): item 0 in 3 of 4 bundles -> (3+1)/(4+4)
    pm = cond_prob(B, [], n_bundles=4)
    assert abs(pm[0] - (3 + 1) / (4 + 4)) < 1e-9

def test_cooc_score_masks_observed():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    s = cooc_score(B, [0, 1], N_ITEMS)
    assert s[0] == -np.inf and s[1] == -np.inf   # observed masked
    assert s[3] > s[2]                            # 3 co-occurs with {0,1} more than 2 does
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_counts.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'synergy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/synergy/__init__.py
```
(empty file — package marker)

```python
# src/synergy/counts.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_counts.py -q
```
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/synergy/__init__.py src/synergy/counts.py tests/test_synergy_counts.py
git commit -m "feat(synergy): conditional co-occ count primitives for the synergy gate"
```

---

### Task 2: The pairwise synergy contrast + additive baseline

**Files:**
- Create: `src/synergy/interaction.py`
- Test: `tests/test_synergy_interaction.py`

**Interfaces:**
- Consumes: `synergy.counts.{cond_prob, cooc_score}`.
- Produces:
  - `synergy_vectors(B, obs_pair: tuple[int,int], n_bundles: int) -> dict[str, np.ndarray]` — for the two observed items `(i,k)`, returns per-candidate-`j` vectors:
    `{"joint": P(j|{i,k}), "additive": P(j|i)+P(j|k)-P(j), "tau": joint-(P(j|i)+P(j|k))+P(j), "p_i": P(j|i), "p_k": P(j|k), "p_marg": P(j)}`. Observed `i,k` entries of `joint/additive/tau` set to `-inf`/`nan` per key (see impl).
  - `null_bundles_product_of_marginals(bundles, n_items, sizes, rng) -> list[list[int]]` — placebo bundles drawn item-independently from the marginal, preserving per-bundle size (the additivity null).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synergy_interaction.py
import numpy as np
from synergy.counts import item_bundle_incidence
from synergy.interaction import synergy_vectors, null_bundles_product_of_marginals

def test_pure_interaction_gives_positive_tau():
    # item 3 appears IFF both 0 and 1 present; never with only one of them.
    bundles = [[0, 1, 3]] * 20 + [[0, 2]] * 20 + [[1, 2]] * 20
    n_items = 4
    B = item_bundle_incidence(bundles, n_items)
    v = synergy_vectors(B, (0, 1), n_bundles=len(bundles))
    # synergy for item 3 is strongly positive: joint high, but P(3|0)=P(3|1)~1/2, additive underestimates
    assert v["tau"][3] > 0.15
    # joint conditional ranks item 3 top among non-observed candidates (2 and 3)
    assert v["joint"][3] > v["joint"][2]

def test_additive_data_gives_near_zero_tau():
    # items 2 and 3 each appear independently of the (0,1) co-presence -> tau ~ 0
    rng = np.random.default_rng(0)
    bundles = []
    for _ in range(400):
        b = [0, 1]
        if rng.random() < 0.5: b.append(2)
        if rng.random() < 0.5: b.append(3)
        bundles.append(b)
    n_items = 4
    B = item_bundle_incidence(bundles, n_items)
    v = synergy_vectors(B, (0, 1), n_bundles=len(bundles))
    assert abs(v["tau"][2]) < 0.08 and abs(v["tau"][3]) < 0.08

def test_null_bundles_preserve_sizes_and_are_additive():
    rng = np.random.default_rng(1)
    bundles = [[0, 1, 3]] * 20 + [[0, 2]] * 20 + [[1, 2]] * 20
    null = null_bundles_product_of_marginals(bundles, n_items=4, sizes=[len(b) for b in bundles], rng=rng)
    assert [len(b) for b in null] == [len(b) for b in bundles]     # sizes preserved
    B = item_bundle_incidence(null, 4)
    v = synergy_vectors(B, (0, 1), n_bundles=len(null))
    # on additive/independent null data the interaction estimate is small
    assert abs(v["tau"][3]) < 0.12
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_interaction.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'synergy.interaction'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
    """Placebo: redraw each bundle's items independently from the item marginal (no replacement
    within a bundle), preserving per-bundle size. Breaks any i-k-j interaction while keeping
    marginals ~fixed -> the additivity null for the synergy estimator."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_interaction.py -q
```
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/synergy/interaction.py tests/test_synergy_interaction.py
git commit -m "feat(synergy): pairwise synergy contrast tau_syn(j|{i,k}) + additivity-null placebo"
```

---

### Task 3: Rank-inversion & completion metrics

**Files:**
- Create: `src/synergy/rankinv.py`
- Test: `tests/test_synergy_rankinv.py`

**Interfaces:**
- Produces:
  - `item_rank(scores: np.ndarray, item: int) -> float` — 1-based average-tie rank (higher score = better) (copied from `cec.stratify`).
  - `hit_at_k(scores: np.ndarray, item: int, k: int) -> float` — `1.0` if `item_rank <= k` else `0.0`.
  - `rank_agreement(scores_a, scores_b, candidates: np.ndarray, topk: int) -> dict` — restricted to `candidates`, returns `{"spearman", "kendall", "topk_jaccard"}` comparing the two score orderings. `topk_jaccard` = Jaccard of the top-`topk` item sets.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synergy_rankinv.py
import numpy as np
from synergy.rankinv import item_rank, hit_at_k, rank_agreement

def test_item_rank_and_hit():
    s = np.array([0.9, 0.5, 0.9, 0.1])
    assert item_rank(s, 3) == 4.0
    assert item_rank(s, 0) == 1.5       # tied top -> average-tie rank 1.5
    assert hit_at_k(s, 0, 1) == 0.0     # avg-tie rank 1.5 is NOT <= 1 (conservative, matches cec.stratify)
    assert hit_at_k(s, 0, 2) == 1.0     # 1.5 <= 2
    assert hit_at_k(s, 3, 1) == 0.0

def test_rank_agreement_identical_is_one():
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([6.0, 4.0, 2.0, 0.0])   # monotone reweight of a -> same ranking
    cand = np.array([0, 1, 2, 3])
    r = rank_agreement(a, b, cand, topk=2)
    assert abs(r["spearman"] - 1.0) < 1e-9
    assert abs(r["kendall"] - 1.0) < 1e-9
    assert r["topk_jaccard"] == 1.0      # same top-2 => inert reweight

def test_rank_agreement_reversed_is_negative():
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])   # reversed
    cand = np.array([0, 1, 2, 3])
    r = rank_agreement(a, b, cand, topk=2)
    assert r["spearman"] < -0.9
    assert r["topk_jaccard"] == 0.0      # disjoint top-2 => strong rank inversion
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_rankinv.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'synergy.rankinv'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/synergy/rankinv.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_rankinv.py -q
```
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/synergy/rankinv.py tests/test_synergy_rankinv.py
git commit -m "feat(synergy): rank-agreement + completion metrics for rank-inversion test"
```

---

### Task 4: Gate-metric aggregation over a dataset (pure, testable)

**Files:**
- Create: `src/synergy/gate.py`
- Test: `tests/test_synergy_gate.py`

**Interfaces:**
- Consumes: `synergy.counts.{item_bundle_incidence, cooc_score, ppmi_score}`, `synergy.interaction.{synergy_vectors, null_bundles_product_of_marginals}`, `synergy.rankinv.{item_rank, hit_at_k, rank_agreement}`.
- Produces:
  - `run_gate(train_bundles, test_obs, test_gt, n_items, *, ks=(1,20), topk=200, rng) -> dict` — computes, over test instances whose observed set has ≥2 items (using the first two observed items as the `(i,k)` pair):
    - `tau_abs_mean` — mean over instances of `mean_j |tau_syn(j)|` across the top-`topk` co-occ candidates (the residual-magnitude statistic);
    - `tau_abs_mean_null` — same statistic on product-of-marginals placebo bundles;
    - `rank` — mean Spearman / mean top-20 Jaccard between the **joint-conditional** score (`additive+tau`) and the **co-occ** score over top-`topk` co-occ candidates;
    - `hit` — per-`k` gt hit rate under co-occ vs under joint-conditional, and the per-instance paired arrays (for Wilcoxon in the runner);
    - `masked_frac` — fraction of instances where the gt item has `tau_syn` in its top decile among candidates AND co-occ rank > 100 (a confounding-masked complement co-occ misses);
    - `n_instances`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synergy_gate.py
import numpy as np
from synergy.gate import run_gate

def test_gate_detects_synergy_on_interaction_data():
    # gt item 3 appears IFF both observed items {0,1}; co-occ alone under-ranks it vs item 2 noise
    train = [[0, 1, 3]] * 40 + [[0, 2]] * 40 + [[1, 2]] * 40 + [[2, 3]] * 5
    test_obs = [[0, 1]] * 10
    test_gt = [[3]] * 10
    rng = np.random.default_rng(0)
    out = run_gate(train, test_obs, test_gt, n_items=4, ks=(1, 2), topk=3, rng=rng)
    assert out["n_instances"] == 10
    assert out["tau_abs_mean"] > out["tau_abs_mean_null"]       # residual above additivity null
    assert set(out["hit"].keys()) == {1, 2}
    assert len(out["hit"][1]["cooc"]) == 10                     # per-instance paired arrays present

def test_gate_skips_instances_with_fewer_than_two_observed():
    train = [[0, 1, 3]] * 10
    out = run_gate(train, [[0]], [[3]], n_items=4, ks=(1,), topk=3, rng=np.random.default_rng(0))
    assert out["n_instances"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_gate.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'synergy.gate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/synergy/gate.py
from __future__ import annotations
import numpy as np
from synergy.counts import item_bundle_incidence, cooc_score
from synergy.interaction import synergy_vectors, null_bundles_product_of_marginals
from synergy.rankinv import item_rank, hit_at_k, rank_agreement


def _instances(test_obs, test_gt):
    for obs, gt in zip(test_obs, test_gt):
        if len(obs) >= 2 and gt:
            yield obs[0], obs[1], obs, gt


def _tau_abs_over_candidates(B, obs_pair, n_bundles, topk):
    """mean_j |tau_syn(j)| over the top-`topk` co-occ candidates for this instance."""
    v = synergy_vectors(B, obs_pair, n_bundles)
    cs = cooc_score(B, list(obs_pair), B.shape[0])
    cand = np.argsort(-cs)[:topk]
    tau = v["tau"][cand]
    tau = tau[np.isfinite(tau)]
    return float(np.mean(np.abs(tau))) if tau.size else 0.0, v, cs, cand


def run_gate(train_bundles, test_obs, test_gt, n_items, *, ks=(1, 20), topk=200, rng):
    B = item_bundle_incidence(train_bundles, n_items)
    n_bundles = B.shape[1]
    inst = list(_instances(test_obs, test_gt))

    # placebo (additivity null) incidence, same sizes
    sizes = [len(b) for b in train_bundles]
    null_bundles = null_bundles_product_of_marginals(train_bundles, n_items, sizes, rng)
    Bn = item_bundle_incidence(null_bundles, n_items)

    tau_abs, tau_abs_null = [], []
    sp_list, jac_list, masked = [], [], []
    hit = {k: {"cooc": [], "joint": []} for k in ks}

    for i, k_, obs, gt in inst:
        ta, v, cs, cand = _tau_abs_over_candidates(B, (i, k_), n_bundles, topk)
        tau_abs.append(ta)
        tn, _, _, _ = _tau_abs_over_candidates(Bn, (i, k_), Bn.shape[1], topk)
        tau_abs_null.append(tn)

        joint = v["joint"].copy()          # additive + tau == joint conditional P(j|{i,k})
        ra = rank_agreement(joint, cs, cand, topk=20)
        if np.isfinite(ra["spearman"]):
            sp_list.append(ra["spearman"]); jac_list.append(ra["topk_jaccard"])

        for g in gt:
            for k in ks:
                hit[k]["cooc"].append(hit_at_k(cs, g, k))
                hit[k]["joint"].append(hit_at_k(joint, g, k))
            # masked-complement: gt has high tau but low co-occ rank
            tau_g = v["tau"][g]
            finite_tau = v["tau"][np.isfinite(v["tau"])]
            hi_tau = np.isfinite(tau_g) and tau_g >= np.percentile(finite_tau, 90)
            lo_cooc = item_rank(cs, g) > 100
            masked.append(1.0 if (hi_tau and lo_cooc) else 0.0)

    return {
        "n_instances": len(inst),
        "tau_abs_mean": float(np.mean(tau_abs)) if tau_abs else 0.0,
        "tau_abs_mean_null": float(np.mean(tau_abs_null)) if tau_abs_null else 0.0,
        "rank": {"spearman_mean": float(np.mean(sp_list)) if sp_list else float("nan"),
                 "top20_jaccard_mean": float(np.mean(jac_list)) if jac_list else float("nan")},
        "hit": {k: {"cooc": hit[k]["cooc"], "joint": hit[k]["joint"],
                    "cooc_mean": float(np.mean(hit[k]["cooc"])) if hit[k]["cooc"] else 0.0,
                    "joint_mean": float(np.mean(hit[k]["joint"])) if hit[k]["joint"] else 0.0}
                for k in ks},
        "masked_frac": float(np.mean(masked)) if masked else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_gate.py -q
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/synergy/gate.py tests/test_synergy_gate.py
git commit -m "feat(synergy): dataset gate-metric aggregation (tau residual, rank-inversion, masked slice)"
```

---

### Task 5: The runner — measure the gate on MealRec+ and write the verdict JSON

**Files:**
- Create: `scripts/synergy_m1_gate.py`
- Test: `tests/test_synergy_runner_smoke.py`

**Interfaces:**
- Consumes: `synergy.gate.run_gate`, predecessor `causalbr.eval.significance.paired_wilcoxon`.
- Produces: a `main()` CLI writing `results/synergy/m1/<dataset>.json` with `{dataset, n_items, n_instances, tau residual + null, rank agreement, hit deltas + wilcoxon p, masked_frac, verdict}` and a `decide(metrics) -> dict` pure function applying the Task-6 pre-registered thresholds.

- [ ] **Step 1: Write the failing test** (smoke-test `decide` on synthetic metrics; no data files needed)

```python
# tests/test_synergy_runner_smoke.py
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "synergy_m1_gate",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "synergy_m1_gate.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _metrics(**over):
    m = {"tau_abs_mean": 0.10, "tau_abs_mean_null": 0.02, "tau_perm_p": 0.001,
         "rank": {"spearman_mean": 0.60, "top20_jaccard_mean": 0.50},
         "hit": {20: {"cooc_mean": 0.30, "joint_mean": 0.38, "wilcoxon_p": 0.001}},
         "masked_frac": 0.05, "placebo_ok": True}
    m.update(over); return m

def test_decide_go_when_all_pass():
    d = mod.decide(_metrics())
    assert d["verdict"] == "GO"

def test_decide_kill_when_rank_equivalent():
    d = mod.decide(_metrics(rank={"spearman_mean": 0.95, "top20_jaccard_mean": 0.95}))
    assert d["verdict"] == "KILL"
    assert "rank_equivalent_to_cooc" in d["failed"]

def test_decide_kill_when_residual_at_null():
    d = mod.decide(_metrics(tau_perm_p=0.9))
    assert d["verdict"] == "KILL"
    assert "residual_at_null" in d["failed"]

def test_decide_kill_when_placebo_miscalibrated():
    d = mod.decide(_metrics(placebo_ok=False))
    assert d["verdict"] == "KILL"
    assert "placebo_miscalibrated" in d["failed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_runner_smoke.py -q
```
Expected: FAIL with `FileNotFoundError`/`AttributeError: module ... has no attribute 'decide'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/synergy_m1_gate.py
"""Synergy falsification gate (Design 1, slice-1): measure whether within-set synergy
tau_syn(j|{i,k}) carries non-additive completion signal co-occ cannot represent. CPU-only,
full-catalog, no training. KILL-or-PROCEED per the pre-registered thresholds in `decide`.

Run (predecessor venv + its src on PYTHONPATH):
  PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
  /Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
  scripts/synergy_m1_gate.py \
    --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/mealrec_l \
    --dataset mealrec_l --topk 200 --seed 0 --out results/synergy/m1/mealrec_l.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synergy.gate import run_gate
from causalbr.eval.significance import paired_wilcoxon


# ---- PRE-REGISTERED THRESHOLDS (frozen; see plan Task 6) ----
TAU_PERM_P_MAX = 0.05        # residual must beat the additivity-null permutation (one-sided)
RANK_SPEARMAN_MAX = 0.90     # >= this AND jaccard >= JACCARD_MAX => rank-equivalent (inert)
RANK_JACCARD_MAX = 0.90
HIT_WILCOXON_P_MAX = 0.05    # joint-conditional > co-occ on gt hit@20 (one-sided)
MASKED_FRAC_MIN = 0.01       # >=1% masked-complement instances
PLACEBO_TAU_RATIO_MAX = 0.5  # placebo tau must be <= half the observed tau (calibration)


def decide(m: dict) -> dict:
    """Apply the frozen thresholds. KILL if inert/miscalibrated; else GO (STRONG/WEAK)."""
    failed = []
    if m["tau_perm_p"] > TAU_PERM_P_MAX:
        failed.append("residual_at_null")
    if (m["rank"]["spearman_mean"] >= RANK_SPEARMAN_MAX
            and m["rank"]["top20_jaccard_mean"] >= RANK_JACCARD_MAX):
        failed.append("rank_equivalent_to_cooc")
    if not m.get("placebo_ok", False):
        failed.append("placebo_miscalibrated")
    if failed:
        return {"verdict": "KILL", "failed": failed}
    strong = (m["hit"][20]["wilcoxon_p"] <= HIT_WILCOXON_P_MAX
              and m["hit"][20]["joint_mean"] > m["hit"][20]["cooc_mean"]
              and m["masked_frac"] >= MASKED_FRAC_MIN)
    return {"verdict": "GO", "strength": "STRONG" if strong else "WEAK", "failed": []}


def _load_mealrec(data_dir):
    tr = np.load(os.path.join(data_dir, "train_examples.npy"), allow_pickle=True)
    te = np.load(os.path.join(data_dir, "test_examples.npy"), allow_pickle=True)
    train = [list(row[1]) for row in tr]
    obs = [list(row[1]) for row in te]
    gt = [list(row[2]) for row in te]
    n_items = 1 + max(max((max(b) for b in train if b), default=0),
                      max((max(b) for b in obs if b), default=0),
                      max((max(g) for g in gt if g), default=0))
    return train, obs, gt, int(n_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, obs, gt, n_items = _load_mealrec(args.data_dir)
    rng = np.random.default_rng(args.seed)
    g = run_gate(train, obs, gt, n_items, ks=tuple(args.ks), topk=args.topk, rng=rng)

    # residual vs null: one-sided paired Wilcoxon over per-instance tau_abs vs null
    # (recompute paired arrays through run_gate is overkill; use the summary + a permutation p)
    # permutation p: fraction of null instances whose tau_abs >= observed mean.
    tau_perm_p = 1.0
    if g["n_instances"]:
        # conservative: compare observed mean to null distribution mean via Wilcoxon on the
        # per-instance arrays returned below; run_gate exposes them via re-run in strict mode.
        tau_perm_p = _tau_perm_p(train, obs, gt, n_items, args.topk, args.seed, g["tau_abs_mean"])

    # completion lift: paired Wilcoxon joint > cooc on hit@20
    h20 = g["hit"][20]
    wp = paired_wilcoxon(h20["joint"], h20["cooc"], alternative="greater")["p_value"] \
        if h20["cooc"] else 1.0
    g["hit"][20]["wilcoxon_p"] = float(wp)

    placebo_ok = (g["tau_abs_mean_null"] <= PLACEBO_TAU_RATIO_MAX * max(g["tau_abs_mean"], 1e-12))

    metrics = {
        "tau_abs_mean": g["tau_abs_mean"], "tau_abs_mean_null": g["tau_abs_mean_null"],
        "tau_perm_p": float(tau_perm_p), "rank": g["rank"],
        "hit": {20: {"cooc_mean": h20["cooc_mean"], "joint_mean": h20["joint_mean"],
                     "wilcoxon_p": float(wp)}},
        "masked_frac": g["masked_frac"], "placebo_ok": bool(placebo_ok),
    }
    verdict = decide(metrics)
    out = {"dataset": args.dataset, "n_items": n_items, "n_instances": g["n_instances"],
           "topk": args.topk, "seed": args.seed, "metrics": metrics,
           "hit_full": {k: {"cooc_mean": g["hit"][k]["cooc_mean"],
                            "joint_mean": g["hit"][k]["joint_mean"]} for k in args.ks},
           "verdict": verdict}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"dataset": args.dataset, "n_instances": g["n_instances"],
                      "metrics": metrics, "verdict": verdict}, indent=2))


def _tau_perm_p(train, obs, gt, n_items, topk, seed, observed_mean):
    """One-sided permutation p: build R additivity-null datasets, compute the tau_abs_mean
    statistic on each; p = (#{null_mean >= observed_mean} + 1)/(R+1)."""
    from synergy.gate import run_gate as _rg
    R = 19
    ge = 0
    for r in range(R):
        rng = np.random.default_rng(seed * 1000 + r + 1)
        gr = _rg(train, obs, gt, n_items, ks=(20,), topk=topk, rng=rng)
        # run_gate already compares to its own null; here we want the null statistic itself:
        if gr["tau_abs_mean_null"] >= observed_mean:
            ge += 1
    return (ge + 1) / (R + 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/test_synergy_runner_smoke.py -q
```
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/synergy_m1_gate.py tests/test_synergy_runner_smoke.py
git commit -m "feat(synergy): m1 gate runner + pre-registered GO/KILL decision rule"
```

---

### Task 6: Freeze the pre-registered thresholds in the plan + full test run

The thresholds are already encoded as module constants in `scripts/synergy_m1_gate.py` (Task 5). This task records them as the frozen contract and verifies the whole suite is green before any data run.

**Pre-registered GO/KILL contract (frozen 2026-07-01, do NOT tune post-hoc):**

- **KILL if ANY:**
  1. `residual_at_null` — `tau_perm_p > 0.05` (synergy residual not above the additivity-null band).
  2. `rank_equivalent_to_cooc` — `spearman_mean ≥ 0.90` AND `top20_jaccard_mean ≥ 0.90` (joint-conditional ranking is a monotone reweight of co-occ = inert in effect).
  3. `placebo_miscalibrated` — placebo `tau_abs_mean_null > 0.5 × tau_abs_mean` (estimator reports interaction on additive data → not trustworthy).
- **GO otherwise**, graded:
  - **STRONG-GO** — additionally `hit@20 wilcoxon_p ≤ 0.05` with `joint_mean > cooc_mean` AND `masked_frac ≥ 0.01` (synergy lifts completion AND finds confounding-masked complements co-occ misses).
  - **WEAK-GO** — non-inert and well-calibrated, but no completion lift yet (proceed to the build plan with caution; the build must earn the lift).

- [ ] **Step 1: Run the full unit-test suite**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python -m pytest tests/ -q
```
Expected: PASS (all synergy tests + the pre-existing `test_exposure.py`, `test_stratify.py`, `test_smoke.py`).

- [ ] **Step 2: Commit (thresholds frozen)**

```bash
git commit --allow-empty -m "chore(synergy): freeze pre-registered GO/KILL thresholds before data run"
```

---

### Task 7: Run the gate on MealRec+ L and H, record the verdict

**Files:**
- Create: `results/synergy/m1/mealrec_l.json` (runner output)
- Create: `results/synergy/m1/mealrec_h.json` (runner output)
- Create: `docs/findings/2026-07-01-synergy-gate-verdict.md`

- [ ] **Step 1: Run the gate on MealRec+ L**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
scripts/synergy_m1_gate.py \
  --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/mealrec_l \
  --dataset mealrec_l --topk 200 --seed 0 --out results/synergy/m1/mealrec_l.json
```
Expected: prints a JSON blob ending in `"verdict": {"verdict": "GO"|"KILL", ...}`. Runtime: minutes on CPU.

- [ ] **Step 2: Run the gate on MealRec+ H**

Run:
```
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
scripts/synergy_m1_gate.py \
  --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/mealrec_h \
  --dataset mealrec_h --topk 200 --seed 0 --out results/synergy/m1/mealrec_h.json
```
Expected: prints the verdict JSON; writes the file.

- [ ] **Step 3: Write the findings doc** (record the numbers verbatim from both JSONs and the decision)

Create `docs/findings/2026-07-01-synergy-gate-verdict.md` with this structure, filling every `<...>` from the two result files (no placeholders left):

```markdown
# Synergy falsification gate — verdict (2026-07-01)

Design 1 slice-1 (Egami-Imai within-set synergy `τ_syn(j|{i,k})`), CPU-only, full-catalog,
no training. Pre-registered thresholds frozen in `scripts/synergy_m1_gate.py` (plan Task 6).

## Results

| dataset | n_inst | tau_abs (obs / null) | tau_perm_p | spearman / jac20 | hit@20 cooc→joint (wilcoxon p) | masked_frac | verdict |
|---|---|---|---|---|---|---|---|
| mealrec_l | <..> | <..> / <..> | <..> | <..> / <..> | <..>→<..> (<..>) | <..> | <STRONG/WEAK GO or KILL> |
| mealrec_h | <..> | <..> / <..> | <..> | <..> / <..> | <..>→<..> (<..>) | <..> | <...> |

## Decision

<One paragraph: which KILL conditions fired (if any), or GO strength. State the next action:
if KILL — Design 1 is closed here; report the characterized negative and pivot to Design 2
(anonymized-LLM). If GO — author the reranker build plan (separate), regime = dense, and the
build must clear the full baseline suite + inertness/do()-flip battery.>
```

- [ ] **Step 4: Commit**

```bash
git add results/synergy/m1/mealrec_l.json results/synergy/m1/mealrec_h.json docs/findings/2026-07-01-synergy-gate-verdict.md
git commit -m "test(synergy): m1 gate verdict on MealRec+ L/H (pre-registered)"
```

---

## What happens after this plan

- **KILL** (either dataset inert / rank-equivalent / placebo-miscalibrated per the frozen rule):
  Design 1 is cheaply closed on dense data — record the characterized negative (like M0.5) and pivot
  to Design 2 (anonymized-LLM complementarity), whose own gate is its slice-1.
- **GO** (STRONG on ≥1 dense dataset): author a **separate** reranker build plan — non-additive
  seed-set→completion scorer reusing predecessor `recovery/reranker.py`, full baseline suite (co-occ,
  PPMI, retrieve-rerank, Cadence-UACR, content SetCompleter, CLHE, DDBC-full), the inertness /
  rank-inversion / `do()`-flip battery, ≥5 seeds paired Wilcoxon, and the L-I Rosenbaum-Γ bounds. That
  plan decides the decode regime (dense; consider constructing `m≥2` splits or a smaller candidate pool
  to raise decode SNR).

## Self-Review

- **Spec coverage:** This plan implements spec §4.8 (the up-front CPU falsification gate) and §9 (first
  slice = the gate). The residual-magnitude, rank-inversion (Kendall/Spearman + top-k), masked-complement
  slice, and additivity-null placebo of §4.8 all map to Tasks 4–6. Point-estimate identification and the
  build/bounds (spec §4.3–4.9) are explicitly deferred to the post-GO build plan (stated above) — correct,
  since the gate is the kill-or-proceed slice.
- **Placeholder scan:** No TBD/TODO. The only `<...>` are in the Task-7 findings template, to be filled
  verbatim from the two result JSONs during execution (a data-transcription step, not a code placeholder).
- **Type consistency:** `synergy_vectors` returns the `{joint,additive,tau,p_i,p_k,p_marg}` dict used by
  `gate.run_gate`; `run_gate` returns the `{tau_abs_mean, tau_abs_mean_null, rank, hit, masked_frac,
  n_instances}` dict consumed by the runner's `metrics`/`decide`; `decide` reads exactly the keys the
  runner builds (`tau_perm_p`, `rank.spearman_mean`, `rank.top20_jaccard_mean`, `hit[20].wilcoxon_p`,
  `hit[20].joint_mean/cooc_mean`, `masked_frac`, `placebo_ok`). `item_rank`/`hit_at_k` signatures match
  across `rankinv` and `gate`.
```
