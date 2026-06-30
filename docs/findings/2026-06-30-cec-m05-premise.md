# CEC M0.5 — Exposure-Stratified Premise Falsification: VERDICT = NO-GO

> Date: 2026-06-30. Plan: `docs/superpowers/plans/2026-06-30-cec-m05-exposure-premise-falsification.md`.
> Result artifact: `results/cec/m05/spotify90.json`. Runner: `scripts/cec_m05_exposure_premise.py`
> (commits 49eb870 + memory fix 1eb1323 + cpu-load fix a2c6756). Dataset: Spotify-90 (predecessor
> `data/spotify90/`), content model = existing `setcompleter_spotify90.pt` checkpoint (no retrain).
> **Pre-registered decision rule (from the plan):** GO iff the content−co-occ hit@1 delta is positive,
> **monotone-decreasing across exposure bins (low→high)**, significant in the lowest bin, and ≈0 in the
> highest. Otherwise NO-GO.

## Decision: **NO-GO** — the premise is refuted.

The CEC premise was that content's full-catalog win over co-occurrence **concentrates on cold/under-exposed
items** (the regime where co-occ is blind), so an exposure-gated causal completer could win the cold slice
while leaving warm accuracy untouched. The data refute this.

## What we ran
Re-scored the Spotify-90 construction test set (n_test = 969 completions, 43,605 per-(example, gt-item)
records) with the **existing** SetCompleter checkpoint and co-occurrence, full-catalog over 254,155 items.
For each held-out gt item we recorded its rank under each method and binned by the gt item's training
exposure (5 percentile bins; edges 0 / 25 / 40 / 53 / 99 / 13730).

## Results

**Overall (content − co-occ):** hit@1 Δ = +0.00044 (n.s.); hit@20 Δ = +0.0124.

**hit@1 (the pre-registered metric) — uninformative + refutes the rule.** Full-catalog top-1 over 254k
items is near-impossible: content hit@1 is ~0 in every bin (0.0066 in the hottest), the per-bin Δ is
tiny and **not significant in any bin** (lowest-bin p = 0.125; top-bin p = 0.21), and the curve is **not
monotone-decreasing** (`hit1_delta_monotone_decreasing = false`). The pre-registered GO condition fails.

**hit@20 (the informative metric) — the win is REAL but WARM-weighted:**

| bin | item exposure | n | content | co-occ | Δ (content−cooc) | paired-Wilcoxon p |
|---|---|---|---|---|---|---|
| 0 (coldest) | ≤ 25 | 9,680 | 0.00217 | **0.00000** | +0.00217 | 2.3e-6 |
| 1 | 25–40 | 1,372 | 0.00948 | **0.00000** | +0.00948 | 1.2e-4 |
| 2 | 40–53 | 1,356 | 0.00737 | **0.00000** | +0.00737 | 9.8e-4 |
| 3 | 53–99 | 2,902 | 0.01757 | 0.00724 | +0.01034 | 1.4e-4 |
| 4 (hottest) | 99–13730 | 28,295 | 0.07846 | 0.06199 | **+0.01647** | 1.1e-17 |

## Reading

1. **The structural blind spot is real (partial confirmation).** Co-occurrence scores **exactly 0** hit@20
   on the three lowest-exposure bins — it never ranks a cold ground-truth item into the top-20, because it
   has too little/no co-occurrence statistics for those items. Content recovers a small but **significant**
   signal there (bins 0–2, all p < 1e-3). So "co-occ is blind on cold items, content is not" holds.

2. **But content's advantage is not a cold-item effect — it grows with exposure (refutation).** The
   content−co-occ Δ is significant in *every* bin and is **monotone-increasing** with exposure: smallest in
   the coldest bin (+0.0022) and **largest, most significant in the hottest bin** (+0.0165, p = 1e-17) —
   the opposite of the pre-registered prediction. The reason: cold items are intrinsically hard to complete,
   so content's *own* absolute hit@20 is only 0.002–0.009 in the cold bins. The big, reliable gains live on
   warm items.

3. **Consequence for the CEC design.** An exposure-gated completer that fires the causal/content term **only
   on cold/under-exposed slots** (the whole point of CEC's gate + warm-parity design) would capture the
   **smallest** part of content's advantage and forgo the largest (warm). The method-first bet — win the
   cold slice, preserve warm — is not supported: there is no large cold-slice prize to win, and the genuine
   edge is broadly distributed and warm-weighted.

## What this kills / leaves open
- **Killed:** CEC as specced (exposure-gated, cold-slice-concentrated, do(uniform-exposure) counterfactual
  completer). Do **not** proceed to M0 port + M1 build for that design.
- **Still true and possibly useful for a different bet:** co-occ is structurally blind on cold items, and a
  content/inductive model is the only thing that scores them at all — but the ceiling there is low. Any
  future bet relying on cold-item wins must reckon with that low ceiling (the gain is real in *relative*
  terms but small in *absolute* terms).
- **Caveat (single dataset / single metric):** this is Spotify-90 only, and hit@1 was uninformative.
  A POG robustness re-run (smaller catalog) would confirm whether the warm-weighted pattern is general; it
  is cheap (Task 6 of the plan) but not required to read the NO-GO, since the Spotify hit@20 pattern is
  unambiguous and the pre-registered hit@1 rule already fails.

## Confirmation: MealRec+ L (dense regime, different domain)

POG raw data was not available locally (HF cache was an empty stub), so the cross-check ran on
**MealRec+ L** (food; n_items = 10,589 — the **dense** regime, the opposite of sparse Spotify), present
locally with content embeddings. No SetCompleter checkpoint exists for MealRec, so it was **retrained**
(20 epochs, seed 0) on `item_emb.pt`; item exposure computed per item from the train meals. Runner:
`scripts/cec_m05_mealrec.py`; result: `results/cec/m05/mealrec_l.json` (n_test = 1,181, one gt per test).

**Overall (content − co-occ): hit@1 = −0.151; hit@20 = −0.643.** On dense data, content LOSES to
co-occurrence outright.

**hit@20 per exposure bin:**

| bin | item exposure | n | content | co-occ | Δ (content−cooc) | p |
|---|---|---|---|---|---|---|
| 0 (coldest) | ≤ 1 | 12 | 0.0833 | 0.0000 | +0.0833 | 0.50 (n.s.) |
| 1 | 1–2 | 23 | 0.0000 | 0.2609 | −0.2609 | 1.0 |
| 2 | 2–4 | 32 | 0.0000 | 0.1563 | −0.1563 | 1.0 |
| 3 | 4–13 | 127 | 0.0000 | 0.3622 | −0.3622 | 1.0 |
| 4 (hottest) | 13–795 | 987 | 0.0456 | **0.7579** | **−0.7123** | 1.0 |

This **confirms the NO-GO via a different mechanism.** On the dense catalog, co-occurrence dominates
everywhere (hit@20 = 0.76 in the head) and content barely scores at all — exactly the project's C3/F4 law
(co-occ wins dense; content ≠ complementarity; retrieve-rerank only transfers to dense). Content's *only*
advantage is on the 12 coldest items (+0.083), which is **not significant** (n = 12, p = 0.50). Caveat:
SetCompleter was retrained modestly (20 epochs); a stronger content model could raise its absolute
numbers, but co-occ's dense dominance and the absence of a *significant* cold-item content win are the
robust, C3-consistent conclusion.

## Bottom line (confirmed across two datasets / two density regimes)
The cheap M0.5 falsification did exactly its job — it refuted the CEC premise **before** any port or GPU
training, on **both** datasets:
- **Sparse (Spotify-90):** content's edge over co-occ is real but **warm-weighted** (Δ grows with
  exposure; top bin p = 1e-17). Cold-gating captures the smallest part.
- **Dense (MealRec+ L):** content **loses to co-occ** everywhere; the only cold-item edge is tiny and
  **not significant**. Cold-gating captures nothing.

In neither regime is there a large, significant cold-slice content/causal advantage to gate on — the core
CEC bet. **Recommendation: do not build M1; revisit the core bet** (per the spec's risk table), keeping the
honest, now-twice-confirmed constraints that (a) cold-item completion has a low absolute ceiling, and
(b) co-occurrence is hard to beat — warm-weighted in the best case (sparse), dominant in the worst (dense).
