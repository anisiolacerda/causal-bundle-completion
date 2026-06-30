# Strategic re-think after CEC (the contribution question)

> Date: 2026-06-30. Trigger: the CEC M0.5 falsification refuted the cold-item bet across two datasets
> ([2026-06-30-cec-m05-premise.md](2026-06-30-cec-m05-premise.md)). This note records *why* and what the
> defensible contribution likely is, so the next session does not re-walk dead ground.
> (The predecessor project is unpublished; its results below are inherited internal givens, not citations.)

## The pattern: 0/2 causal-method bets, one wall

Two designed causal-method bets, both killed cheaply *before* GPU/port spend:

1. **FD-Set (front-door SCM over the partial-bundle mediator)** — killed at design time by the
   predecessor's own adversarial ruling: the **textbook front-door is mis-specified** for completion (a
   dominant direct seed→item compatibility path exists), the user-theme mediator was already **refuted**
   (C4), and popularity-deconfounding of co-occurrence collapses to **prior art** (Cadence) plus the
   known sparse negative (C3 / PPMI).
2. **CEC (cold-item / `do(uniform exposure)` counterfactual completion)** — killed by the M0.5
   falsification on Spotify-90 (sparse) **and** MealRec+ L (dense): the content/causal edge is
   warm-weighted where it exists (sparse) and absent where co-occ dominates (dense); the cold-item
   advantage is real only in *relative* terms and is **not significant** in absolute terms.

These failures are **consistent, not random**. Every causal angle tried has hit one of four fates:

| causal angle | fate | evidence |
|---|---|---|
| user-theme `do(seed)` | **inert** at decode | C4 (τ0_flip ~2% vs ~99% null) |
| popularity-deconfounding of co-occ | **prior art** + PPMI-collapse | Cadence (2512.17733); C3 sparse |
| total-effect front-door | **mis-specified** | predecessor front-door adversarial ruling |
| cold-item counterfactual completion | **low-ceiling / wrong slice** | M0.5 ×2 (this work) |

**The single cause: co-occurrence is an extremely hard full-catalog wall.** It is warm-weighted-beatable
at best (sparse) and dominant at worst (dense), and a causal term that is not inert tends to be either
already-published or to capture a slice too small to move the headline.

## What this implies for the contribution

The project's stated goal — *a new causal method that beats co-occ / retrieve-rerank full-catalog AND
shows a verified causal gain* — now has substantial, cheaply-bought evidence against its feasibility on
these datasets. The honest, defensible contribution is more likely:

- **A benchmark + a characterized NEGATIVE result.** "Across the major causal-recsys families
  (debiasing, deconfounding, counterfactual/uplift, front-door, cold-item counterfactual), no causal
  term beats trivial co-occurrence on full-catalog bundle construction — and here is the rigorous *why*,
  family by family, with the inertness / prior-art / mis-specification / low-ceiling mechanism for each."
  This is a real D&B / negative-results contribution, and most of its substance is already established
  (the predecessor's C1–C4 + the FD-Set adversarial ruling + this M0.5 work).
- The cheap-falsification methodology itself (the inertness battery + the exposure-stratified premise
  gate that killed CEC for ~minutes of CPU) is a reusable, citable methodological contribution.

## Concrete options for the next session (do not start without brainstorming)

1. **Re-frame the paper as benchmark + characterized-negative** (recommended by the evidence). Assemble
   C1–C4 + the two killed bets into one rigorous "causal terms don't beat co-occ in bundle construction,
   and why" story. Lowest risk; largely already proven.
2. **Change the task so co-occ is not the wall.** Pursue a setting co-occ structurally cannot address
   *and* that has a non-trivial ceiling — candidates from the backlog: variable-k construction
   (DDBC fixes k), or a genuinely interventional/online objective. Note the cold-item route is now closed
   (low ceiling, confirmed).
3. **Drop "beat co-occ" as the bar** and target a different, honestly-motivated quantity (e.g. diversity,
   constraint-satisfaction, or a deployment metric) where a causal term has room — but this departs from
   the project's current accuracy-completion framing.

## Reusable assets produced here
- `src/cec/` — exposure binning + per-bin paired stratified stats (pure, tested).
- `scripts/cec_m05_exposure_premise.py` (Spotify, memory-safe co-occ via incidence mat-vec) and
  `scripts/cec_m05_mealrec.py` (dense regime, retrains SetCompleter). Both reusable for any future
  exposure-stratified completion analysis.
- `results/cec/m05/*.json` + the verdict doc — the negative result, with numbers.

## Bottom line
Two bets, two cheap kills, one wall. The methodology worked exactly as intended (fail fast, before GPU).
The recommended pivot is from "find a winning causal method" to "rigorously characterize why causal terms
lose to co-occurrence in bundle construction" — a defensible contribution that the accumulated evidence
already largely supports.
