# Promising research lines (FILLED — 2026-06-28 deep sweep)

> Populated by the `causal-recsys-bundle-construction` sweep. Full landscape + per-paper evidence:
> [outputs/causal-recsys-bundle-construction.md](outputs/causal-recsys-bundle-construction.md);
> provenance + citation verification: [outputs/causal-recsys-bundle-construction.provenance.md](outputs/causal-recsys-bundle-construction.provenance.md).
> The pre-seeded hypotheses L1–L4 below were **validated/refuted against the literature** (results inline).

## Scoring rubric (per line)

| criterion | meaning |
|---|---|
| **Novelty** | not already done in the causal-recsys / bundle literature. |
| **Fit-to-data** | leverages signal we KNOW exists (co-occ / exposure / popularity), not user-theme (refuted). |
| **Verifiability** | the causal term passes the inertness diagnostic (ablation / τ@0 / gradient-flow) **and a rank-inversion test** (a monotone reweight can change scores but not the ranking = inert). |
| **Feasibility** | tractable vs DDBC + co-occ/retrieve-rerank baselines on our 3 datasets. |
| **Bar-clearing** | plausibly beats co-occ/retrieve-rerank **full-catalog** (not the DDBC ρ=100 shortlist). |

## Headline findings (what the sweep settled)

- **GAP CONFIRMED (both):** no set/bundle-level causal **estimand for construction**, and no causal
  **generative/diffusion decoder** — verified across the field's own survey, both curated repos
  (~70 papers, ~90% pointwise, 0 set-construction), and web search.
- **NOVELTY DOWNGRADE (adversarial):** "deconfound co-occurrence against popularity" is **PRIOR ART** —
  **Cadence** (arXiv 2512.17733) already does item-item co-purchase deconfounding excluding popularity
  AND user attributes, full-catalog. Defensible novelty = **set-level + generative-decode-path +
  intervention-verified + within-set interaction**. *A Cadence-UACR reranker is now a mandatory baseline.*
- **MECHANISM EXISTS, NON-INERT:** PDA (popularity^γ at inference), MACR (TIE subtraction at decode),
  D3 (decode-score edit in generative recsys), UpliftRec (set-CATE on decode via DP), OPCB (set-policy
  gradient) — only their **combination** is unoccupied.
- **IDENTIFIABILITY WALL:** MealRec & Spotify-MPD have no exposure logs; Spotify-MPD has no user ids →
  uplift/IPS/IV/slate-OPE lines are unidentifiable as stated. Prefer observational contrasts with
  declared assumptions.

## Updated pre-seeded hypotheses (L1–L4: validate/refute)

- **L1 — Exposure/popularity deconfounding at the SET level:** *PARTIALLY REFUTED as novel.* The
  estimand is sound and fits the data, but Cadence (2512.17733) already occupies item-item co-occ
  deconfounding vs popularity. Survives **only** if lifted to the constructed-set / decode path with
  `do()`-flip verification (→ becomes L1′ = lines #1/#2).
- **L2 — Front-door / decode-path causal objective for generative construction:** *CONFIRMED novel &
  open.* No causal estimand lives on a generative decode trajectory (D3 is a heuristic; Causal Prompting
  is LLM-QA only; DMSG is non-causal). Highest-novelty, highest-risk (weak DDBC backbone) → line #2/#7.
- **L3 — Counterfactual/uplift reranking (treatment = adding item j):** *CONFIRMED novel*, but
  **unidentifiable as stated** (no exposure logs). Viable only re-stated as an observational
  complementarity contrast → line #3.
- **L4 — Observable-gated causal term:** *CONFIRMED & strongly supported.* Matches gate-g; A2G-DiffRec
  (global weight) is evidence that the non-gated alternative hurts clean accuracy. → line #1 (top).
- **L5 (from survey Future Directions):** 5.3 causality-aware GNN over the co-occ graph → line #10
  (heavy, likely inert at readout — bottom-ranked).

## Ranked shortlist

| rank | line | nov | fit | verif | feas | bar | total | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Observable-gated completion-deconfounding reranker** (fire only on cold/exposure-imbalanced slots, over retrieve-rerank) | 4 | 4 | 4 | 4 | 4 | **20** | best realistic bar shot; gate-g; falsifiable up front |
| 2 | **Set-PDA** (per-pick decode multiplier in a generative bundle decoder, observable-γ) | 4 | 5 | 4 | 4 | 3 | **20** | fills both gaps; risk = PPMI collapse + weak DDBC decode |
| 3 | **Completion-uplift reranker** (select by do(add j) lift) | 5 | 4 | 3 | 3 | 4 | **19** | identifiability without exposure logs is the crux |
| 4 | **Synergy-as-interaction decode term** (within-set interaction residual) | 5 | 4 | 3 | 2 | 3 | **17** | the Cadence differentiator; falsify residual≈0 first |
| 5 | Joint-exposure pair-level deconfounder (control-function/IV) | 3 | 5 | 3 | 3 | 3 | **17** | Cadence prior-art + no instrument → PPMI-relabel risk |
| 6 | Bundle-OPCB (off-policy set learning, factored main+residual) | 4 | 4 | 4 | 2 | 3 | **17** | needs logged set-propensities we lack |
| 7 | Front-door-over-partial-bundle (partial bundle as mediator) | 5 | 3 | 3 | 2 | 3 | **16** | positivity/variance; unidentifiable-as-stated |
| 8 | Substitute-set-confounder (factor-model co-occ, residualize at decode) | 3 | 4 | 3 | 3 | 2 | **15** | risk re-estimates popularity (= PPMI) |
| 9 | Slate-OPE-as-objective (pseudoinverse slate value drives decoder) | 4 | 3 | 3 | 2 | 2 | **14** | additive-per-slot kills synergy; needs propensities |
| 10 | Causal message-passing over co-occ graph (Future-Dir 5.3) | 5 | 3 | 2 | 2 | 2 | **14** | heavy GNN; edge term likely inert at readout |
| — | ~~Counterfactual-sequence augmentation~~ | 2 | 2 | 1 | 4 | 2 | **11** | dead — augmentation only, bypassable |

> Lines 5/6/7/9 are **unidentifiable-as-stated on our datasets** (no exposure logs / no user ids / no
> valid instrument). Kept for completeness; not shortlist candidates.

## Hand-off: 3 candidate method designs → brainstorming → writing-plans

1. **S1 — Observable-gated completion-deconfounding reranker** (safest bar-clearing; rides the winning
   retrieve-rerank backbone; estimand identifiable without propensities via matched leave-one-out).
2. **S2 — Set-PDA decode-path deconfounding** in a generative bundle decoder (highest novelty, fills
   both gaps; per-slot observable-γ to defeat PPMI collapse; bar gated on full-catalog, not shortlist).
3. **S3 — Within-set interaction (synergy) estimand** — the formal differentiator vs Cadence; runs an
   **up-front falsification** (residual magnitude after popularity deconfounding) before decode
   integration; rides S1 or S2.

**Mandatory new baselines for any of the above:** raw co-occ, retrieve-then-rerank (PPMI-SVD),
PPMI-normalized co-occ, **Cadence-UACR reranker**, DDBC (full-catalog, not ρ=100 shortlist).
**Mandatory diagnostics:** ablation Δ, τ@0 / set-correlation vs the ~99% sampler-noise null,
gradient-flow, and the **rank-inversion** test on the gated subpopulation.
