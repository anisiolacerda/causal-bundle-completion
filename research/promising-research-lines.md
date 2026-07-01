# Promising research lines (RE-RANKED — 2026-07-01 brainstorm)

> Re-ranked after the 0/2-bets pattern (FD-Set killed at design = dead end #3; CEC killed by M0.5 across
> 2 datasets = dead end #4). Reconciled with the 2026-06-28 sweep
> ([outputs/causal-recsys-bundle-construction.md](outputs/causal-recsys-bundle-construction.md)) and the
> four surveys in `refs/`. Full design + falsification gates:
> [docs/superpowers/specs/2026-07-01-causal-synergy-bundle-construction-design.md](../docs/superpowers/specs/2026-07-01-causal-synergy-bundle-construction-design.md).
> Kickoff prompt that drove this: [01-brainstorm-kickoff-causal-construction-lines.md](01-brainstorm-kickoff-causal-construction-lines.md).

## The meta-lesson (governs the ranking)

All four dead ends assumed the objective is **full-catalog, in-distribution completion accuracy**, for
which **co-occurrence is optimal**. The escape is to **change the estimand/objective/distribution/
deliverable** so co-occ is not the optimal baseline. Lines that do this are ranked highest. Decision this
round: escape via a **new ESTIMAND — within-set synergy** (top), with the **LLM lane ranked fully** as an
independent carried design.

## Dead ends — CLOSED (do not surface as promising)

| # | dead end | why closed |
|---|---|---|
| 1 | user-theme `do(seed)` deconfounding | **INERT** at decode (flip θ ~2% vs ~99% sampler-noise null; oracle θ ≈ 0). |
| 2 | popularity/exposure deconfounding of co-occ | **PRIOR ART** (Cadence 2512.17733) + PPMI-collapse + sparse-fail. Kills prior **S1/S2**. |
| 3 | total-effect front-door over partial-bundle mediator | **MIS-SPECIFIED** (dominant direct seed→item path). |
| 4 | cold-item / `do(uniform exposure)` completion | **LOW CEILING / WRONG SLICE** (M0.5; content edge warm-weighted, p=1e-17). |

Also closed / down-weighted: whole popularity-deconfounding family (prior **S1 gated deconfounding
reranker**, **S2 Set-PDA**); IPS/DR/uplift/IV/slate-OPE/OPCB (unidentifiable — no exposure logs,
Spotify-MPD has no user ids); counterfactual-sequence augmentation (CauseRec/CASR/CLBR — bypassable);
globally-forced decode guidance (A2G-DiffRec — global weight displaced clean accuracy 3×).

## Scoring rubric (kickoff §4; each line 1–5)

Escapes-wall · Novelty · Intervention-verifiable · Cheap-falsifiable · Identifiable-on-our-data · Feasibility.

## Re-ranked lines

| tier | line (lane) | esc | nov | ver | kill | id | feas | total | escapes-wall / cheap kill |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Within-set SYNERGY** (L-A/L-B) — non-additive `τ_syn(j\|S)=E[Y(S∪j)]−E[Y(S)]−E[Y(j)]+E[Y(∅)]` | 5 | 5 | 4 | 5 | 5 | 3 | **27** | new estimand co-occ can't represent / synergy-residual magnitude + rank-inversion vs PPMI |
| 1 | **Anonymized-LLM complementarity** (L-F2) — "Corr2Cause mode", names denied | 4 | 4 | 5 | 5 | 5 | 4 | **27** | different signal source + test-dist / LLM-vs-PPMI r>0.7 + name-strip collapse |
| 1.5 | **Partial-ID / bounds** (L-I) — Rosenbaum-Γ / Kallus-Zhou on `τ_syn` | 3 | 4 | 5 | 4 | 4 | 3 | **23** | reframes the claim / Γ-sweep bounds blow up to include 0 = kill. **Rigor layer for L-A, not standalone.** |
| 2 | **Item-item DAG discovery** (L-E) — DAG is the deliverable, co-occ an input | 4 | 4 | 3 | 4 | 4 | 3 | **22** | co-occ not a competitor / edge-vs-PPMI Spearman>0.7 or >80% Jaccard or popularity-direction AUC |
| 2 | **Distribution / OOD** (L-D) — temporal / cross-domain / genre-shift | 5 | 3 | 3 | 4 | 3 | 2 | **20** | wall is in-distribution only / co-occ-near-random-on-split check. Temporal feasible; cross-domain has NO ready shared-user benchmark. |
| 3 | **Objective / reward** (L-C) — bundle-level `E[Y(B)]` off-policy/DTR | 5 | 4 | 3 | 3 | 1 | 2 | **18** | co-occ predicts popularity not reward / proxy-reward-vs-co-occ Spearman<0.7. **HARD-blocked: no reward/impression logs; needs causal simulator.** |
| 3 | **Intent-mediated generation** (L-G) | 4 | 2 | 2 | 3 | 3 | 2 | **16** | different objective / mediation d-separation test on intent-annotated Amazon. Collapse-risk into dead-end #1; heavy prior art (AICL/Text2Bundle/MIDGN/**BunCa**). |
| 3 | **Fairness / exposure objective** (L-H) | 5 | 2 | 3 | 3 | 4 | 3 | **20*** | wall gone by construction / rerank-vs-generation ablation. *Departs from accuracy goal; A2G-DiffRec/DifFaiRec turf — only if a fairness contribution is accepted. |

`L-B interference` = the formal causal scaffolding for L-A (spillover / SUTVA-violation), not a separate
line. `Gao §5.3 causal-GNN` = an architecture layer that pairs with L-E, not an estimand lane.

### Novelty landmines (mandatory to differentiate / baseline)
Cadence-UACR (2512.17733, item-item pop-deconfounding — mandatory baseline); PDA/MACR/DICE/IPS-DR
(pointwise deconfounding); UpliftRec/OPCB/Slate-PI-Swaminathan (set-level but additive value / need
propensities); NCoRE (2103.11175 — interaction *estimation*, not construction); COR/CausPref/CausalDiffRec
(representation-path OOD); DT-CDBR (only cross-domain BR); BunCa (closest causal-BR by name, item-level
discriminative); Kiciman/Cai/Jiralerspong/Ban/LACR (LLM causal discovery — all FEED names, opposite of
the anonymized move); DMSG/DiFashion (generative-BR substrate, non-causal); A2G-DiffRec (decode-path but
globally-weighted = the anti-pattern).

## Hand-off: 3 carried designs → writing-plans (each with its up-front CPU-minutes gate)

Every design's **first implementation slice IS its falsification gate** (kill-or-proceed before any GPU),
mirroring the M0.5 pattern that killed the last two bets cheaply.

1. **DESIGN 1 (TOP → writing-plans first) — within-set synergy completion term + partial-ID bounds.**
   Estimand `τ_syn(j|S)` (pairwise/low-order; higher-order deferred for positivity). Placement:
   per-pick logit correction / non-additive seed-set→completion reranker (reuse predecessor
   `recovery/reranker.py`), **NOT** the DDBC `cond_dim=128` FiLM slot. Regime: **dense |gt|≥2**
   (MealRec+ H/L, POG_dense) — a small term cannot be load-bearing on |gt|=1 sparse (sampler variance
   ~99%). Metric: full-catalog completion vs co-occ / PPMI / retrieve-rerank / Cadence-UACR / content
   SetCompleter / CLHE / DDBC-full; ≥5 seeds, paired Wilcoxon. L-I bounds prove the gain is causal.
   **GATE (CPU-min):** fit additive PPMI main effects → estimate `τ_syn` on held-out slice
   (reuse `cec_m05_exposure_premise.py` + `src/cec`) → **KILL if** `|τ_syn|≈0` vs permutation null, OR
   synergy ranking rank-equivalent to co-occ (Kendall/Spearman ≳0.9 = inert), OR the "large-τ-but-low-co-occ"
   masked-complement slice is empty; plus additive-null placebo (product-of-marginals ⇒ τ≈0) and a
   positivity/overlap pre-check. **GO** only on a non-trivial non-rank-equivalent residual.

2. **DESIGN 2 — anonymized "Corr2Cause-mode" LLM item-item complementarity (L-F2).** Deny item names;
   give only anonymized IDs + structural/behavioral context, so any lift over co-occ is provably not
   memorized (name-stripped LLMs go to chance — Corr2Cause/parrots). Reranks co-occ candidates / feeds
   Design 3. Sidesteps the identifiability wall (needs only names + relation queries).
   **GATE (CPU-min + few-hundred LLM calls):** (1) LLM-score vs PPMI `r>~0.7` ⇒ KILL; (2) named-vs-anonymized
   collapse ⇒ KILL if anonymized→chance and named≈co-occ; (3) contamination probe (verbatim-recall +
   head/tail gradient + pre/post-cutoff split); (4) necessity-label self-consistency if L-F1 labels used.

3. **DESIGN 3 — item-item complementarity DAG discovery (L-E), fed by Design 2.** DAG is the primary
   deliverable (co-occ = input feature); CI-test / PC-FCI over incidence, seeded/pruned by Design-2
   scores; optional causal-GNN over the DAG (Gao §5.3). Metric: OOD/temporal lift + explainability +
   plausibility (NOT in-distribution completion alone). **GATE (CPU-min):** KILL if
   `Spearman(edge, PPMI)>~0.7`, OR >80% Jaccard with top-k co-occ neighbors, OR edge direction predicted
   by popularity ratio `P(j|i)/P(i|j)` alone.

## Mandatory baselines & diagnostics (any empirical claim)

- **Baselines:** raw co-occ, retrieve-then-rerank (PPMI-SVD), PPMI-normalized co-occ, **Cadence-UACR**
  (adapt), content SetCompleter, CLHE (full-catalog anchor), DDBC (full-catalog, NOT ρ=100 shortlist).
- **Diagnostics:** ablation Δ, τ@0 / set-correlation vs the ~99% sampler-noise null, gradient-flow, the
  **rank-inversion** test, and a **`do()`-flip** wherever a causal term is claimed.
- **Honesty guards:** full-catalog eval, ≥5 seeds, paired Wilcoxon, pre-registered thresholds, report
  negatives plainly.
