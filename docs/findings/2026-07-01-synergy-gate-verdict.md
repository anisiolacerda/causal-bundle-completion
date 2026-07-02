# Synergy falsification gate — verdict: KILL (both dense datasets)

Design 1 slice-1 (Egami–Imai within-set synergy `τ_syn(j|{i,k}) = P(j|{i,k}) − P(j|i) − P(j|k) + P(j)`),
CPU-only, full-catalog, no training. Pre-registered thresholds frozen in
`scripts/synergy_m1_gate.py` (plan Task 6, commit a80cba6). Runs completed 2026-07-02.

## Results

| dataset | n_items | n_inst | tau_abs (obs / null) | tau_perm_p | spearman / jac20 | hit@1 cooc→joint | hit@20 cooc→joint (wilcoxon p) | masked_frac | placebo_ok | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| mealrec_l | 10,436 | 1,181 | 0.00145 / 0.00334 | 1.0 | 0.147 / 0.064 | 0.151→0.758 | 0.682→0.873 (2.9e-45) | 0.0457 | false | **KILL** |
| mealrec_h | 4,805 | 4,677 | 0.00526 / 0.00788 | 1.0 | 0.141 / 0.062 | 0.132→0.631 | 0.635→0.999 (0.0) | 0.00043 | false | **KILL** |

Failed gates (both datasets): `residual_at_null`, `placebo_miscalibrated`.

## Decision: KILL — the non-additive synergy term is INERT on dense data

On BOTH dense regimes the observed synergy-residual magnitude is **below** the additivity-null
floor (`tau_abs_mean` < `tau_abs_mean_null`; `tau_perm_p = 1.0` = the observed statistic never exceeds
any of the 19 product-of-marginals null draws). The non-additive interaction carries no signal above
the noise floor. Pre-registered rule ⇒ KILL. **Do NOT proceed to the reranker build.**

### The crucial diagnostic: the completion "win" is ADDITIVE, not synergy
The joint conditional `P(j|{i,k})` massively beats raw co-occ counts (hit@1 **0.63–0.76 vs 0.13–0.15**;
hit@20 up to **0.999 vs 0.635**). This looks like a huge win — but `joint = additive + τ`, and `τ ≈ 0`
(indeed below the null). So the entire gain is the **additive conditional / PPMI-normalization** effect
(dividing co-occ by `n({i,k})`), NOT the non-additive synergy. This re-confirms the project's own prior
finding (C3: conditional/PPMI-SVD retrieve-rerank beats raw co-occ **on dense** catalogs) and Cadence's
additive item-item deconfounding — **prior art, not a new contribution.** The synergy-specific term,
which was the whole novelty claim, adds nothing on top. The `rank` agreement (Spearman ≈ 0.14, top-20
Jaccard ≈ 0.06) confirms the joint ranking is not a monotone reweight of co-occ — but that reordering is
driven by the additive conditional, not by `τ`.

### Honest caveats (do not change the verdict)
- **Laplace smoothing** (`+1/+n_items`) shrinks all conditional probabilities, compressing `τ`. But it
  is NOT crushing signal broadly: the joint conditional is highly discriminative (hit@1 up to 0.76), and
  the same smoothing applies to the null, so the observed-vs-null comparison is fair. It is the
  non-additive part specifically that is ≈0.
- **`placebo_miscalibrated`** (null `τ` > observed `τ`) is partly a null-noise artifact: product-of-
  marginals bundles have sparser conditioning cells → noisier `τ` estimates. This weakens that specific
  label, but the PRIMARY kill (`residual_at_null`, observed not exceeding the null) holds regardless of
  null-noise: there is simply no synergy signal above the floor.
- **`ConstantInputWarning`** (scipy, `rankinv.py:34`) fires on instances whose candidate scores are
  constant → nan correlation, which is dropped by the finite-guard. Verdict unaffected; a guard against
  constant input is a hygiene TODO (does not change any reported number).

## Where this leaves the project

- **0/3 designed causal bets now cheaply killed** before GPU: FD-Set (design), CEC/cold-item (M0.5),
  and now within-set synergy (this gate) — each falsified in CPU-minutes. The pre-registered cheap-gate
  methodology continues to pay off.
- **Reconciliation:** the surveys' central risk for L-A fired exactly as flagged ("synergy may collapse
  to co-occ / be recoverable from additive co-purchase"). On these datasets it does.
- **Next:** pivot to **Design 2 — anonymized "Corr2Cause-mode" LLM item-item complementarity** (its own
  cheap gate: LLM-score vs PPMI `r>0.7` kill + name-strip collapse). See
  `research/promising-research-lines.md` and the design spec
  `docs/superpowers/specs/2026-07-01-causal-synergy-bundle-construction-design.md`.
- **Open question worth a follow-up (not a rescue of synergy):** whether a *different* synergy estimator
  (lighter smoothing; higher-order `τ_ijk` where positivity allows; or a decode-path placement on a
  larger-bundle regime) could revive a non-additive signal. The frozen pre-registered gate says the
  pairwise, Laplace-smoothed, dense-|obs|=2 instantiation is inert — changing the estimator to chase a
  positive would be moving the goalposts and is out of scope for this verdict.
