# Design — Causal within-set SYNERGY for bundle construction (+ anonymized-LLM complementarity, + item-item DAG discovery)

- **Date:** 2026-07-01
- **Status:** Design approved (brainstorming) → next: `writing-plans` for Design 1
- **Supersedes ranking in:** [research/promising-research-lines.md](../../../research/promising-research-lines.md) (2026-06-28 sweep)
- **Prior killed bets (same spec dir):** [FD-Set](2026-06-29-fd-set-causal-bundle-design.md) (dead end #3), [CEC](2026-06-30-cec-counterfactual-exposure-completion-design.md) (dead end #4)
- **Predecessor code (UNPUBLISHED — reuse, never cite):** `/Users/anisiomlacerda/code/Bundle_construction/src/causalbr/`

---

## 1. Problem, bar, and the meta-lesson

Task: **bundle construction / completion** — given a seed / partial bundle `S`, produce the missing items.

Inherited hard bar (non-negotiable):
1. **Beat trivial co-occurrence AND retrieve-then-rerank on FULL-CATALOG completion** (no shortlist, no
   negative sampling), **OR** change the estimand/objective/distribution/deliverable so "beat co-occ
   full-catalog" is no longer the yardstick; **AND**
2. show a **real, intervention-verified causal gain** — the causal term must pass an inertness /
   `do()`-flip battery (ablation Δ; τ@0 / set-correlation vs a ~99% sampler-noise null; gradient-flow; a
   **rank-inversion** test — a monotone reweight changes scores but not the ranking = inert). Causal
   terms are **decode-path or observable-gated, never globally forced**.
3. Honesty guards: full-catalog eval, ≥5 seeds, paired Wilcoxon, pre-registered thresholds, negatives
   reported plainly.

**The wall (inherited, empirical):** plain co-occurrence is extremely hard to beat on full-catalog,
in-distribution completion. co-occ beats DDBC full-catalog on its own benchmark (recall@100 0.096 vs
0.065, paired Wilcoxon p=1.5e-42); retrieve-then-rerank (PPMI-SVD CF) beats co-occ **only on dense**
catalogs (MealRec+ hit@1 0.164 vs 0.132/0.149, p=0.031) and **does not transfer** to the 254k-sparse
Spotify catalog; content ≠ complementarity.

**Meta-lesson:** all four prior dead ends assumed the objective is full-catalog in-distribution
completion accuracy, for which co-occ is optimal. The way out is to change the estimand so co-occ is not
the optimal baseline — **the non-additive causal object co-occ cannot represent.**

## 2. Dead ends — CLOSED (do not re-propose)

| # | dead end | why closed |
|---|---|---|
| 1 | user-theme `do(seed)` deconfounding (latent bundle theme mediates seed→completion) | **INERT** at decode: flip θ moved output ~2% vs a ~99% sampler-noise null; oracle θ added ~0 (τ_cond−τ_free ≈ +0.001..+0.003). |
| 2 | popularity/exposure deconfounding of co-occurrence | **PRIOR ART** (Cadence, arXiv 2512.17733, item-item co-purchase deconfounding vs popularity + user attrs, full-catalog) + collapses to PPMI + fails on sparse. Kills the whole prior S1/S2 family. |
| 3 | total-effect **front-door** over a partial-bundle mediator | **MIS-SPECIFIED**: dominant direct seed→item compatibility path (DDBC win grows with bundle length; item cosine carries signal) ⇒ full mediation false. |
| 4 | cold-item / `do(uniform exposure)` counterfactual completion | **LOW CEILING / WRONG SLICE** (M0.5): co-occ blind on cold items but content's edge is warm-weighted on sparse (Δ grows with exposure, p=1e-17), and cold-item content win is tiny + non-significant. |

Also closed / down-weighted: the entire popularity/exposure-deconfounding family (prior **S1 gated
deconfounding reranker**, **S2 Set-PDA** — now dead by #2 + Cadence + PPMI-collapse + M0.5); IPS/DR/uplift
/IV/slate-OPE/OPCB (unidentifiable — no exposure logs, Spotify-MPD has no user ids);
counterfactual-sequence augmentation (CauseRec/CASR/CLBR — augmentation-only, bypassable); globally-forced
decode guidance (A2G-DiffRec anti-pattern — global weight displaced clean accuracy 3× in the sibling
project).

## 3. The decision (this design)

Two framing choices were made in brainstorming:
- **Escape dimension = new ESTIMAND (within-set synergy, L-A/L-B).** Top pick.
- **Synergy framing = completion-method + bounds:** synergy as a decode-path / reranker term on a
  regime where it can be load-bearing, with **partial identification (L-I)** as the rigor layer proving
  the gain is causal, not a co-occ artifact.
- **LLM lane (L-F) ranked fully** as an independent carried design.
- **Carry three designs:** Design 1 synergy (→ `writing-plans` first), Design 2 anonymized-LLM
  complementarity, Design 3 item-item DAG discovery (fed by Design 2).

**Critical regime constraint (drives Design 1):** on **|gt|=1 sparse long-tail** completion
(Spotify full-catalog), NO conditioning term can be load-bearing — MDLM sampler variance flips ~99% of
items per re-decode (τ0_null≈0.99) while any conditioning flip moves ~1.5–5%, and decode sits ~4.4×
random floor. To make a synergy term move the metric you **must change the signal-to-noise regime:
|gt|≥2, dense catalog, smaller candidate pool.** The retrieve-rerank win was dense-only for the same
reason. **⇒ Design 1's primary home is DENSE, multi-item completion (MealRec+ H/L, POG_dense).** The
positive corollary: the complementarity signal already **lives in the raw seed set** (predecessor
seed-pair oracle = 7.5× the backbone decode) — it must be extracted *non-additively*, not mean-pooled
into a theme (which destroyed it before).

---

## 4. DESIGN 1 — TOP: within-set synergy completion term + partial-ID bounds (L-A + L-I)

### 4.1 Estimand
Pairwise (and low-order) **causal synergy / non-additive interaction**:

```
τ_syn(j | S) = E[Y(S ∪ {j})] − E[Y(S)] − E[Y({j})] + E[Y(∅)]
```

the super-additive effect of adding item `j` to partial set `S`, beyond the additive main effects. `Y` =
held-out completion outcome (does `j` belong to the true completion of the bundle whose observed part is
`S`). The pairwise base case is `τ_ij = E[Y(both) − Y(i only) − Y(j only) + Y(neither)]` — a factorial
(2×2) interaction contrast, **not** a covariate-heterogeneity CATE and **not** a slate/set VALUE.

**Higher-order `τ_ijk…` are deferred** — the exponential-treatment **positivity** problem
(Complex-Treatments survey, arXiv 2407.14022 §8.1.2): combinatorial treatment cells have vanishing
overlap. Start pairwise/low-order; escalate only where positivity holds.

### 4.2 Why it escapes the wall and is novel
- **Non-additive**, so it is exactly what co-occ (a count), Cadence-UACR (additive item-item), and
  slate-PI / SlateQ (additive-by-assumption reward) all structurally cannot represent.
- Named **OPEN** in Yao TKDD 2021 §7 (Egami & Imai 2019, ref [45], never operationalized in the survey
  body) and in the Complex-Treatments survey §8.1.2; Gao TOIS 2024 reaches set-level only via slate
  off-policy *value* estimation (Swaminathan [95], Li [53], Kiyohara Cascade-DR [46]) — never the
  non-additive within-set interaction.
- **NCoRE** (arXiv 2103.11175) is the only structural precedent — combination-of-treatments interaction
  *estimation*, **not** recsys set construction, not decode-path, not full-catalog. Instantiating the
  synergy estimand *for recsys set construction, full-catalog, decode-path, with a `do()`-flip* is the
  contribution.
- **Identifiable on our data** — needs only the item×bundle incidence matrix we already have (no
  exposure logs, no user ids, no instrument).

### 4.3 Identification & the partial-ID (L-I) rigor layer
Set membership is confounded by unobserved intent, so `τ_syn` is not point-identified in general. Two
moves:
1. **Observational contrast under a declared assumption** (no-unobserved-confounding-given-partial-set)
   for the point estimate — via a DR / plug-in interaction estimator (residualize additive main effects,
   estimate the interaction residual).
2. **Partial identification (L-I):** report **Rosenbaum-Γ / Kallus-Zhou ambiguity-set bounds** on
   `τ_syn` under unmeasured confounding. The causal claim becomes: *the complementarity gain (and its
   ranking of completions) survives bounded confounding*, distinguishing a real causal synergy from a
   co-occ artifact. This is absent from Yao's toolbox ("Manski / sensitivity-as-method" = zero hits);
   differentiate from Kallus-Zhou (they bound a *policy value*, we bound the *set-completion interaction
   contrast* `τ_syn`).

### 4.4 Placement (non-inert by construction)
Decode-path / reranker scorer on dense |gt|≥2 completion. The synergy term is a **per-pick logit
correction / interaction score** added on top of the additive base (co-occ / PPMI + retrieve-rerank), or
a **learned seed-set→completion reranker** whose score IS the non-additive interaction (cross-terms /
factorization-machine-style, or a DR-learner interaction head) — **reusing predecessor
`recovery/reranker.py`** (the validated Stage-2 MealRec winner). Explicitly **NOT** the DDBC
`cond_dim=128` FiLM slot (information-free timestep slot ⇒ inert global conditioning). This satisfies the
"decode-path or observable-gated, never globally forced" rule.

### 4.5 Datasets / regime
Primary: **MealRec+ H/L** (dense, |gt|≥2, food; has `exposure.npy`) and **POG_dense** (CLHE
`xhLiu/BundleConstruction`). Sparse Spotify kept only as a **stress/negative slice** (expected to show
the term cannot be load-bearing there — a characterization, not the headline).

### 4.6 Objective / metric & mandatory baselines
Full-catalog completion (recall@k, hit@1, MRR), ≥5 seeds, paired Wilcoxon, pre-registered thresholds.
Must beat: **raw co-occ, PPMI-normalized co-occ, retrieve-then-rerank (PPMI-SVD), Cadence-UACR (adapt),
content SetCompleter, CLHE (full-catalog anchor), DDBC (full-catalog, NOT ρ=100 shortlist).**

### 4.7 Diagnostics (mandatory, every causal claim)
ablation Δ; τ@0 / set-correlation vs the ~99% sampler-noise null; gradient-flow at convergence;
**rank-inversion**; **`do()`-flip** = `do(add j)` vs observational `P(j|S)`. Reuse predecessor
`eval/{inertness,oracle_ceiling,significance}.py` + this repo's `src/cec/`.

### 4.8 ⛔ UP-FRONT FALSIFICATION GATE (slice-1, CPU-minutes, run BEFORE any training)
The **synergy-residual magnitude + rank-inversion** test — mirrors M0.5 exactly, reuses
`scripts/cec_m05_exposure_premise.py` (incidence mat-vec) + `src/cec/{exposure,stratify}.py` +
predecessor `eval/significance.py`:

1. Fit the **additive baseline**: pairwise main effects from co-occ / PPMI on train.
2. Estimate `τ_syn(j|S)` on a **held-out slice** via a DR / plug-in interaction estimator over the
   incidence matrix (no GPU).
3. **Pre-registered KILL if ANY:**
   - `|τ_syn|` distribution is concentrated at 0 vs a **permutation null** (residual ≈ 0); OR
   - the synergy ranking of `(j|S)` is **rank-equivalent** to the co-occ / PPMI ranking
     (Kendall-τ / Spearman ≳ 0.9 = monotone reweight, no rank change = inert in effect); OR
   - the **"large-|τ_syn| but LOW co-occ"** slice (the masked-complements that would be the win) is
     empty / negligible.
4. **Placebo / additivity null:** synthesize sets from a product-of-marginals (items conditionally
   independent by design); the estimator must return `τ_syn ≈ 0`. Systematic non-zero ⇒ estimator picks
   up artifacts ⇒ recalibrate or kill.
5. **Positivity pre-check:** verify overlap (enough `S` with and without `j` across target pairs). If
   positivity fails for the target pairs ⇒ restrict order, or go **bounds-only** (L-I) as the deliverable.
6. **GO** only on a non-trivial slice with large synergy residual that is **not** rank-equivalent to
   co-occ. Pre-register the exact thresholds before running.

### 4.9 If the gate passes — the build
Train the reranker / per-pick logit correction (reuse `recovery/reranker.py`), run the full baseline
suite + diagnostic battery full-catalog, and attach the L-I Γ-sweep bounds. Ship only if it clears the
bar on **≥5 seeds** with the causal term passing the inertness + rank-inversion + `do()`-flip battery.

---

## 5. DESIGN 2 — anonymized "Corr2Cause-mode" LLM item-item complementarity (L-F2)

### 5.1 Target
Directed item-item **complementarity / functional need**: does owning `i` create a functional need for
`j` (`do(own i) → need j`), beyond co-occ frequency and popularity? An LLM produces a pairwise
causal-complement score that reranks co-occ candidates (or feeds Design 3's DAG).

### 5.2 The reframe (novelty and falsification, unified)
Run the LLM in **Corr2Cause mode: deny item names**, give only **anonymized IDs + structural/behavioral
context** (co-occurrence-controlled neighborhoods, category/interaction patterns without nameable
identity). Evidence base (Ma NAACL 2025 survey): name-based PCD *succeeds* (Kiciman 2023) precisely
because LLMs recite memorized name-knowledge; symbol-only **Corr2Cause** (Jin 2023) sends them to
**chance**; Zecevic 2023 "causal parrots" is the named critique. Therefore, with names stripped, **any
lift over co-occ cannot be memorized name-knowledge ⇒ falsification-by-construction.** Every named
landmine (Kiciman flagship, Cai fine-tuned Mistral, Jiralerspong BFS-graph, Ban LLM+solver, LACR/RAG)
*feeds* names + retrieval — the opposite move.

**Fit signal:** L-F sidesteps the identifiability wall entirely — it needs only names + relation
queries, which even Spotify-MPD (no user ids, no impressions) can supply.

### 5.3 ⛔ UP-FRONT FALSIFICATION GATE (CPU-minutes + a few hundred LLM calls, run FIRST)
The **parrot / contamination triple**, all pre-registered:
1. **PPMI-correlation kill:** correlate LLM causal-complement scores with co-occ PPMI on a held-out pair
   sample. `r > ~0.7` ⇒ reciting co-occ ⇒ **KILL**.
2. **Name-strip collapse:** score **named vs anonymized**; report the delta. If anonymized accuracy
   collapses toward chance AND named ≈ co-occ ⇒ signal was memorized names ⇒ **KILL**. (Anonymized >
   co-occ = the paper.)
3. **Contamination probe:** name-completion attack (does the LLM recall held-out completions verbatim?)
   + head/tail accuracy gradient + **pre- vs post-knowledge-cutoff** temporal split. Wins only on
   famous/head/pre-cutoff pairs ⇒ leakage ⇒ discount / kill.
4. **Necessity-label self-consistency** (only if L-F1 necessity labels are used): paraphrase / reorder
   prompts; flip rate above a pre-registered threshold ⇒ labels unreliable ⇒ do not treat as
   intervention-verified ground truth (use as a soft stratifier at most; Ma Table 4 shows Rung-3 labels
   are noisy/prompt-sensitive).

---

## 6. DESIGN 3 — item-item complementarity DAG discovery (L-E), fed by Design 2

### 6.1 Deliverable
An **interpretable directed item-item complementarity graph** as the PRIMARY output (edges =
`do()`-verified functional-need, with direction / necessity); **co-occ is an INPUT feature, not a
competitor.** Built by constraint-based CI tests / PC-FCI over the incidence matrix, **seeded / pruned by
Design 2's anonymized-LLM pairwise scores**; optionally a causal GNN over the discovered DAG (Gao §5.3 —
"intertwine message-passing with causal reasoning", flagged open) evaluated on downstream OOD / temporal
completion. Gao §5.1 explicitly flags recsys causal discovery as open and supplies no item-item method.

### 6.2 Metric
DAG quality: downstream OOD / temporal completion lift + explainability + human / LLM plausibility.
**Not** in-distribution completion accuracy alone (that path returns to the wall) — the DAG must BE the
contribution.

### 6.3 ⛔ UP-FRONT FALSIFICATION GATE (CPU-minutes, run FIRST)
The **discovery-vs-co-occ redundancy** test:
1. Recover edges via pairwise CI test / PC-FCI on the incidence matrix.
2. **KILL if ANY:** `Spearman(edge-score, PPMI) > ~0.7`; OR discovered edge set has `>80%` Jaccard with
   top-k co-occ neighbors (no structure beyond co-occ); OR edge **direction** is predicted by the
   popularity ratio `P(j|i)/P(i|j)` alone (AUC ≈ popularity-direction baseline ⇒ direction is a
   popularity artifact).
3. **GO** only if the DAG has non-redundant edges whose direction is not explained by popularity.

---

## 7. Re-ranked lines (rubric §4 of the kickoff) — summary

Scores 1–5 on Escapes-wall / Novelty / Intervention-verifiable / Cheap-falsifiable / Identifiable /
Feasibility. Full table maintained in `research/promising-research-lines.md`.

- **TIER 1 (carry):** L-A synergy `5/5/4/5/5/3` (TOP); L-F2 anonymized-LLM complementarity
  `4/4/5/5/5/4`.
- **TIER 1.5 (rigor layer, not standalone):** L-I partial-ID / bounds — pair with L-A.
- **TIER 2 (pivots if data secured):** L-E discovery (carry as Design 3); L-D distribution/OOD (temporal
  feasible; cross-domain has no ready shared-user benchmark).
- **TIER 3 (down-weighted):** L-C reward/off-policy (hard-blocked — no reward logs); L-G intent (collapse
  risk into dead-end #1 + heavy prior art AICL/Text2Bundle/MIDGN/BunCa); L-H fairness (departs from
  accuracy goal, A2G-DiffRec turf); L-B interference = L-A's scaffolding, not separate.
- **DEAD (closed):** the four dead ends of §2 + popularity-deconfounding family (prior S1/S2) +
  front-door-over-partial-bundle + cold-item + counterfactual-seq augmentation + globally-forced decode
  guidance.

## 8. Shared: mandatory baselines, diagnostics, honesty guards, reusable assets

- **Baselines (any empirical claim):** raw co-occ, retrieve-then-rerank (PPMI-SVD), PPMI-normalized
  co-occ, Cadence-UACR (adapt), content SetCompleter, CLHE (full-catalog anchor), DDBC (full-catalog,
  NOT ρ=100 shortlist).
- **Diagnostics:** ablation Δ, τ@0 / set-correlation vs the ~99% sampler-noise null, gradient-flow,
  rank-inversion, `do()`-flip wherever a causal term is claimed.
- **Honesty guards:** full-catalog eval, ≥5 seeds, paired Wilcoxon, pre-registered thresholds, negatives
  reported plainly.
- **Reusable assets (port / reuse — do NOT rebuild):**
  - This repo: `src/cec/{exposure,stratify}.py` (exposure binning + per-bin paired stratified stats,
    tested); `scripts/cec_m05_exposure_premise.py` (memory-safe co-occ via item×bundle incidence
    mat-vec); `scripts/cec_m05_mealrec.py` (retrains SetCompleter, dense).
  - Predecessor `src/causalbr/` (UNPUBLISHED — reuse, never cite): full-catalog metrics
    (`recovery/{ranking_metrics,set_metrics}.py`), significance (`eval/significance.py`), inertness /
    `do(T)` / oracle diagnostics (`eval/{inertness,theme_contrast,oracle_ceiling,loadbearing_driver}.py`),
    ObservableGate (`causal/gate.py`), SetCompleter + reranker (`recovery/{set_completer,reranker,
    fusion}.py`), DDBC decode (`recovery/ddbc_decode.py`), loaders (`datasets/*`), exposure tooling
    (`eval/{popularity*,lift}.py`, `diagnostics/exposure_decorrelation.py`).
  - Compute: remote GPU box for heavier runs (confirm it is up before relying on it; every slice-1
    falsification gate here runs on CPU in minutes and does NOT need it).

## 9. Scope note for `writing-plans` (Design 1)

Scope the plan so the **first implementation slice IS the §4.8 falsification gate** (kill-or-proceed on
CPU, before any GPU / reranker training), mirroring the M0.5 pattern that cheaply killed the last two
bets. Only after a pre-registered GO does the plan proceed to the reranker build, the full baseline
suite, the diagnostic battery, and the L-I bounds.

## 10. Open risks (carry into the plan)
- Synergy residual may be ≈0 / rank-equivalent to co-occ ⇒ that is exactly what §4.8 kills cheaply.
- Positivity for higher-order synergy ⇒ stay pairwise/low-order or go bounds-only.
- Dense-only headroom ⇒ Design 1 does not claim the sparse Spotify full-catalog; Spotify is a
  characterized negative slice.
- L-F contamination on nameable catalogs ⇒ the anonymized "Corr2Cause mode" is the mitigation AND the
  falsification.
- L-E redundancy with co-occ ⇒ §6.3 kills it cheaply.
