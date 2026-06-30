# Fresh-session prompt — brainstorm & rank causal research lines for bundle construction

> **How to use:** paste the body of this file as the opening prompt of a NEW session. It is
> self-contained: it carries the goal, every hard-won constraint, the four dead ends (do not
> re-propose them), the survey-grounded fresh lanes, the selection rubric, and the deliverable.
> Source material lives in this repo (`refs/*.pdf`, `docs/findings/*`, `research/*`) and the
> predecessor code at `/Users/anisiomlacerda/code/Bundle_construction` (UNPUBLISHED — use it to
> reuse code and avoid overlap; never cite it as prior art).

---

## 0. Your task

Run the **brainstorming skill** to produce an **updated, ranked set of promising research lines** for a
**new causal-inference method for bundle CONSTRUCTION / COMPLETION** (given a seed / partial bundle of
items, produce the missing items to complete the set). Reconcile with the prior ranking in
`research/promising-research-lines.md` and the negative results below; then hand 2–3 candidate designs
to `writing-plans`. **Each candidate must carry a CHEAP, up-front falsification** (a CPU-minutes test
that can kill it before any GPU/training spend) — that is this project's signature and the reason it has
cheaply killed two bets already.

Deliverable: overwrite `research/promising-research-lines.md` with the new ranked lines + the 2–3 carried
designs (each with its falsification gate), then proceed to `writing-plans` for the top pick.

---

## 1. The goal and the BAR (inherited, non-negotiable)

A method that:
- **Beats trivial co-occurrence AND retrieve-then-rerank on FULL-CATALOG completion** (no shortlist /
  no negative sampling), **OR** changes the objective so that "beat co-occ full-catalog" is no longer
  the right yardstick (see §3 — this is the recommended route), AND
- shows a **real, intervention-verified causal gain** — the causal term must pass an **inertness /
  `do()`-flip diagnostic** (ablation Δ; τ@0 / set-correlation vs a ~99% sampler-noise null;
  gradient-flow; a **rank-inversion** test — a monotone reweight can change scores but not the ranking =
  inert in effect). Causal terms must be **decode-path or observable-gated**, never globally forced.
- Honesty guards: full-catalog eval, ≥5 seeds, paired Wilcoxon, pre-registered thresholds, report
  negatives plainly.

---

## 2. The wall and the FOUR dead ends (DO NOT re-propose these)

**The wall:** plain **co-occurrence is extremely hard to beat on full-catalog, in-distribution
completion accuracy.** Established facts (predecessor results, treat as inherited givens):
- co-occ wins full-catalog and beats the whole learned family on DDBC's own benchmarks (DDBC's headline
  is a ρ=100 **shortlist**; co-occ wins once you score full-catalog).
- retrieve-then-rerank (PPMI-SVD CF over co-occ candidates) beats co-occ **only on dense** catalogs
  (MealRec+); it does **not** transfer to the 254k-sparse Spotify catalog.
- content ≠ complementarity: a content SetCompleter beats co-occ full-catalog **only on Spotify**
  (768-d CLHE); the ID-only ablation loses.

**The four causal angles already EMPIRICALLY KILLED — do not surface as promising:**

| # | dead end | why it died |
|---|---|---|
| 1 | user-theme `do(seed)` deconfounding (a latent "bundle theme" mediates seed→completion) | **INERT** at decode (flip θ moved output ~2% vs a ~99% sampler-noise null; oracle θ added ~0) |
| 2 | popularity/exposure-deconfounding of co-occurrence | **PRIOR ART** (Cadence, arXiv 2512.17733, deconfounds item-item co-purchase vs popularity AND user attrs, full-catalog) + collapses to **PPMI** + fails on sparse (C3) |
| 3 | total-effect **front-door** over a partial-bundle mediator | **MIS-SPECIFIED**: a dominant **direct seed→item compatibility path** exists (DDBC's win grows with bundle length; item-embedding cosine carries the signal), so full mediation is false |
| 4 | cold-item / `do(uniform exposure)` counterfactual completion | **LOW CEILING / WRONG SLICE** (the M0.5 falsification): co-occ is blind on cold items, but content's edge is *warm-weighted* on sparse (Δ grows with exposure, p=1e-17) and co-occ *dominates* on dense; the cold-item content win is tiny + non-significant. Verdict: `docs/findings/2026-06-30-cec-m05-premise.md`. |

**The meta-lesson (the most important thing in this prompt):** all four dead ends shared ONE assumption —
that the objective is **full-catalog, in-distribution completion accuracy**, for which co-occ is optimal.
The way out is to **change the estimand / objective / test-distribution / deliverable** so co-occ is no
longer the optimal baseline. Prioritize lanes that do this.

Also inherited: **identifiability wall** — MealRec & Spotify-MPD have **no exposure/impression logs**;
Spotify-MPD has **no user ids**. So IPS/uplift/IV/slate-OPE lines that need propensities or instruments
are unidentifiable *as stated* — re-state as observational contrasts with declared assumptions, or pick
a dataset/objective that supplies the needed signal.

---

## 3. Survey-grounded FRESH lanes (de-duplicated across the four surveys + the prior ranking)

Four surveys in `refs/` were re-read for this brainstorm (read them for depth; condensed digests follow):
- **A Survey on Bundle Recommendation** (Sun et al., CSUR 2026) — bundle-rec field structure + open challenges.
- **A Survey on Causal Inference** (Yao et al., TKDD 2021) — general causal toolbox.
- **Causal Inference in Recommender Systems** (Gao et al., TOIS 2024) — causal-recsys taxonomy + 4 future directions.
- **Causal Inference with Large Language Model** (Ma, NAACL 2025) — the LLM×causal lane (untried here).

Each lane below is tagged **[escapes wall?]** and **[dead-end risk]**. None is a recommendation —
they are the option space to brainstorm, score, and (crucially) attach a cheap falsification to.

### L-A. Within-set INTERACTION / SYNERGY estimand  — *converges across 2 surveys + prior line #4*
Causal interaction (Egami & Imai 2019, named OPEN in Yao §7; also the Gao "set-level" gap):
`τ_synergy(j|S) = E[Y(S∪{j})] − E[Y(S)] − E[Y({j})] + E[Y(∅)]` — the *non-additive* effect of adding j
given S, beyond the sum of marginals. **[escapes wall: yes — a counterfactual not recoverable from
marginal co-occ]. [dead-end risk: low, but]** the killer risk is empirical collapse: if τ is identified
from co-purchase alone it may correlate ~1 with co-occ. **Cheap falsification:** measure the non-additive
synergy residual magnitude (after removing additive pairwise effects) on a held-out slice; if ≈0 or
rank-equivalent to co-occ, drop. This is the cleanest differentiator vs Cadence (which is additive
item-item) and is the surviving line from the prior ranking.

### L-B. Within-bundle INTERFERENCE / SUTVA-violation  — *Yao §4.1*
Items in a bundle definitionally interfere (adding one changes the others' value). Model bundle quality
via interference-aware estimators (spillover effects). **[escapes wall: potentially — spillover ≠
frequency]. [risk]** the interference graph reverts to co-occ without exogenous variation. Closely
related to L-A; treat as the formal causal scaffolding for synergy.

### L-C. OBJECTIVE change — off-policy / DTR / causal-utility  — *Yao §6.4 + Gao §5.4 + bundle-survey F*
Optimize **bundle-level reward** `E[Y(B)]` (purchase of the full bundle, completion rate, engagement),
not `P(items co-occur)`. Framings: off-policy / doubly-robust slate-policy learning (Swaminathan 2017);
sequential **dynamic treatment regime** / Q-learning over item-by-item construction. **[escapes wall: yes
— co-occ predicts popularity, not interventional reward]. [risk]** needs logged reward or a simulator;
slate variance; our datasets lack reward logs → may need a (causal) simulator or a proxy reward.

### L-D. DISTRIBUTION change — OOD / cross-domain / temporal robustness  — *Gao §5.2 + bundle-survey D,E*
Co-occ has **zero** OOD signal. Test the causal method on cross-genre / cold-catalog / temporal-cut /
cross-domain bundle splits; claim a verified gain **where co-occ structurally cannot compete**.
**[escapes wall: yes — the wall is in-distribution only]. [risk]** must commit to the OOD metric as
primary; cold-item OOD overlaps dead end #4's low ceiling, so prefer cross-domain / temporal / genre-shift
over pure cold-item. Cross-domain (NetEase+Youshu shared users; Douban music/books/movies) gives
near-zero associational baseline.

### L-E. DELIVERABLE change — causal-DISCOVERY of an item-item DAG  — *Gao §5.1*
Produce an **interpretable item-item causal graph** (complementarity edges with direction/necessity) as
the *primary output*; co-occ is an **input feature**, not a competitor. Evaluate by downstream OOD
completion, explainability, and human/LLM plausibility. **[escapes wall: yes — co-occ is not a competing
model]. [risk]** if you still score it only by held-out completion accuracy you are back at the wall —
the DAG must be the contribution.

### L-F. LLM × CAUSAL  — *Ma 2025; entirely untried here*
The LLM lane is fresh (existing LLM-bundle methods — Text2Bundle, AICL, Bundle-MLLM — are NON-causal).
Concrete sub-angles, best first:
- **L-F1. LLM counterfactual bundle EVALUATION as intervention-verified labels.** LLM judges which
  held-out items are *causally necessary* vs *coincidental*; stratify metrics on the necessary subset.
  Redefines the target (objective change) and supplies the do()-verification labels the project lacks.
- **L-F2. LLM pairwise causal discovery of item-item complementarity** (does owning i create a
  *functional need* for j?) — a signal distinct from co-occ direction/popularity; feeds L-E or a reranker.
- **L-F3. LLM-as-causal-generator** for candidates (esp. where co-occ has no signal) → re-rank with a
  behavioral model.
**[escapes wall: potentially — different signal source / objective]. [MANDATORY falsification — the
"causal parrots" check]:** correlate the LLM causal scores with co-occ PPMI; **if r > ~0.7 the LLM is
just recalling co-occ from training data and there is no genuine causal gain** — kill the lane. Also
guard against benchmark contamination (the LLM may have seen these catalogs). This built-in cheap kill is
exactly the project's style.

### L-G. INTENT-mediated causal generation  — *bundle-survey #1 generative-BR open challenge*
`do(Intent = θ) → item inclusion`; objective = **intent coherence** (annotated on the Amazon-intent
bundle datasets, or LLM/human-judged), not recall. **[escapes wall: yes — different objective]. [risk:
moderate]** distinct from dead end #1 (different intervention point: intent in a *generator*, not a
user-theme in a *scorer*) ONLY if intent is not just re-learned from co-occ — needs an instrument or
exclusion restriction; otherwise it collapses into the refuted user-theme.

### L-H. FAIRNESS / counterfactual-exposure objective  — *bundle-survey §6.4*
Objective = diversity / exposure / Gap / Coverage, not accuracy → the co-occ wall disappears by
construction. **[escapes wall: yes]. [caveat]** this departs from the project's accuracy-completion goal;
only pursue if you accept a fairness/diversity contribution. Distinct from dead end #2 (fairness estimand,
not accuracy-deconfounding).

### L-I. PARTIAL IDENTIFICATION / bounds + sensitivity analysis  — *absent from Yao, flagged HIGH-fresh*
Rather than a point estimate, **bound** the causal completion gain under unmeasured confounding; a
rigorous way to *distinguish* a real causal gain from a co-occ artifact, and a methodological
contribution in its own right. **[escapes wall: reframes the claim]. [risk]** more theory than method;
could pair with any of L-A…L-G as the rigor layer.

### Carry-over from the prior ranking (`research/promising-research-lines.md`)
The prior top picks (S1 observable-gated deconfounding reranker; S2 Set-PDA decode-path deconfounding)
are **popularity-deconfounding-family** — now further weakened by dead end #2 (Cadence prior art) and the
M0.5 result. The prior **S3 = within-set synergy** is exactly **L-A** and is the surviving converging
line. Treat the deconfounding lines as down-weighted; treat synergy + the objective/distribution/
deliverable/LLM lanes as the fresh frontier.

---

## 4. Selection rubric (score each candidate line 1–5)

| criterion | meaning |
|---|---|
| **Escapes-the-wall** | changes estimand/objective/distribution/deliverable so co-occ is NOT the optimal baseline (the meta-lesson). |
| **Novelty** | not a dead end (§2), not Cadence/PDA/MACR/UpliftRec/OPCB prior art, not already in the surveys' covered work. |
| **Intervention-verifiable** | a real causal term that passes inertness + rank-inversion + `do()`-flip — not a relabel of co-occ. |
| **Cheap-falsifiable** | has a CPU-minutes up-front kill (synergy-residual magnitude; LLM-vs-co-occ correlation; OOD-baseline-near-random check) BEFORE any GPU. |
| **Identifiable-on-our-data** | works given no exposure logs / no user ids (Spotify-MPD) — or names the dataset/signal that supplies what it needs. |
| **Feasibility** | tractable vs the existing harness + baselines. |

Rank by total; for the top 2–3, write a one-paragraph design sketch naming the estimand, the placement
(decode-path / reranker / discovery / eval), the objective/metric, and the **up-front falsification gate**.

---

## 5. Mandatory baselines & diagnostics (any empirical claim)
Baselines: raw co-occurrence, retrieve-then-rerank (PPMI-SVD), PPMI-normalized co-occ, **Cadence-UACR**
(adapt), content SetCompleter, CLHE (full-catalog anchor), DDBC (full-catalog, NOT ρ=100 shortlist).
Diagnostics: ablation Δ, τ@0 / set-correlation vs the ~99% sampler-noise null, gradient-flow, the
**rank-inversion** test, and a **do()-flip** wherever a causal term is claimed.

---

## 6. Datasets (and their limits)
- **Construction benchmark:** CLHE `xhLiu/BundleConstruction` — POG / POG_dense / Spotify / Spotify_sparse
  (the DDBC / Bundle-MLLM benchmark). Dense/sparse pairs give a density spread.
- **Ranking trio (canonical bundle-rec):** Youshu / NetEase / iFashion (Recall@20/40, NDCG@20/40).
- **MealRec+ H/L** (dense food; H/L density pair; cold-item splits; precomputed `exposure.npy`).
- Intent-annotated Amazon bundles (for L-G), cross-domain (NetEase+Youshu shared users; Douban) for L-D.
- Limits: no exposure/impression logs anywhere; Spotify-MPD has no user ids; DDBC's reported numbers are
  shortlist-only; Spotify-MPD redistribution is restricted (obtain officially).

---

## 7. Reusable assets (port / reuse — do not rebuild)
- This repo: `src/cec/` (exposure percentile binning + per-bin paired stratified stats, tested);
  `scripts/cec_m05_exposure_premise.py` (memory-safe co-occ via item×bundle incidence mat-vec) and
  `scripts/cec_m05_mealrec.py` (retrains SetCompleter; dense). These already implement a cheap
  exposure-stratified completion analysis — reuse for any new falsification.
- Predecessor `/Users/anisiomlacerda/code/Bundle_construction/src/causalbr/` (UNPUBLISHED — reuse, don't
  cite): full-catalog metrics (`recovery/ranking_metrics.py`, `recovery/set_metrics.py`), significance
  (`eval/significance.py`), the inertness/`do(T)`/oracle diagnostics (`eval/{inertness,theme_contrast,
  oracle_ceiling,loadbearing_driver}.py`), the ObservableGate (`causal/gate.py`), SetCompleter
  (`recovery/set_completer.py` + checkpoints `models/setcompleter_{spotify90,pog}.pt`), DDBC decode
  (`recovery/ddbc_decode.py` + `models/models_{spotify90,pog}.pt`), loaders (`datasets/*`), exposure
  tooling (`eval/popularity*.py`, `eval/lift.py`, `diagnostics/exposure_decorrelation.py`).
- Compute: a remote GPU box exists for heavier runs (was down last session — confirm it's up before
  relying on it; M0.5-style falsifications run on CPU in minutes and do NOT need it).

---

## 8. Prior context to read first (in this repo)
- `research/promising-research-lines.md` — the prior ranking (reconcile with it).
- `research/outputs/causal-recsys-bundle-construction.md` (+ `.provenance.md`) — the deep landscape.
- `docs/findings/2026-06-30-cec-m05-premise.md` — the M0.5 cold-item refutation (dead end #4, with numbers).
- `docs/findings/2026-06-30-strategic-rethink-after-cec.md` — the 0/2 pattern + the "change the objective"
  recommendation (this prompt operationalizes it).
- `docs/findings/00-SUMMARY.md`, `docs/learnings/LEARNINGS-from-bundle-ranking-project.md`,
  `docs/baselines/ddbc-repro-spec.md` — the inherited findings, the inertness history, the DDBC spec.

---

## 9. Output
1. A re-ranked `research/promising-research-lines.md` (rubric §4), with the dead ends explicitly listed
   as closed and each new line tagged escapes-wall / novelty / verifiable / cheap-falsifiable / identifiable.
2. The 2–3 top candidate designs, each with a **cheap up-front falsification gate** to run FIRST.
3. Hand the top pick to `writing-plans`, scoped so the **first implementation slice is the falsification
   gate** (kill-or-proceed before any training), mirroring the M0.5 pattern that worked.
