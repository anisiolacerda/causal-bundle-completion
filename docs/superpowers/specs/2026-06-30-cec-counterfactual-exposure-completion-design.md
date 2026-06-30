# CEC — Counterfactual Exposure-balanced Completion (Design Spec)

> Date: 2026-06-30. Status: **design approved in brainstorming; ready for writing-plans.**
> **Supersedes** [2026-06-29-fd-set-causal-bundle-design.md](2026-06-29-fd-set-causal-bundle-design.md)
> (the front-door SCM). *Why the pivot:* BundleBench's own adversarial work falsifies the front-door
> spine — the textbook front-door is mis-specified for completion (a dominant direct seed→item path
> exists; `frontdoor-adversarial-ruling.md`), the user-theme mediator is refuted (C4), and
> popularity-deconfounding of co-occ collapses to prior art (Cadence) + a known negative (C3 sparse;
> PPMI already normalizes). **CEC instead attacks co-occurrence's structural BLIND SPOT** —
> cold/under-exposed items — where the causal/counterfactual framing is load-bearing rather than a relabel.
> Scope: **bundle CONSTRUCTION**, **method-first**. Inherited givens, BundleBench paths, and external-number
> confidence tags carried over from the FD-Set spec's §1.5/§11.
> Upstream: [research/promising-research-lines.md](../../../research/promising-research-lines.md),
> [research/outputs/causal-recsys-bundle-construction.md](../../../research/outputs/causal-recsys-bundle-construction.md);
> BundleBench at `/Users/anisiomlacerda/code/Bundle_construction` (`docs/FD-Set-contribution-overlap-eval.md`,
> `literature_synopsis/{frontdoor-adversarial-ruling,novelty-critique}.md`, `ideas/candidate_directions_backlog.md`).
> Working name **CEC** (rename welcome).

## 1. Goal & the bar (inherited from BundleBench — cite, do not re-claim)

A causal/counterfactual **bundle-completion method** that wins where the dominant baseline is
**structurally blind** — cold/under-exposed items — verified by an exposure-shift completion evaluation.

**The bar (inherited C1–C3, F-ledger in §1.5):** full-catalog eval (shortlist over-claims, C1); plain
**co-occurrence already beats DDBC and the learned family full-catalog** (C2), and retrieve-then-rerank
beats co-occ on **dense** catalogs only (C3). So a causal term has no room on co-occ's home turf — the
contribution must live where co-occ has **no statistics**. Causal terms must be observable-gated /
decode-path, never globally forced (3× displacement, F10); and must pass the inertness battery (C4/F5–F7).

**The reframe (lead-line + characterized boundary, per the project's own §8 learning):**
co-occurrence = co-consumption under the **factual** (exposure-confounded) distribution. True
complementarity = co-consumption under **`do(uniform exposure)`**. Split by item exposure `e`:
- **Warm items:** `do`-counterfactual ≈ factual → deconfounding adds ~0. *This explains the dead
  popularity-deconfounding line (Cadence/PPMI/C3) — it is the setup, not a failure.*
- **Cold/under-exposed items:** `do`-counterfactual ≫ factual; **co-occ and DDBC are structurally
  blind** (co-occ has too few/zero co-stats; DDBC has no RVQ code). **All the opportunity is here.**

## 1.5 Relationship to BundleBench (inherited vs new)

**Inherited (F1–F10) — cite as prior own work; reuse/port, never re-derive.** Same ledger as the FD-Set
spec §1.5. The ones CEC leans on most:
- **F3a (decisive pre-evidence):** the content SetCompleter is *the only method that beats co-occ
  full-catalog* (on Spotify, 768-d CLHE); ID-only ablation 0.085→0.038 loses (p=5.8e-65). → content is a
  working outcome model where ID/co-occ run out — exactly CEC's cold-slice hypothesis, half-confirmed already.
- **C2/C3 (the setup):** co-occ wins warm; RR wins dense, loses sparse. CEC does not contest these — it
  exploits the *complement* (cold).
- **F7a ObservableGate** (`causal/gate.py`) — reuse for the exposure gate.
- **C1/F6/F7** the eval-critique + diagnostic suite (`do(T)` null, inertness report, oracle ceiling) — reuse.
- **C4/F5** the refuted theme — CEC's causal content is **exposure**, not theme.

**New (CEC's claims):**
1. The **`do(uniform exposure)` complementarity estimand** + the **warm≈factual / cold≫factual
   characterization** (turns the dead deconfounding line into a boundary result).
2. A **doubly-robust counterfactual cold-completer** (frequency-propensity × content-outcome).
3. An **exposure-gated two-regime completer for construction** (gate reused; this application is new).
4. The **counterfactual / exposure-shift cold-slice EVALUATION** (extends C1).
5. **Beating co-occ + DDBC + CLHE + content-SetCompleter on the cold slice, with warm-parity preserved.**

## 2. The estimand

For partial bundle (seed) `S` and candidate item `j` with exposure `e_j`:
- **Factual:** `f(j|S)` = observed co-consumption / co-occ complementarity (the C2 winner on warm).
- **Counterfactual target:** `c(j|S)` = co-consumption(`S+j`) under **`do(e_j = uniform/balanced)`** — what
  `j`'s compositional fit would be had it received balanced exposure. On warm `c≈f`; on cold `c≫f`.
- **Method output:** an exposure-gated blend (`§3`) that equals `f` on warm and surfaces `c` on cold.
The causal claim is **not** a point-ID of `P(Y|do(seed))` (that object is mis-specified, per the ruling);
it is the **controlled effect of the exposure intervention on completion ranking**, validated operationally
(`do`-flip + cold-slice eval), with the warm regime as the built-in negative control.

## 3. Architecture (Approach A — two-regime exposure-gated DR reranker)

```
seed S ─► content-augmented retrieval ──► candidate set (incl. COLD items co-occ would never surface)
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
        factual co-occ / RR  f(j|S)                          DR counterfactual  c(j|S)
        (the C2/C3 winner, untouched)                (frequency-propensity × content-outcome)
              └───────────────────────────────┬───────────────────────────────┘
                         score(j|S) = (1 − g(e_j))·f(j|S) + g(e_j)·c(j|S)
                         g = monotone exposure gate (reuse ObservableGate); g→0 warm, g→1 cold
```
- **Content-augmented retrieval** is mandatory: co-occ retrieval cannot surface cold candidates, so the
  candidate set must include content-near items (reuse the SetCompleter content space).
- **`c(j|S)` is doubly-robust:** propensity = item exposure/frequency model (the only exposure signal we
  have); outcome = a content/structure model imputing balanced-exposure compositional fit (reuse
  SetCompleter as the outcome model). DR is consistent if **either** is right — robust to the crude
  frequency-propensity.
- **The gate makes it non-inert and non-displacing:** g→0 on warm reverts to the proven `f` (warm-parity,
  no 3× displacement); g→1 on cold activates `c`. Ablating `c` (or setting g≡0) is the clean `do`-flip.
- **Why causal, not cold-start content rec:** the target is a counterfactual under `do(exposure)` (the
  ranking *changes* under the intervention — `do`-flip verifiable), DR-identified; content is the outcome
  model, not the claim. Cold-start content rec has no exposure-intervention estimand and no `do`-flip.

## 4. Identification & assumptions (stated, falsifiable)

- **Exposure as the treatment; frequency as the propensity proxy.** No slate/impression logs exist → the
  propensity is an item exposure/frequency model. Declared limitation; DR hedges it.
- **Outcome model (A2-style unconfoundedness given seed + content):** after conditioning on the seed and
  observed content features, the cold item's compositional fit is estimable from content/structure. This is
  the substantive, **operationally falsifiable** assumption (M0.5 / M2), not a point-ID theorem.
- **Positivity:** truly-zero-stat items rely entirely on the outcome model (propensity uninformative) — DR
  degrades gracefully to the content outcome there; report the zero-stat sub-slice separately.
- **`do`-flip non-inertness:** moving an item across the exposure gate (or ablating `c`) must change the
  completion vs a sampler null — the test the refuted theme failed and Cadence never runs.

## 5. Evaluation (method-first; the cold slice defines "win")

- **Cold slice — primary:** items below an exposure/frequency percentile; hold out their bundle
  memberships as cold-completion targets. Universal (frequency always computable), matches the estimand.
  **Robustness:** a temporal cut (items entering after a time) where timestamps exist. Reuse
  `eval/popularity_shift.py`, `eval/lift.py` (`causal_ood_gain`), `diagnostics/exposure_decorrelation.py`.
- **Metrics:** full-catalog Recall@K / NDCG@K / Hit@1 / F1 / Jaccard, **stratified by exposure percentile**;
  per-example variants for paired Wilcoxon (reuse `recovery/{ranking_metrics,set_metrics}.py`,
  `eval/significance.py`).
- **Baselines:** co-occ, retrieve-rerank, PPMI-norm co-occ, **content SetCompleter** (the must-beat on
  cold), CLHE (full-catalog anchor), DDBC (re-run; full-catalog + ρ=100), Cadence-UACR (the deconfounding
  collision). Related (cold-start, cite): inductive-RVQ / content-fallback variants.
- **Win conditions (pre-registered):**
  - Beat co-occ + DDBC + CLHE + **content-SetCompleter** on the **cold slice** (paired Wilcoxon, ≥5 seeds).
  - **Warm-parity:** on the warm slice, match the `f` baseline within noise (gate off → no displacement).
  - **`do`-flip** moves the completion above the sampler null; **rank-inversions** vs co-occ/PPMI on cold;
    the **DR variant beats outcome-only** (proves the exposure correction, not just content, is load-bearing).
  - The **warm≈factual / cold≫factual curve** (gain vs exposure percentile) as the characterization figure.
- **Datasets:** CLHE benchmark POG/POG_dense/Spotify/Spotify_sparse + MealRec+ H/L; exposure percentiles
  everywhere, temporal cuts where available. (Spotify-90 + POG already integrated; POG_dense/Spotify_sparse
  splits are new — same as before.)

## 6. Novelty & differentiation
| Prior art | CEC differentiator |
|---|---|
| Cadence (deconfound co-occ vs popularity, diversity reranking) | CEC concedes warm (≈factual) and wins **cold** via a `do(exposure)` counterfactual + content outcome; construction not diversity; `do`-flip verified |
| DDBC / Bundle-MLLM (non-causal, shortlist) | wins the cold slice where DDBC has no RVQ code; full-catalog (C1) |
| CLHE / content SetCompleter (associational content completion) | the **exposure-intervention estimand + DR + gate**, beating content-only on cold (the DR-vs-outcome-only ablation) |
| classic cold-start content rec | a counterfactual `do(exposure)` estimand with a `do`-flip, not content-similarity |
| this project's refuted theme front-door (C4, FD-Set) | causal content = **exposure on the cold slice**, where it is load-bearing, not theme/total-effect |

## 7. Risks & kill-criteria
| risk | mitigation | KILL if |
|---|---|---|
| collapses to content cold-start (causal thin) | DR + `do`-flip + exposure-intervention estimand + DR-vs-outcome-only ablation | `do`-flip ≈ null / DR ≯ outcome-only |
| frequency-propensity too crude (no exposure logs) | DR (robust if outcome right); report propensity-free outcome too | DR ≯ co-occ AND ≯ outcome-only on cold |
| content-outcome just relearns co-occ | rank-inversion vs co-occ/PPMI on cold | no inversions / no cold-slice gain |
| cold slice immaterial to headline | choose percentile by support; report cold-slice + curve, not just global | gated slice can't move any reported metric |
| warm displacement (the 3× trap) | gate g→0 on warm = exact `f` | warm parity fails |

## 8. Milestones (note the near-free premise check)
- **M0** — port harness/baselines/**SetCompleter**/ObservableGate/diagnostics/DDBC checkpoints; build the
  exposure-percentile cold slice (+ temporal cut where available; reuse `popularity_shift.py`/`lift.py`).
- **M0.5 — cheap falsification (mostly re-analysis of existing results).** Stratify the *existing*
  content-vs-co-occ Spotify win (F3a) by exposure percentile: **does content's win concentrate on cold
  items?** If yes, the premise is validated from already-computed numbers before any GPU. Also: does the DR
  counterfactual beat content-only on the cold slice, and does `φ`/propensity actually track exposure?
- **M1 — the exposure-gated DR counterfactual completer.** Cold-slice win + warm-parity. **Go/no-go gate.**
- **M2 —** `do`-flip + inertness battery (reused) + temporal-cut robustness + the warm/cold characterization
  curve. (Optional future: Approach C decode-path port.)

> **First implementation slice (for writing-plans):** **M0 + M0.5 + M1.** M0.5 is the cheapest possible
> falsification (re-analysis of owned results) and gates the GPU spend on M1; do it before building the DR
> completer. M2 follows only if M1's go/no-go passes.

## 9. Reusable assets (port from BundleBench — real paths)
Source: `/Users/anisiomlacerda/code/Bundle_construction/src/causalbr/`. **Vendored already in this repo
(`code/reference/`):** `ranking_metrics.py`, `significance.py`, `ddbc_decode.py`, `reranker.py`, `fusion.py`.
**To port (exist in BundleBench, not here):**
- **Outcome model / cold:** `recovery/set_completer.py` (+ checkpoints `models/setcompleter_{spotify90,pog}.pt`).
- **Gate:** `causal/gate.py` (ObservableGate), `model/gate.py`.
- **Exposure / cold-slice eval:** `eval/popularity.py`, `eval/popularity_shift.py`, `eval/lift.py`
  (`causal_ood_gain`, `personalization_lift`), `diagnostics/exposure_decorrelation.py`.
- **Metrics / diagnostics:** `recovery/set_metrics.py`, `eval/{metrics,construction_metrics,all_ranking}.py`,
  `eval/{inertness,theme_contrast,oracle_ceiling,loadbearing_driver,verdict}.py`, `diagnostics/hsic_permutation.py`.
- **Baselines / SOTA:** the recovery baseline family (co-occ, item-kNN, popularity, PPMI-SVD, `recovery/sasrec.py`,
  `recovery/set_completer.py`), `recovery/ddbc_decode.py` + `diffusion/*` + checkpoints `models_{spotify90,pog}.pt`.
- **Loaders:** `datasets/{mpd,mealrec,mealrec_construction,bundlerec,crosscbr_style,splits}.py`.
**Do NOT re-implement** the metrics, significance, gate, SetCompleter, or diagnostics — port them. New code =
the DR counterfactual estimator, the content-augmented retrieval for cold candidates, the exposure-percentile
cold-slice builder (if not already covered by `popularity_shift.py`), Cadence-UACR, and the POG_dense/Spotify_sparse splits.

## 10. Open questions
- Does content's existing co-occ-beating win (F3a) concentrate on cold items? (M0.5 — near-free.)
- Does the DR counterfactual beat content-outcome-only on cold (is the exposure correction load-bearing)? (M0.5/M1.)
- Is the frequency-propensity informative enough, or does DR collapse to the outcome model? (M1.)
- How large/important is the cold slice per dataset (does it move a headline)? (M0.)
- Spotify-MPD obtainable under license in time? (access risk — POG/MealRec+ fallback.)

## 11. References & verification confidence
Inherited BundleBench findings (C1–C4 / F1–F10) — see the FD-Set spec §1.5 and `docs/findings/*`,
`docs/learnings/*`, `docs/baselines/ddbc-repro-spec.md`, and BundleBench's `literature_synopsis/*` +
`ideas/candidate_directions_backlog.md` (#3 counterfactual-completion eval, #4 cold-item generalization,
#8 observable gate — CEC's seeds). **High-confidence external:** DDBC ρ=100 protocol (repo `evaluator.py`),
CLHE (arXiv 2310.18770), Bundle-MLLM (arXiv 2407.11712), DieT (arXiv 2411.19107, scope verified). **Verified
resolving:** Cadence (2512.17733), A2G-DiffRec (2602.14706), PDA (2105.06067), MACR (2010.15363), DiffRec
(2304.04971), DreamRec (2310.20453), TIGER (2305.05065). **Re-verify before camera-ready:** DDBC exact cells
(re-run the repo); MealRec+ construction reframing (our framing). **Deferred (out of scope):** ranking track,
decode-path port (Approach C).
