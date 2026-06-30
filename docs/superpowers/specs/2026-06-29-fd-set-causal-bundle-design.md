# FD-Set — A Front-Door SCM for Causal Bundle Construction (Design Spec)

> Date: 2026-06-29 (rev. construction-only). Status: **design in brainstorming review.**
> **Scope decision:** focus on **bundle CONSTRUCTION**. Bundle **ranking is deferred** to future work —
> revisited only after FD-Set reaches SOTA on construction. The SCM is built so ranking is the *same
> model's second `do`-operator* (kept as a forward-looking hook in §2/§3.3), but no ranking datasets,
> baselines, or milestones are in the active plan.
> Upstream: [research/promising-research-lines.md](../../../research/promising-research-lines.md),
> [research/outputs/causal-recsys-bundle-construction.md](../../../research/outputs/causal-recsys-bundle-construction.md).
> Inherited constraints: [docs/findings/00-SUMMARY.md](../../findings/00-SUMMARY.md),
> [docs/learnings/LEARNINGS-from-bundle-ranking-project.md](../../learnings/LEARNINGS-from-bundle-ranking-project.md),
> [docs/baselines/ddbc-repro-spec.md](../../baselines/ddbc-repro-spec.md).
> Working name: **FD-Set** (Front-Door Set). External-number confidence tags in §11.

## 1. Goal & scope

A new causal-inference method for **bundle construction** (seed → complete a set), with a real,
**intervention-verified** causal gain. **Bundle ranking is out of scope for now** and resumes only once
construction clears the bar (§5.5). The method stays a Structural Causal Model whose construction
`do`-operator is active and whose ranking `do`-operator is reserved for later.

**The bar (inherited, non-negotiable).**
- Full-catalog evaluation; shortlist/sampled-neg eval over-claims (proven twice in the predecessor).
- Beat **co-occurrence AND retrieve-then-rerank full-catalog** — not just DDBC (plain co-occ already
  beats DDBC full-catalog; DDBC's reported gains are within a **ρ=100 shortlist**, confirmed verbatim
  from its `evaluator.py`).
- Show a **real, intervention-verified causal gain.** The predecessor's user-theme `do(seed)` term was
  **inert** (flipping it moved output ~2% vs a ~99% sampler-noise null). Causal terms must be
  **decode-path or observable-gated**, never globally forced (global forcing displaced clean accuracy 3×).

**The gap (deep sweep, primary-source verified).** No set/bundle-level causal **estimand for
construction**, and no causal **generative/diffusion decoder**. Building blocks exist and are non-inert
(PDA, MACR, D3); only their combination is unoccupied.

**The collisions (must differentiate).** **Cadence** (arXiv 2512.17733) already deconfounds item-item
co-purchase vs popularity → "deconfound co-occ vs popularity" is **prior art**; it is a *diversity*
reranker, runs **no `do()`-flip verification**. **A2G-DiffRec** (arXiv 2602.14706) does decode-path
guidance on a set-level diffusion recommender but with a **global learned weight** for *fairness/long-tail*
(the globally-forced anti-pattern). **DieT** ("Headache to Overstock?", arXiv 2411.19107) debiases
popularity in bundle construction **non-causally** — independent motivation that leaves the causal route open.

## 2. The SCM (construction active; ranking reserved)

One DAG; the construction intervention is active now.

```
            Z  (item popularity / joint-exposure — the CONFIRMED confounder)
          /   \
         v     v
   U → [  B  ] → Y           U = user, B = bundle composition (set),
         ^                    Y = completion / feedback
  seed S ┘
         └─► M (partial-bundle / set-fit MEDIATOR) ─► Y
```

- **Construction estimand (active):** `do(item ∈ set | partial set)` → complete B.
- **Ranking estimand (deferred):** `do(show b | u)` — same DAG, second `do`-operator; future work.
- **Confounder Z = popularity / joint-exposure** (the signal our data confirmed; **not** the refuted
  user-theme). Z taints the B→Y / co-occurrence path.
- **Why front-door (spine).** The partial bundle **M** is an *observed mediator* on seed→completion.
  Front-door identifies `P(Y | do(seed)) = Σ_m P(m|seed) Σ_{seed'} P(Y|m,seed') P(seed')` even when Z
  (or a latent "concept") is **unobserved** — so it works on **Spotify** (no exposure logs; raw MPD has
  no user ids) where backdoor/propensity cannot. It also handles the unobserved confounder that plain
  co-occ and Cadence's *observed*-confounder backdoor cannot.

## 3. The method — FD-Set

Same front-door estimand, **two placements**, both for construction.

### 3.1 Phase 1 — FD-Set reranker (identifiable, bar-shot)
- Backbone = retrieve-then-rerank (co-occ retrieves candidates from seed/partial bundle).
- Replace the raw co-occ rerank score with the **factorized front-door partial-effect** (HCR-style):
  `g(j|S) = h₁(M(S), j) · h₂(M(S))`, where M(S) is the partial-bundle/set-fit mediator and the
  seed′-marginal is collapsed algebraically. Estimates `P(complete | do(seed S), add j)` deconfounded
  of the unobserved Z.
- **Estimated from observed completed bundles — no propensities, no exposure logs, no user ids** →
  runs on Spotify. Selection = argmax front-door score (changes the estimand → non-bypassable).

### 3.2 Phase 2 — FD-Set decode-path (novelty; gated on Phase 1)
- Port the *same* front-door term into a generative decoder (DDBC clamp/generate = the `do(item∈set)`
  carrier). At each unmask step, sample partial-bundle continuations (the mediator the decoder already
  materializes; Causal-Prompting style), marginalize, and **reweight the per-pick logit** — **not** the
  inert `cond_dim=128` FiLM slot.
- **Observable-gate** the term (gate-g): full strength on cold/exposure-imbalanced slots, damped on
  dense clean co-occ. Never globally forced (A2G-DiffRec's global weight is the anti-pattern).

### 3.3 Ranking track — DEFERRED (future work)
The same SCM scores a pre-built bundle via `do(show b | u)` with mediator = item-level user affinities.
**Not in the active plan.** Revisited only after construction reaches SOTA (§5.5). Recorded here so the
construction design choices (shared front-door mediator machinery, observable gate) stay reusable.

### 3.4 Inter-phase kill-gate
Phase 2 (and heavy generative work) runs **only if** Phase 1's front-door term shows held-out signal
co-occ lacks: τ@0 ≪ baseline; **rank-inversions** vs co-occ **and** PPMI; beats Cadence-UACR. If Phase 1
is inert/cosmetic, stop — do not bet on the weak DDBC decode.

## 4. Identification & assumptions (stated, falsifiable)

- **Front-door validity:** (i) M intercepts all directed seed→Y paths; (ii) no unblocked back-door
  seed→M; (iii) all back-door M→Y paths are blocked by seed. **Risk:** popularity leaking into *both*
  seed→M and M→Y breaks (iii).
- **Backdoor cross-check (construction).** On datasets where popularity is observable per item
  (POG / POG_dense; MealRec+), compute a backdoor-on-popularity estimate and require the front-door
  estimate to **agree** with it — a within-construction identification sanity check (replaces the
  deferred ranking cross-check).
- **Reranker estimand (Phase 1):** identified **without** propensities under an explicit
  *no-unobserved-confounding-given-the-partial-set* assumption — declared as a limitation, not hidden.
- **Positivity:** front-door marginalization needs adequate mediator support; on sparse catalogs
  (Spotify/Spotify_sparse, ~213–254k items) high-variance mediator sampling can wash the term out →
  monitored, gated.
- **Decode-path non-inertness (Phase 2):** the per-pick logit reweight is exercised every step by
  construction; verified empirically (§5.4), not assumed.

## 5. Evaluation (construction only)

### 5.1 Benchmark
The construction world = the **CLHE `xhLiu/BundleConstruction` benchmark** (POG / POG_dense / Spotify /
Spotify_sparse), reused by DDBC and Bundle-MLLM — maximum comparability with the SOTA we must beat.
The dense/sparse pairs give the density spread the observable gate needs.

### 5.2 Datasets (construction slate)
| # | Dataset | Role |
|---|---|---|
| 1 | **Spotify** (MPD construction variant) | primary; long bundles (~63); cleanest front-door (no exposure logs; raw MPD has no user ids). *CLHE benchmark form re-adds a user-id via Last.fm join — pure no-user-id story = raw MPD.* |
| 2 | **Spotify_sparse** | sparse variant (~213k items) → positivity/gate stress |
| 3 | **POG / iFashion** (construction variant) | fashion; small bundles (~3.6) so each slot is decisive; highest documented popularity bias (DieT) |
| 4 | **POG_dense** | dense variant → the dense control for the gate |
| 5 | **MealRec+ H/L** (meal completion) | optional contribution-showcase: same-domain high(0.77%)/low(0.17%)-density pair + cold-item splits + tiny full-catalog. *Reframed as completion; no construction-SOTA uses it → low comparability, high contribution.* |

**Access risk:** Spotify-MPD redistribution is **prohibited** (obtain via Spotify Research). If blocked,
drop Spotify/Spotify_sparse and keep POG/POG_dense (+ MealRec+) as the construction pillars.

### 5.3 Baselines (construction)
- **Generative SOTA (must beat):** DDBC (ICLR'26 — **ρ=100 shortlist**), Bundle-MLLM (KDD'25 — **10-way
  MCQ HitRate@1**), **CLHE (WSDM'24 — full-catalog table + code = anchor)**, generative TIGER / SASRec.
- **Our bar baselines:** co-occurrence, retrieve-then-rerank, **PPMI-normalized co-occ**,
  **Cadence-UACR adapted to completion** (the deconfounding-collision baseline).
- **Related work (cite; non-causal popularity-debiasing for construction):** DieT.
- *(Ranking baselines — CrossCBR/MultiCBR/EBRec/DCBR/BunCa/CLBR/CoHeat — are out of scope.)*

### 5.4 Experimental protocol
- **Metrics:** full-catalog Recall@K / NDCG@K (CLHE-comparable Recall@20/NDCG@20), Hit@1, F1, Jaccard,
  per-example variants for paired tests.
- **Dual-protocol fair comparison (the C1 contribution):** report every method **both** full-catalog
  **and** under DDBC's ρ=100 / Bundle-MLLM's 10-MCQ protocol → demonstrates the shortlist→full-catalog
  collapse *and* gives protocol-matched head-to-heads. DDBC exact cells via cloning `LiAi16/DDBC` +
  running on HF checkpoints (its PDF is CAPTCHA-walled).
- **Inertness / intervention battery (every causal term, both phases):**
  - **Ablation Δ** — remove front-door → baseline; must move the metric.
  - **τ@0 / set-correlation** under ablation — must diverge ≫ the ~99% sampler-noise null (predecessor's
    `do(seed)` managed ~2%).
  - **Rank-inversion** — corrected ranking must invert vs **both** raw co-occ and PPMI (a monotone
    reweight changes scores but not order = inert in effect).
  - **Gradient-flow** — front-door pathway carries non-negligible gradient at convergence.
  - **`do()`-flip** — flip the mediator/added item, measure outcome shift vs the sampler null (the test
    Cadence never runs — our methodological contribution).

### 5.5 Win conditions (pre-registered) — and the ranking unlock
- Beat raw co-occ, retrieve-rerank, **PPMI-normalized co-occ**, **Cadence-UACR**, and DDBC (full-catalog,
  not shortlist), and clear CLHE on the full-catalog metric.
- Gain **concentrated where the observable gate fires** (Spotify_sparse > Spotify; POG > POG_dense;
  MealRec+ L > H; near-inert on the dense controls).
- Front-door variant beats a non-front-door ablation (causal term, not backbone, is load-bearing).
- Honesty guards: ≥5 seeds, paired Wilcoxon, pre-registered thresholds, negatives reported plainly.
- **Ranking unlock:** once the above hold (construction at/above SOTA), resume the deferred ranking track
  (§3.3) as a follow-up spec.

## 6. Novelty & differentiation (construction)
| Prior art | FD-Set differentiator |
|---|---|
| Cadence (item-item co-occ deconfounding, diversity, no flip-test) | set/decode-level estimand on a **constructor**, `do()`-flip verified, completion not diversity; Cadence-UACR is a baseline |
| A2G-DiffRec (global decode guidance, fairness) | **observable-gated** (per-slot), formal causal estimand, complementarity not long-tail |
| DDBC / Bundle-MLLM (non-causal, shortlist) | causal decode-path term + **full-catalog** eval |
| DieT (non-causal long-tail debiasing in bundling) | causal identification + decode-path placement |
| UpliftRec (category-ratio CATE) / OPCB (flat set bandit) | per-item add-to-partial-bundle, generative/iterative constructor |

## 7. Risks & kill-criteria
| risk | mitigation | KILL if |
|---|---|---|
| PPMI-collapse (monotone decode term) | per-slot observable γ + rank-inversion test | no inversions vs co-occ **and** PPMI |
| weak DDBC backbone (ρ=100 shortlist flatters it) | Phase-1 reranker first | Phase-1 term inert (τ@0 ≈ null) |
| front-door unidentifiable (popularity in both paths) | mediator-validity + backdoor cross-check (POG/MealRec+) | front-door ≠ backdoor where both identify |
| positivity on sparse catalogs (Spotify_sparse) | gate + mediator-sample monitoring | mediator support too thin → term washes out |
| Cadence/A2G collision | set + decode + `do()`-flip + observable gate | can't beat Cadence-UACR |
| Spotify-MPD access (redistribution prohibited) | obtain officially; POG/MealRec+ fallback | blocked → drop Spotify, keep POG pillar |

## 8. Milestones (construction)
- **M0** — full-catalog harness + baselines (co-occ, RR, PPMI, CLHE, DDBC re-run, Cadence-UACR) on the
  CLHE benchmark (POG/POG_dense/Spotify/Spotify_sparse).
- **M1** — Phase-1 front-door **reranker**. **Go/no-go gate** (§3.4).
- **M2** — Phase-2 **decode-path** construction (only if M1 passes).
- **M3** — inertness battery + popularity-confounded scenario (DieT-style long-tail slice) +
  observable-gate stress (dense vs sparse variants; MealRec+ H/L; dense controls for inertness).
- **(future)** ranking track — only after M1–M3 reach construction SOTA.

> **First implementation slice (for writing-plans):** scope the first plan to **M0 + M1 only** — the
> full-catalog harness/baselines and the Phase-1 front-door construction reranker, ending at the
> go/no-go gate (§3.4). M2–M3 are planned *after* M1 passes, to avoid building on an unverified estimand.

## 9. Reusable assets (`code/reference/`)
`ranking_metrics.py` (full-catalog multi-gt recall@k / hit@1 / MRR + per-example), `significance.py`
(paired Wilcoxon / bootstrap / Bonferroni), `ddbc_decode.py` (full-catalog DDBC scoring),
`reranker.py` + `fusion.py` (Phase-1 backbone). Re-implement the inertness diagnostics in the new harness.

## 10. Open questions
- Does front-door deconfounded co-occ beat **both** PPMI and **Cadence-UACR**, or collapse to popularity
  normalization? (M1 adjudicates.)
- Is the front-door mediator valid (no popularity leak into both paths)? (front-door-vs-backdoor test on
  POG/MealRec+.)
- Is the decode-path term non-inert on the weak DDBC backbone, or does it beat DDBC yet lose to co-occ? (M2.)
- Can the gate concentrate gains on sparse/cold slots while staying inert on dense controls? (M3.)
- Spotify-MPD obtainable under license in time? (access risk.)

## 11. References & verification confidence
**Verified verbatim / from primary tables (high):** DDBC eval protocol (repo `evaluator.py`, ρ=100
shortlist); CLHE (WSDM'24, arXiv 2310.18770 — full-catalog table from PDF); Bundle-MLLM (KDD'25,
arXiv 2407.11712 — 10-MCQ protocol + numbers); DieT (arXiv 2411.19107 — title/authors/scope verified;
*its POG/Spotify + "Pop-to-LT" specifics NOT confirmed from the abstract*). **Verified resolving (prior
cite pass):** Cadence (2512.17733), A2G-DiffRec (2602.14706), PDA (2105.06067), MACR (2010.15363),
HCR (2205.07499), D3 (2406.14900), Causal Prompting (2403.02738), UpliftRec (2405.08582), OPCB
(2408.11202), DiffRec (2304.04971), DreamRec (2310.20453), TIGER (2305.05065). **Medium / re-verify before
camera-ready:** DDBC exact per-cell numbers (CAPTCHA-walled — re-run the repo); MealRec+ construction
reframing (no prior construction usage — our framing). **Deferred (ranking, out of active scope):**
CrossCBR / MultiCBR / EBRec / DCBR / MoDiffE / CoHeat / BunCa / CLBR.
