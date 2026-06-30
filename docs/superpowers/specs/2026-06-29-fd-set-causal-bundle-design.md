# FD-Set — A Front-Door SCM for Causal Bundle Construction & Ranking (Design Spec)

> Date: 2026-06-29. Status: **design approved in brainstorming; ready for writing-plans.**
> Upstream: [research/promising-research-lines.md](../../../research/promising-research-lines.md),
> [research/outputs/causal-recsys-bundle-construction.md](../../../research/outputs/causal-recsys-bundle-construction.md).
> Inherited constraints: [docs/findings/00-SUMMARY.md](../../findings/00-SUMMARY.md),
> [docs/learnings/LEARNINGS-from-bundle-ranking-project.md](../../learnings/LEARNINGS-from-bundle-ranking-project.md),
> [docs/baselines/ddbc-repro-spec.md](../../baselines/ddbc-repro-spec.md).
> Working name: **FD-Set** (Front-Door Set). Confidence tags on external numbers are in §11.

## 1. Context & motivation

**Goal.** A new causal-inference method for **bundle construction** (seed → complete a set) *and*
**bundle ranking** (score pre-built bundles for a user), unified by one Structural Causal Model.

**The bar (inherited, non-negotiable).**
- Full-catalog evaluation; shortlist/sampled-neg eval over-claims (proven twice in the predecessor).
- Beat **co-occurrence AND retrieve-then-rerank full-catalog** — not just DDBC (plain co-occ already
  beats DDBC full-catalog; DDBC's reported gains are within a **ρ=100 shortlist**, confirmed verbatim
  from its `evaluator.py`).
- Show a **real, intervention-verified causal gain.** The predecessor's user-theme `do(seed)` term was
  **inert** (flipping it moved output ~2% vs a ~99% sampler-noise null). Causal terms must be
  **decode-path or observable-gated**, never globally forced (global forcing displaced clean accuracy 3×).

**The gap (from the deep sweep, primary-source verified).** No set/bundle-level causal **estimand for
construction**, and no causal **generative/diffusion decoder**. The building blocks exist and are
non-inert (PDA, MACR, D3, UpliftRec, OPCB); only their combination is unoccupied.

**The collisions (must differentiate).** **Cadence** (arXiv 2512.17733) already deconfounds item-item
co-purchase vs popularity — so "deconfound co-occ vs popularity" is **prior art**; it is a *diversity*
reranker on LightGCN, runs **no `do()`-flip verification**. **A2G-DiffRec** (arXiv 2602.14706) does
decode-path guidance on a set-level diffusion recommender but with a **global learned weight** for
*fairness/long-tail* (the globally-forced anti-pattern). **DieT** ("Headache to Overstock?",
arXiv 2411.19107) documents popularity bias in bundle construction and debiases it **non-causally** —
independent motivation that leaves the causal route open.

## 2. The unified SCM

One DAG, two interventions:

```
            Z  (item popularity / joint-exposure — the CONFIRMED confounder)
          /   \
         v     v
   U → [  B  ] → Y           U = user, B = bundle composition (set),
         ^                    Y = completion / feedback
  seed S ┘
         └─► M (partial-bundle / set-fit MEDIATOR) ─► Y
```

- **Construction estimand:** `do(item ∈ set | partial set)` → complete B.
- **Ranking estimand:** `do(show b | u)` → score a pre-built B. Same DAG, second `do`-operator.
- **Confounder Z = popularity / joint-exposure** (the signal our data confirmed; **not** the refuted
  user-theme). Z taints the B→Y / co-occurrence path.
- **Why front-door (spine).** The partial bundle **M** is an *observed mediator* on seed→completion.
  Front-door identifies `P(Y | do(seed)) = Σ_m P(m|seed) Σ_{seed'} P(Y|m,seed') P(seed')` even when Z
  (or a latent "concept") is **unobserved** — so it works on **Spotify-MPD** (no exposure logs, no user
  ids) where backdoor/propensity cannot. It also handles the unobserved confounder that plain co-occ and
  Cadence's *observed*-confounder backdoor cannot.

## 3. The method — FD-Set

Same front-door estimand, **two placements**, **two tracks**.

### 3.1 Phase 1 — FD-Set reranker (identifiable, bar-shot)
- Backbone = retrieve-then-rerank (co-occ retrieves candidates from seed/partial bundle).
- Replace the raw co-occ rerank score with the **factorized front-door partial-effect** (HCR-style):
  `g(j|S) = h₁(M(S), j) · h₂(M(S))`, where M(S) is the partial-bundle/set-fit mediator and the
  seed′-marginal is collapsed algebraically. This estimates `P(complete | do(seed S), add j)`
  deconfounded of the unobserved Z.
- **Estimated from observed completed bundles — no propensities, no exposure logs, no user ids** →
  runs on Spotify-MPD. Selection = argmax front-door score (changes the estimand → non-bypassable).

### 3.2 Phase 2 — FD-Set decode-path (novelty; gated on Phase 1)
- Port the *same* front-door term into a generative decoder (DDBC clamp/generate = the `do(item∈set)`
  carrier). At each unmask step, sample partial-bundle continuations (the mediator the decoder already
  materializes; Causal-Prompting style), marginalize, and **reweight the per-pick logit** —
  **not** the inert `cond_dim=128` FiLM slot.
- **Observable-gate** the term (gate-g): full strength on cold/exposure-imbalanced slots, damped on
  dense clean co-occ. Never globally forced (A2G-DiffRec's global weight is the anti-pattern).

### 3.3 Ranking track (the 2nd `do`-operator)
- Mediator M = the **item-level user affinities** the bundle's constituent items induce (U→item-affinities→
  bundle relevance), Z = popularity. Estimand `P(Y | do(show b | u))` deconfounded.
- Ranking datasets carry user-ids → **backdoor on observed popularity is also available**; where both
  identify, **front-door must agree with backdoor** (an identification cross-check).
- Contribution: the **first rigorous-causal (identified, intervention-verified) bundle ranking** method.
  BunCa (asymmetric relation) and CLBR (counterfactual augmentation) are causal-*flavored*, not
  identified; CoHeat/GAMNBRec attack popularity bias non-causally.

### 3.4 Inter-phase / inter-track kill-gate
Phase 2 (and heavy generative work) runs **only if** Phase 1's front-door term shows held-out signal
co-occ lacks: τ@0 ≪ baseline; **rank-inversions** vs co-occ **and** PPMI; beats Cadence-UACR. If Phase 1
is inert/cosmetic, stop — do not bet on the weak DDBC decode.

## 4. Identification & assumptions (stated, falsifiable)

- **Front-door validity:** (i) M intercepts all directed seed→Y paths; (ii) no unblocked back-door
  seed→M; (iii) all back-door M→Y paths are blocked by seed. **Risk:** popularity leaking into *both*
  seed→M and M→Y breaks (iii) — checked by the front-door-vs-backdoor agreement test on ranking data.
- **Reranker estimand (Phase 1):** identified **without** propensities under an explicit
  *no-unobserved-confounding-given-the-partial-set* assumption — declared as a limitation, not hidden.
- **Positivity:** front-door marginalization needs adequate mediator support; on sparse catalogs
  (Spotify-254k) high-variance mediator sampling can wash the term out → monitored, gated.
- **Decode-path non-inertness (Phase 2):** the per-pick logit reweight is exercised every step by
  construction; verified empirically (§5.4), not assumed.

## 5. Evaluation

### 5.1 Two-world structure
Bundle work splits into a **construction** world (POG/Spotify, the CLHE benchmark) and a **ranking**
world (Youshu/NetEase/iFashion + MealRec+). FD-Set's unified SCM spans both; the cross-world transfer
is itself a claim to test.

### 5.2 Datasets (prioritized slate)
| # | Dataset | Track | Role |
|---|---|---|---|
| 1 | **Spotify** (MPD construction variant) | construction | table-stakes + cleanest front-door (raw MPD: no user-ids/exposure). *Benchmark form re-adds a user-id via Last.fm join — pure no-user-id story = raw MPD.* |
| 2 | **POG / iFashion** (construction variant) | construction | 2nd construction pillar; highest documented popularity bias; small bundles → gate fires per slot |
| 3 | **NetEase** | ranking | hard/sparse, avg bundle ~78, ~0.04 ceiling → headroom |
| 4 | **Youshu** | ranking | **easy control** — gate should stay inert → inertness/rank-inversion negative control |
| 5 | **MealRec+ H/L** | both | built-in high(0.77%)/low(0.17%)-density pair → controlled observable-gate stress-test; cold-item splits |
| 6 | iFashion (CrossCBR subset) | ranking (opt.) | completes the canonical ranking trio |

**Access risk:** Spotify-MPD redistribution is **prohibited** (obtain via Spotify Research); if blocked,
drop Spotify and keep POG as the construction pillar.

### 5.3 Baselines (per track; protocol flagged)
- **Construction:** DDBC (ICLR'26, SOTA — **ρ=100 shortlist**), Bundle-MLLM (KDD'25 — **10-way MCQ
  HitRate@1**), **CLHE (WSDM'24 — full-catalog table + code = anchor)**, generative TIGER/SASRec; plus
  our own co-occ / retrieve-rerank / **PPMI-normalized co-occ** / **Cadence-UACR (adapted)**.
- **Ranking:** CrossCBR (KDD'22), MultiCBR (TOIS'24, strongest classic), EBRec (TORS'24), **DCBR
  (AAAI'25, conditional diffusion, full-catalog)**, MoDiffE; cold-start **CoHeat (WWW'24)**; causal
  **BunCa (ECML-PKDD'24)** + **CLBR**; ladder MFBPR/LightGCN/SGL/DAM/BGCN/MIDGN. Related (popularity-bias,
  non-causal, cite don't necessarily beat): GAMNBRec, DieT.
- **Fair-comparison resolution:** report every method **both** full-catalog **and** under DDBC's ρ=100 /
  Bundle-MLLM's 10-MCQ protocol → demonstrates the shortlist→full-catalog collapse (our C1 critique) and
  gives protocol-matched head-to-heads. DDBC exact cells via cloning `LiAi16/DDBC` + running on HF
  checkpoints (its PDF is CAPTCHA-walled).

### 5.4 Inertness / intervention battery (every causal term, both phases)
- **Ablation Δ** — remove front-door → revert to co-occ/baseline; must move the primary metric.
- **τ@0 / set-correlation** under ablation — corrected vs baseline set must diverge ≫ the ~99%
  sampler-noise null (predecessor's `do(seed)` managed ~2%).
- **Rank-inversion** — corrected ranking must produce inversions vs **both** raw co-occ and PPMI (a
  monotone reweight changes scores but not order = inert in effect).
- **Gradient-flow** — front-door pathway carries non-negligible gradient at convergence.
- **`do()`-flip** — flip the mediator/added item, measure outcome shift vs the sampler null (the test
  Cadence never runs — our methodological contribution).

### 5.5 Win conditions (pre-registered)
- Beat raw co-occ, retrieve-rerank, **PPMI-normalized co-occ**, **Cadence-UACR**, and DDBC (full-catalog,
  not shortlist) on construction; CrossCBR/MultiCBR/EBRec/DCBR + BunCa/CLBR on ranking.
- Gain **concentrated where the observable gate fires** (cold/exposure-imbalanced slots; MealRec+ L > H;
  near-zero on the Youshu control).
- Front-door variant beats a non-front-door ablation (proves the causal term, not the backbone, is
  load-bearing).
- Honesty guards: ≥5 seeds, paired Wilcoxon, pre-registered thresholds, negatives reported plainly.

## 6. Novelty & differentiation
| Prior art | FD-Set differentiator |
|---|---|
| Cadence (item-item co-occ deconfounding, diversity, no flip-test) | set/decode-level estimand on a **constructor**, `do()`-flip verified, completion not diversity; Cadence-UACR is a baseline |
| A2G-DiffRec (global decode guidance, fairness) | **observable-gated** (per-slot), formal causal estimand, complementarity not long-tail |
| DDBC / Bundle-MLLM (non-causal, shortlist) | causal decode-path term + **full-catalog** eval |
| BunCa / CLBR (causal-flavored ranking) | **identified** front-door/backdoor, intervention-verified |
| UpliftRec (category-ratio CATE) / OPCB (flat set bandit) | per-item add-to-partial-bundle, generative/iterative constructor |
| DieT (non-causal long-tail debiasing) | causal identification + decode-path placement |

## 7. Risks & kill-criteria
(See the table presented in design Section 4 — reproduced.)
| risk | mitigation | KILL if |
|---|---|---|
| PPMI-collapse (monotone decode term) | per-slot observable γ + rank-inversion test | no inversions vs co-occ **and** PPMI |
| weak DDBC backbone | Phase-1 reranker first | Phase-1 term inert (τ@0 ≈ null) |
| front-door unidentifiable (popularity in both paths) | mediator-validity + backdoor cross-check | front-door ≠ backdoor where both identify |
| synergy ≈ additive (the S3 spine) | falsify residual magnitude first | residual ≈ 0 |
| two worlds = two bolted methods | share the front-door mediator machinery | SCM doesn't transfer construction↔ranking |
| Cadence/A2G collision | set + decode + `do()`-flip + interaction | can't beat Cadence-UACR |
| Spotify-MPD access (redistribution prohibited) | obtain officially; POG fallback | blocked → drop Spotify |

## 8. Milestones
- **M0** — full-catalog harness + baselines (co-occ, RR, PPMI, CLHE, DDBC re-run, Cadence-UACR) on the
  construction pair + ranking trio.
- **M1** — Phase-1 front-door **reranker, construction** (POG/Spotify). **Go/no-go gate.**
- **M2** — Phase-1 front-door **ranking** (Youshu/NetEase/iFashion + MealRec+).
- **M3** — Phase-2 **decode-path** construction (only if M1 passes).
- **M4** — inertness battery + popularity-confounded scenario (DieT-style long-tail slice) +
  observable-gate stress (MealRec+ H/L; Youshu inertness control).

> **First implementation slice (for writing-plans):** scope the first plan to **M0 + M1 only** — the
> full-catalog harness/baselines and the Phase-1 front-door construction reranker, ending at the
> go/no-go gate (§3.4). M2–M4 are planned *after* M1 passes, to avoid building on an unverified estimand.

## 9. Reusable assets (`code/reference/`)
`ranking_metrics.py` (full-catalog multi-gt recall@k / hit@1 / MRR + per-example), `significance.py`
(paired Wilcoxon / bootstrap / Bonferroni), `ddbc_decode.py` (full-catalog DDBC scoring),
`reranker.py` + `fusion.py` (Phase-1 backbone). Re-implement the inertness diagnostics in the new harness.

## 10. Open questions
- Does front-door deconfounded co-occ beat **both** PPMI and **Cadence-UACR**, or collapse to popularity
  normalization? (M1 adjudicates.)
- Is the front-door mediator valid (no popularity leak into both paths)? (front-door-vs-backdoor test.)
- Does the SCM genuinely transfer construction↔ranking, or are they two methods? (M1+M2 cross-check.)
- Is within-set synergy non-additive after deconfounding? (S3 falsification gate.)
- Spotify-MPD obtainable under license in time? (access risk.)

## 11. References & verification confidence
**Verified verbatim / from primary tables (high):** DDBC eval protocol (repo `evaluator.py`, ρ=100
shortlist); CLHE (WSDM'24, arXiv 2310.18770 — full-catalog table read from PDF); Bundle-MLLM (KDD'25,
arXiv 2407.11712 — 10-MCQ protocol + numbers); DCBR (AAAI'25, repo recomall/DCBR — all-ranking + numbers);
EBRec (TORS'24, arXiv 2311.16892 — tables); MultiCBR/CrossCBR/BGCN/MIDGN/DAM/SGL/LightGCN/MFBPR (MultiCBR
Table 2); DieT (arXiv 2411.19107 — title/authors/scope verified; *its POG/Spotify + "Pop-to-LT" specifics
NOT confirmed from the abstract*). **Verified resolving (from the prior cite pass):** Cadence (2512.17733),
A2G-DiffRec (2602.14706), PDA (2105.06067), MACR (2010.15363), HCR (2205.07499), D3 (2406.14900), Causal
Prompting (2403.02738), UpliftRec (2405.08582), OPCB (2408.11202), DiffRec (2304.04971), DreamRec
(2310.20453). **Medium / re-verify before camera-ready:** DDBC exact per-cell numbers (CAPTCHA-walled —
re-run the repo); MoDiffE venue (arXiv 2505.05035, TOIS'26 unconfirmed); RDiffBR (2507.03280, AAAI'26
unconfirmed); BunCa (ECML-PKDD'24, 2408.08906); GAMNBRec numbers (internally inconsistent, non-comparable
scale); DGMAE numbers (not obtainable). **Corrections:** "CADSI" = CaDSI (general-rec causal, not bundle);
CausalBR/DCBR-as-causal/DPID do not exist as bundle methods; CLBR is *Counterfactual* (not Contrastive)
Learning, arXiv-only.
