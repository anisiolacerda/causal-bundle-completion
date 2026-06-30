# FD-Set — A Front-Door SCM for Causal Bundle Construction (Design Spec)

> Date: 2026-06-29 (rev. 2026-06-30: construction-only **+ BundleBench overlap re-frame**).
> Status: **design in brainstorming review.**
> **Scope:** bundle **CONSTRUCTION** only; ranking deferred (§3.3, §5.5).
> **Overlap re-frame:** this project already owns the BundleBench D&B contributions (C1–C4), the
> full-catalog harness, the baselines, the diagnostic suite, and a (refuted) causal stack. FD-Set must
> **inherit and cite** those, not re-claim them. The inherited/new split is §1.5; the honest residual
> novelty is §1.6. Source of the overlap ledger: the "FD-Set vs BundleBench" evaluation (2026-06-30).
> Upstream: [research/promising-research-lines.md](../../../research/promising-research-lines.md),
> [research/outputs/causal-recsys-bundle-construction.md](../../../research/outputs/causal-recsys-bundle-construction.md).
> Inherited constraints/findings: [docs/findings/00-SUMMARY.md](../../findings/00-SUMMARY.md),
> [docs/learnings/LEARNINGS-from-bundle-ranking-project.md](../../learnings/LEARNINGS-from-bundle-ranking-project.md),
> [docs/baselines/ddbc-repro-spec.md](../../baselines/ddbc-repro-spec.md), the `docs/findings/2026-06-2*`
> stage records. Working name: **FD-Set** (Front-Door Set). External-number confidence tags in §11.

## 1. Goal & scope

A new causal-inference method for **bundle construction** (seed → complete a set), with a real,
**intervention-verified** causal gain. **Bundle ranking is out of scope for now** and resumes only once
construction clears the bar (§5.5). The model is a Structural Causal Model whose construction
`do`-operator is active and whose ranking `do`-operator is reserved for later.

**The bar — *inherited from BundleBench*, not established by FD-Set (cite C1–C3):**
- Full-catalog evaluation; the shortlist (ρ=100) protocol over-claims and *reverses* method rankings
  (**C1/F2**: the pool contains the ground truth; negatives are popularity-trivial).
- Plain **co-occurrence already beats DDBC full-catalog** and beats the whole learned family on DDBC's
  own benchmarks (**C2/F3**). So the FD-Set bar is **beat co-occ AND retrieve-then-rerank full-catalog**,
  not "beat DDBC."
- Causal terms must be **decode-path or observable-gated**, never globally forced (forcing displaced
  clean accuracy 3× in the ranking project — **F10**).
- The predecessor's user-conditioned, deconfounded `do(seed)` "theme" was **inert** (**C4/F5**): flipping
  θ moved the output only **τ0_flip ≈ 1.5–5 %** vs a different-decode-seed null **τ0_null ≈ 99 %**;
  8-seed median ΔF1 = −0.0004 (Bonferroni p = 1.0); an oracle θ added +0.001–0.003.

**The gap (deep sweep, primary-source verified).** No set/bundle-level causal **estimand for
construction**, and no causal **generative/diffusion decoder**. Building blocks exist and are non-inert
(PDA, MACR, D3); only their combination is unoccupied.

**The collisions (must differentiate).** **Cadence** (arXiv 2512.17733) already deconfounds item-item
co-purchase vs popularity → "deconfound co-occ vs popularity" is **prior art**; it is a *diversity*
reranker, runs **no `do()`-flip verification**. **A2G-DiffRec** (arXiv 2602.14706) does decode-path
guidance on a set-level diffusion recommender but with a **global learned weight** (the globally-forced
anti-pattern). **DieT** ("Headache to Overstock?", arXiv 2411.19107) debiases popularity in bundle
construction **non-causally** — the only *positive* popularity-debiasing signal is external and non-causal.

## 1.5 Relationship to BundleBench (inherited vs new — read before claiming anything)

**Inherited contributions/findings (F1–F10) — already this project's work; cite, do not re-derive:**

| id | inherited finding | key numbers |
|---|---|---|
| F1 | BundleBench artifact: full-catalog protocol + reference baselines + corrected leaderboard + diagnostic suite (the D&B contribution) | — |
| F2 (**C1**) | Eval-protocol critique + dual-protocol prescription | DDBC F1 0.30→**0.033** full-catalog (~9×); popularity hit@1 0.30 (ρ=100) vs 0.024 (full) ~13× |
| F3 (**C2**) | Trivial co-occ beats SOTA generative decode + whole learned family, full-catalog | co-occ>DDBC Spotify 0.053 vs 0.033 (p=2.1e-34), POG 0.139 vs 0.074 (p=5.4e-16); SASRec loses on 5 catalogs |
| F3a | Content ≠ complementarity | SetCompleter wins only on Spotify (768-d CLHE); ID-only ablation 0.085→0.038 (loses, p=5.8e-65) |
| F4 (**C3**) | Complementarity reranker **and its sparse-catalog limit** | retrieve-rerank beats co-occ on dense MealRec+H/L (p=0.031) but **loses on sparse Spotify** (.078–.052 vs co-occ .103); CF p=1.0 on large/sparse |
| F5 (**C4**) | Multi-way NULL refuting the user-theme `do(seed)` causal hypothesis | τ0_flip 1.5–5% vs τ0_null ~99%; HSIC p=0.156; 8-seed p=1.0; free user-emb degrades reranker |
| F6 | The `do(T)` sampler-noise null (τ@0) diagnostic — *shipped* | τ0 = 1 − Jaccard between two decode runs |
| F7 | Reusable diagnostic suite (shortlist-inflation, popularity-triviality, do(T) null, oracle ceiling, **3-part inertness report**) | — |
| F7a | A built **backdoor + decode-path** causal stack (Theme-iVAE, ObservableGate, SDPP estimand-marginal selector, collab proxy, BackdoorScoringHead) — **this is the refuted PIC-Diff** | — |
| F8 | 8-catalog completion leaderboard + "no single simple baseline dominates" (winner tracks density/bundle-size) | per-catalog table |
| F9 | DDBC reproduction + trained checkpoints (Spotify-90 **and** POG); DDBC eval = cosine-NN of CLHE feats, not likelihood | F1 0.298/0.302 Spotify, 0.135 POG |
| F10 | Methodological learnings: forced-global load-bearing displaced accuracy 3× → gate/decode-path; proxy substitutability; **spectral-energy regime criterion** (top-64 energy predicts collab-proxy payoff, ≈10–40%); gate-g validated | 3× displacement; Youshu 39.5%→helps, NetEase/iFashion <10%→no |

**Code provenance (this repo vs BundleBench).** Only `code/reference/{ranking_metrics, significance,
ddbc_decode, reranker, fusion}.py` are **vendored here**. The rest of the BundleBench tree (the diagnostic
suite, the causal stack, the other baselines, set_metrics, loaders, stage scripts, DDBC checkpoints)
lives in the predecessor project and must be **ported** — *not rebuilt, and not assumed already present
in this repo*. "Reuse" below means "port the existing module," never "re-derive."

## 1.6 Honest residual novelty FD-Set can legitimately claim

After subtracting F1–F10, FD-Set's defensible new contributions are exactly:

1. **A set/decode-level front-door causal estimand for construction** `g(j|S) = h₁(M(S),j)·h₂(M(S))` —
   the unoccupied cell from the deep sweep, and the **pivot from this project's refuted *backdoor*
   attempt (F7a)** to a *front-door* one. **(CORE — new claim.)**
2. **The front-door factorization carried as a per-pick logit reweight** — not the inert FiLM slot, and
   not the already-tried SDPP estimand-marginal selector. *Decode-path placement per se is not new (F7a);
   the front-door form of the term is.* **(NEW term; reuses the gate + selector scaffolding.)**
3. **A Cadence-UACR head-to-head on completion + a front-door-vs-backdoor agreement test.** **(NEW.)**
4. **POG_dense / Spotify_sparse gate-concentration experiments.** **(NEW data.)**

Everything else in this spec is inherited infrastructure/findings. The spec must not present any of
F1–F10 as an FD-Set contribution.

## 2. The SCM (construction active; ranking reserved)

```
            Z  (item popularity / joint-exposure — the IMPLICATED confounder)
          /   \
         v     v
   U → [  B  ] → Y           U = user, B = bundle composition (set),
         ^                    Y = completion / feedback
  seed S ┘
         └─► M (partial-bundle / set-fit MEDIATOR) ─► Y
```

- **Construction estimand (active):** `do(item ∈ set | partial set)` → complete B.
- **Ranking estimand (deferred):** `do(show b | u)` — same DAG, second `do`-operator; future work.
- **Confounder Z = popularity / joint-exposure.** Honest status (per C4/F5): popularity is the
  confounder the protocol-triviality (C1) and breadth popularity-bias (F8) **implicate**, and it is the
  alternative to the *refuted* user-theme — but **deconfounding popularity has no positive completion
  precedent in this project** (C4 was a null; the one causal-deconfounding attempt added ~0). The only
  positive popularity-debiasing signal is external and non-causal (DieT). So the premise is a
  **falsifiable hypothesis, not an inherited result.**
- **Why front-door (spine).** The partial bundle **M** is an *observed mediator* on seed→completion.
  Front-door identifies `P(Y | do(seed)) = Σ_m P(m|seed) Σ_{seed'} P(Y|m,seed') P(seed')` even when Z is
  **unobserved** — so it works on Spotify (no exposure logs; raw MPD has no user ids) where
  backdoor/propensity cannot, and it is the **pivot away from the refuted backdoor stack (F7a)**.

## 3. The method — FD-Set

Same front-door estimand, **two placements**, both for construction.

### 3.1 Phase 1 — FD-Set reranker (the genuinely-new estimand on the C3 backbone)
- Backbone = retrieve-then-rerank (co-occ candidates + PPMI-SVD CF) — **this is the inherited C3/F4
  pipeline**, already built.
- **New part:** replace the rerank score with the **factorized front-door partial-effect**
  `g(j|S) = h₁(M(S), j) · h₂(M(S))` (HCR-style; seed′-marginal collapsed algebraically), estimating
  `P(complete | do(seed S), add j)` deconfounded of the unobserved Z. Estimated from observed completed
  bundles — **no propensities, no exposure logs, no user ids**. Selection = argmax `g` (changes the
  estimand → non-bypassable).
- **Pre-registered expectation (priced from C3/F4, F8):** the same co-occ/PPMI statistics already
  **lose to co-occ on sparse Spotify and large/sparse breadth (p=1.0)**. So Phase-1's win is expected on
  **dense** catalogs (where C3 won — MealRec+H/L, POG_dense); a **Spotify_sparse win is the high-risk
  stretch goal, not the headline.** The front-door reweight must be shown not to collapse to a popularity
  normalization (§5.4 rank-inversion).

### 3.2 Phase 2 — FD-Set decode-path (gated on Phase 1; a continuation, not a fresh hook)
- Port the *same* front-door term into the generative decoder (DDBC clamp/generate = the `do(item∈set)`
  carrier). At each unmask step, sample partial-bundle continuations (the mediator the decoder already
  materializes), marginalize, and **reweight the per-pick logit** — **not** the inert `cond_dim=128` FiLM
  slot, **and not** the already-tried SDPP estimand-marginal selector (F7a).
- **Honest framing (F5/F7a):** *decode-path causal placement is not the novel hook* — a decode-path
  estimand-marginal selector was already built and was part of the **inert** PIC-Diff. The defensible
  delta is the **front-door factorization of the term**. The DDBC/MDLM decode's intrinsic sampler noise
  is **τ0_null ≈ 99 %** (C4) — that is the explicit bar: the per-pick reweight must move **τ0_flip well
  above the 99 % floor**. Phase-2 = forward-**option (c)** from the C4 REJECT (stronger injection under
  gate discipline), a continuation under a new term, not a new bet.
- **Observable-gate** (gate-g): full strength on cold/exposure-imbalanced slots, damped on dense clean
  co-occ — **reuse the built ObservableGate (F7a)**. Candidate gate signal: the **spectral-energy
  criterion (F10c)** (validated, ready) in addition to cold-fraction ρ̂.

### 3.3 Ranking track — DEFERRED (future work)
Same SCM via `do(show b | u)`, mediator = item-level user affinities. **Not in the active plan**;
revisited only after construction reaches SOTA (§5.5). Recorded so construction choices stay reusable.

### 3.4 Inter-phase kill-gate
Phase 2 (and heavy generative work) runs **only if** Phase 1's front-door term shows held-out signal
co-occ lacks: τ@0 ≪ baseline; **rank-inversions** vs co-occ **and** PPMI; beats Cadence-UACR. If Phase 1
is inert/cosmetic, stop — do not bet on the weak DDBC decode (its win is shortlist-only, F9).

## 4. Identification & assumptions (stated, falsifiable)

- **Front-door validity:** (i) M intercepts all directed seed→Y paths; (ii) no unblocked back-door
  seed→M; (iii) all back-door M→Y paths blocked by seed. **Risk:** popularity leaking into *both*
  seed→M and M→Y breaks (iii).
- **Backdoor cross-check (NEW, construction).** On datasets where popularity is observable per item
  (POG / POG_dense; MealRec+), compute a backdoor-on-popularity estimate and require the front-door
  estimate to **agree** — a within-construction identification sanity check. Caveat: the project's own
  backdoor causal attempt (F7a) was *refuted* on the completion gain, so agreement validates
  *identification*, not *payoff*.
- **Reranker estimand (Phase 1):** identified **without** propensities under an explicit
  *no-unobserved-confounding-given-the-partial-set* assumption — declared as a limitation.
- **Positivity:** front-door marginalization needs adequate mediator support; on sparse catalogs
  (Spotify/Spotify_sparse, ~213–254k items) high-variance mediator sampling can wash the term out.
- **Decode-path non-inertness (Phase 2):** verified empirically against the τ0_null ≈ 99% floor (§5.4),
  not assumed.

## 5. Evaluation (construction only)

### 5.1 Benchmark
Construction world = the **CLHE `xhLiu/BundleConstruction` benchmark** (POG / POG_dense / Spotify /
Spotify_sparse), reused by DDBC and Bundle-MLLM. Spotify-90 + POG are **already integrated** with loaders
and reproduced DDBC checkpoints (F9); **POG_dense / Spotify_sparse variant splits are absent** (the new
data work, and the dense/sparse pairs are what the gate hypothesis needs).

### 5.2 Datasets (construction slate)
| # | Dataset | Status | Role |
|---|---|---|---|
| 1 | **Spotify** (MPD construction variant) | integrated (F9) | primary; long bundles (~63); cleanest front-door (no exposure logs; raw MPD no user ids). *CLHE form re-adds a user-id via Last.fm join — pure no-user-id story = raw MPD.* |
| 2 | **Spotify_sparse** | **new split** | sparse (~213k items) → positivity/gate stress; **C3 failure regime** |
| 3 | **POG / iFashion** (construction variant) | integrated (F9) | fashion; small bundles (~3.6); highest documented popularity bias (DieT) |
| 4 | **POG_dense** | **new split** | dense control for the gate |
| 5 | **MealRec+ H/L** (meal completion) | integrated (F8) | the **C3-win dense regime** + same-domain density pair + cold-item splits; reframed as completion (low comparability, high contribution) |

**Access risk:** Spotify-MPD redistribution prohibited (obtain via Spotify Research). If blocked, drop
Spotify/Spotify_sparse; keep POG/POG_dense (+ MealRec+).

### 5.3 Baselines (construction)
- **Inherited, already built (F3/F4/F9) — port, don't rebuild:** co-occurrence, retrieve-then-rerank,
  **PPMI-normalized co-occ**, item-kNN, popularity, content SetCompleter, SASRec, **CLHE (full-catalog
  anchor)**, **DDBC (re-run; ρ=100 shortlist + full-catalog)**, TIGER.
- **New baseline (the one to add):** **Cadence-UACR adapted to completion** — the deconfounding-collision
  head-to-head.
- **Related work (cite; non-causal popularity-debiasing for construction):** DieT.
- *(Ranking baselines out of scope.)*

### 5.4 Experimental protocol
- **Metrics:** full-catalog Recall@K / NDCG@K (CLHE-comparable Recall@20/NDCG@20), Hit@1, F1, Jaccard,
  per-example variants — **reuse `ranking_metrics.py` / set_metrics + `significance.py`**.
- **Dual-protocol reporting (this is BundleBench's C1, NOT an FD-Set contribution):** report every method
  under BundleBench's full-catalog protocol **and** the shortlist protocols (DDBC ρ=100, Bundle-MLLM
  10-MCQ). FD-Set *uses* C1; it does not contribute it. DDBC exact cells via the reproduced repo (its PDF
  is CAPTCHA-walled).
- **Inertness / intervention battery — reuse the *shipped* diagnostics (F6/F7), do not re-implement:**
  - **`do(T)` sampler-noise null (τ@0)** — *introduced in BundleBench (C4/F6)*; here **applied** to the
    front-door term. The front-door reweight must push **τ0_flip ≫ τ0_null ≈ 99 %**. Cadence runs no such
    flip — that is the differentiation, but the *diagnostic is prior own work*.
  - **3-part inertness report** (ablation Δ, set-correlation, gradient-flow + load-bearing gate) — the
    packaged `inertness_report`.
  - **Oracle re-rank ceiling.**
  - **Rank-inversion vs co-occ AND PPMI** — **the one genuinely-new diagnostic line** (guards against a
    monotone popularity-reweight that changes scores but not order).

### 5.5 Win conditions (pre-registered) — and the ranking unlock
- Beat raw co-occ, retrieve-rerank, **PPMI-normalized co-occ**, **Cadence-UACR**, and DDBC (full-catalog),
  and clear CLHE on the full-catalog metric.
- **Primary win on dense catalogs** (MealRec+H/L, POG_dense) where C3 already wins; **Spotify_sparse is the
  stretch** (C3 lost there).
- Gain **concentrated where the observable gate fires** (sparse/cold slots), **inert on dense controls**.
- Front-door variant beats a non-front-door ablation (the causal term, not the backbone, is load-bearing),
  and **moves τ0_flip above the 99% floor**.
- Honesty guards: ≥5 seeds (8 for the causal null, matching C4), paired Wilcoxon, pre-registered thresholds.
- **Ranking unlock:** once the above hold (construction at/above SOTA), resume the deferred ranking track.

## 6. Novelty & differentiation (construction)
| Prior art | FD-Set differentiator |
|---|---|
| Cadence (item-item co-occ deconfounding, diversity, no flip-test) | set/decode-level **front-door** estimand on a constructor; we **apply BundleBench's `do(T)` flip** (Cadence runs none); completion not diversity; Cadence-UACR is a baseline |
| A2G-DiffRec (global decode guidance, fairness) | **observable-gated** (per-slot, reuses gate-g), formal front-door estimand, complementarity not long-tail |
| DDBC / Bundle-MLLM (non-causal, shortlist) | causal front-door decode-path term + **full-catalog eval (C1)** |
| DieT (non-causal long-tail debiasing) | causal identification + decode-path placement |
| **This project's own PIC-Diff (F7a, backdoor, refuted)** | **front-door** factorization (the pivot), per-pick logit reweight vs the inert SDPP selector |

## 7. Risks & kill-criteria
| risk | mitigation | KILL if |
|---|---|---|
| Phase-1 is the C3-sparse failure regime (same statistics already lost to co-occ) | pre-register the win on **dense**; sparse = stretch | no dense-catalog win over co-occ + Cadence-UACR |
| PPMI-collapse (monotone reweight) | per-slot observable γ + rank-inversion test | no inversions vs co-occ **and** PPMI |
| Phase-2 decode-noise headwind (τ0_null ≈ 99%) + decode-path already tried (F7a) | front-door term + gate discipline; reuse selector scaffolding | τ0_flip stays near the 99% floor |
| front-door unidentifiable (popularity in both paths) | mediator-validity + backdoor cross-check (POG/MealRec+) | front-door ≠ backdoor where both identify |
| causal premise unconfirmed-positive (C4 null) | treat as falsifiable hypothesis; dense-first | deconfounding adds ~0 (reproduces C4) |
| Cadence/A2G collision | front-door + `do(T)` flip + observable gate | can't beat Cadence-UACR |
| Spotify-MPD access prohibited | POG/MealRec+ fallback | blocked → drop Spotify, keep POG pillar |

## 8. Milestones (construction)
- **M0 — re-scoped (most assets inherited).** Port the BundleBench harness + baselines + diagnostics +
  DDBC checkpoints into this repo, then **add only the genuinely-missing pieces**: the **POG_dense /
  Spotify_sparse** variant splits, the **Cadence-UACR** baseline, an **HF `xhLiu/BundleConstruction`
  puller**, and packaged single-callable wrappers for shortlist-inflation + popularity-triviality. *Do
  not re-spec the inherited metric suite, significance stack, baselines, DDBC decode, or diagnostics as
  new work (F1–F9, Part C of the overlap ledger).*
- **M1 — Phase-1 front-door reranker** (the new `g=h₁·h₂` estimand on the C3 backbone). **Go/no-go gate**
  (§3.4); dense-first win expectation.
- **M2 — Phase-2 decode-path** (only if M1 passes); must clear the τ0_null ≈ 99% bar.
- **M3 — inertness battery (reused) + DieT-style long-tail slice + observable-gate stress** (dense vs
  sparse variants; MealRec+ H/L; dense controls for inertness).
- **(future)** ranking track — only after M1–M3 reach construction SOTA.

> **First implementation slice (for writing-plans):** scope the first plan to **M0 + M1 only** — port the
> inherited harness/baselines/diagnostics, add the 4 missing M0 pieces, and build the Phase-1 front-door
> reranker, ending at the go/no-go gate. M2–M3 are planned *after* M1 passes.

## 9. Reusable assets (port from BundleBench; vendored subset noted)
**Vendored in this repo (`code/reference/`):** `ranking_metrics.py`, `significance.py`, `ddbc_decode.py`,
`reranker.py`, `fusion.py`. **To port (exist in BundleBench, not here):** set_metrics; the baseline family
(popularity, co-occ, item-kNN, PPMI-SVD, content SetCompleter, SASRec); the **diagnostic suite**
(`theme_contrast.py`/`do(T)` null, `inertness.py` 3-part report, `oracle_ceiling.py`, `loadbearing_driver.py`,
`hsic_permutation.py`); the **causal stack** (`gate.py`/ObservableGate, `sdpp_selector.py`, `theme_ivae.py`,
`collab_proxy.py`) — reuse the **gate + selector scaffolding**, the theme path is refuted; DDBC checkpoints
(`models_{spotify90,pog}.pt`); dataset loaders + builders. **Do NOT re-implement the diagnostics** —
§5.4 reuses them.

## 10. Open questions
- Does front-door deconfounded co-occ beat **both** PPMI and **Cadence-UACR** — or collapse to popularity
  normalization? *Existing data (C3/F4, F8) already answers the **sparse** slice in the negative; the live
  question is the **dense** slice.* (M1.)
- Is the front-door mediator valid (no popularity leak into both paths)? (front-door-vs-backdoor on POG/MealRec+.)
- Does the front-door per-pick reweight move τ0_flip above the 99% decode-noise floor that sank the
  backdoor theme term? (M2.)
- Can the gate concentrate gains on sparse/cold slots while staying inert on dense controls? (M3.)
- Spotify-MPD obtainable under license in time? (access risk.)

## 11. References & verification confidence
**Inherited BundleBench findings (this project's prior work):** C1–C4 / F1–F10 — see §1.5;
documented in `docs/findings/00-SUMMARY.md`, the `docs/findings/2026-06-2*` stage records,
`docs/learnings/LEARNINGS-from-bundle-ranking-project.md`, `docs/baselines/ddbc-repro-spec.md`.
**Verified verbatim / from primary tables (high):** DDBC eval protocol (repo `evaluator.py`, ρ=100);
CLHE (WSDM'24, arXiv 2310.18770); Bundle-MLLM (KDD'25, arXiv 2407.11712); DieT (arXiv 2411.19107 —
title/authors/scope verified; *POG/Spotify + "Pop-to-LT" specifics NOT confirmed from the abstract*).
**Verified resolving (prior cite pass):** Cadence (2512.17733), A2G-DiffRec (2602.14706), PDA (2105.06067),
MACR (2010.15363), HCR (2205.07499), D3 (2406.14900), Causal Prompting (2403.02738), UpliftRec (2405.08582),
OPCB (2408.11202), DiffRec (2304.04971), DreamRec (2310.20453), TIGER (2305.05065). **Medium / re-verify
before camera-ready:** DDBC exact per-cell numbers (CAPTCHA-walled — re-run the repo); MealRec+ construction
reframing (our framing). **Deferred (ranking, out of scope):** CrossCBR / MultiCBR / EBRec / DCBR /
MoDiffE / CoHeat / BunCa / CLBR.
