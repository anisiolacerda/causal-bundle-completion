# Learnings from the Bundle-Ranking Project

> **Source project:** `Bundle_recsys` (bundle *ranking* — CausalBR / DCBR / DPID). These are
> point-in-time learnings — verify file:line citations against that repo before relying on them.
>
> **Updated 2026-06-23 with findings that postdate the original transfer.**

This document carries everything from the bundle-*ranking* project that is useful for bundle
*construction* (PIC-Diff). It is organized as nine load-bearing lessons. Read it before touching any
method code.

---

## 1. The Load-Bearing Imperative

**Causal and auxiliary terms went inert — repeatedly.** Across the ranking project, every causal or
auxiliary mechanism that was added to the model was bypassed at inference: the model learned to
minimize the clean reconstruction loss through cheaper pathways and stopped exercising the causal
term. This happened **three separate times**, confirmed across ablations, gradient-flow checks, and
`tau@0` rank-correlation diagnostics:

1. The diffusion-side causal term `C_l` went inert while the text-MLP pathway carried the full
   signal.
2. The dual-proxy blend (DPID) added a text proxy `C_o`; the model routed through it and the
   collaborative proxy `C_u` contributed nothing on NetEase and iFashion.
3. Even a "bypass-proof" pre-dot-blend routing fix was designed and deployed — and still failed:
   the model steered the projection orthogonal to user embeddings, draining `C_u`'s contribution
   while still minimizing loss.

**The root cause is structural, not incidental.** Whenever a causal term is optional — i.e., the
loss can be minimized without it — a sufficiently expressive model will find a way to ignore it.
"It is in the architecture" is not evidence it is exercised at inference; that must be verified.

**Inertness diagnostic (KEEP THIS — it worked every time):** verify causal terms are active
with a three-part CI gate, enforced from the first commit:

- **Ablation Δ** — removing the causal term must move the primary metric by a detectable amount.
- **`tau@0` / set-correlation** — Kendall's tau (or set overlap) of ranked completions under
  the full model vs. the ablated model must be below a threshold, confirming different behavior.
- **Gradient-flow magnitude** — the causal pathway must carry non-negligible gradient at convergence
  (log this every checkpoint).

Do not ship a method paper claiming causal identification without all three checks passing.

---

### The Load-Bearing Reframe (2026-06-23 correction — supersedes the original mandate)

**The original mandate above ("design so the objective *cannot* be minimized without the causal
term") was subsequently REFUTED by the ranking project's own experiments. The corrected picture:**

**FACT (3× refutation of forced-global load-bearing):** Forcing the identified/causal term
load-bearing globally — on the score at every forward pass — DISPLACED clean accuracy in three
independent experiments in the ranking project:

1. Plan-5 drift-consistency: forcing the causal term load-bearing hurt absolute Recall@20.
2. MDSD routing campaign: same pattern across datasets; `concat` (which can down-weight the term)
   won absolute Recall@20 every time.
3. Drift-constraint Approach A: a newly trained encoder `E_b` (a different mechanism from Plan-5)
   still displaced clean accuracy when forced globally load-bearing.

Slogan: **"Load-bearing ≠ better."** The `concat` baseline — which is free to *down-weight* the
identified term — outperformed forced-load-bearing designs all three times.

**FACT (the working resolution — gate-g):** What worked is **drift-SELECTIVE / GATED identified
robustness**. "gate-g" is a single model that gates an identified `[C_o, C_u]` diffusion
reconstruction on an OBSERVABLE drift magnitude ρ̂ via a monotone gate `g(ρ̂)`: off when the
composition is clean, on under drift. Validated results (3 seeds, mealrec_h):

- Beats tuned-RDiffBR (the must-beat drift-robust rival) at EVERY drift level (Δ +0.09→+0.13).
- Recovers full clean recall (≈concat 0.40 — no accuracy displacement).
- Pareto-dominates both `concat` and the 2-model PoC baseline.
- Generalizes to a third dataset (mealrec_l, 3 seeds, clean Recall@20 0.4195, monotone drift
  curve, full clean recovery at ρ=5).

Why it works: the `[C_o, C_u]` reconstruction is composition-INDEPENDENT (`C_u` rides the
untouched user–bundle graph). The gate activates the identified term only when the cost of doing
so (slight clean-accuracy penalty) is justified by the drift benefit. Bounded claim: wins in the
moderate-drift → clean regime; extreme tail (≥70% edge-churn) belongs to a drift-specialized
diffusion model.

**The refined lesson (FACT for ranking; CAUTION for construction):**

- KEEP the inertness diagnostic (ablation Δ, τ@0/set-correlation, gradient-flow). It correctly
  detected inertness all three times.
- The *fix* for inertness is NOT forcing the term load-bearing globally — that displaces clean
  accuracy.
- The *correct fix* is to make the term **conditionally load-bearing via an observable gate** (on
  only when needed), OR to move the term onto the **decode / estimand critical path** (e.g., a
  set-level front-door objective that structurally requires the causal term at decode time).

**HYPOTHESIS (transfer to construction):** The displacement was demonstrated for RANKING
(dot-product scoring). For CONSTRUCTION (set-completion likelihood) this is a CAUTION, not a
proven law. PIC-Diff should treat "force the `do(seed)`/causal term load-bearing globally" as a
hypothesis to test, and PREFER a conditional/observable-gated or estimand-level design so the
causal term does not displace clean construction accuracy.

---

## 2. Proxy Substitutability

The ranking project used two composition-stable signals as proxies for the bundle theme confounder:

- **`C_l`** — the diffusion-side latent (learned from the masked-diffusion trajectory).
- **`C_o`** — the text-revealed proxy (item-text embeddings aggregated over the observed bundle).

Both are computed from the *observed* bundle composition (not the user), so they share the same
information channel. When both are present, the model picks the cheapest one and routes around
the other — a textbook substitutability collapse. Robustness to dropout of one proxy does not mean
both are load-bearing; it means the surviving one is sufficient and the other is redundant.

**Lesson for PIC-Diff:** do not build a causal proxy that is substitutable with an existing
information pathway in the backbone. If `C_o` (text) is already carried by the item tokenizer or
the diffusion encoder, adding it as a "proxy" adds no identification power — the model already has
that channel. The second proxy (`C_u`, collaborative) must bring genuinely orthogonal information
(graph-structural, not text-derivable) to be non-substitutable.

---

## 3. The Dual-Proxy Identification Framework (DPID)

The DPID framework was developed in the ranking project to identify the latent bundle theme
confounder using two proxies with non-overlapping sources of variation:

- **`C_o`** — text-revealed proxy: aggregate of item text/category embeddings in the observed
  bundle. Observable at inference without user data.
- **`C_u`** — collaborative proxy: the top-K left singular vectors of the sparse user–bundle
  interaction matrix `W_ub`, computed via **ARPACK `svds`** (see §7 for the engineering note).
  Captures which users co-adopt which bundles — a signal orthogonal to item text.

Two alternative identification routes were investigated:

- **Proximal-CI** (Miao et al., 2018 / Tchetgen Tchetgen et al.): treat `C_o` and `C_u` as
  treatment-side and outcome-side proxies of the unobserved confounder; derive a moment condition
  that non-parametrically identifies the causal effect.
- **Multi-proxy backdoor** (generalized backdoor adjustment via multiple noisy proxies): average
  over proxy uncertainty in a Bayesian or empirical-Bayes fashion.

**Reusability for construction:** the DPID scaffold — `C_o` as text proxy, `C_u` as sparse-SVD
collaborative proxy — ports directly to the bundle-construction setting. The theme confounder in
construction is the latent "bundle concept" that drives spurious co-occurrence (popularity /
exposure bias). DPID is the identification engine for deconfounding completions.

---

## 4. The Spectral-Energy Operating-Regime Criterion

**This is the surviving paper-able asset from the ranking project.**

**The claim:** the collaborative proxy `C_u` confers ID-only drift-robustness *if and only if*
the user–bundle graph `W_ub` is low-rank / energy-concentrated. When the graph is diffuse
(high-rank, spread spectral energy), `C_u`'s singular vectors are noisy and provide no useful
identification signal.

**The operative predictor:** top-64 spectral-energy concentration — the fraction of total Frobenius
energy captured by the top-64 singular values of `W_ub`. This outperformed all alternatives tested.

Specifically: **split-half R² was refuted on iFashion** — a dataset where R² appeared adequate
but the regime criterion correctly predicted no payoff. The spectral-energy concentration metric
survived all three validation points.

**Validated on n=3 datasets:**

| Dataset | Top-64 spectral energy | C_u helpful? | Payoff |
|---------|------------------------|--------------|--------|
| Youshu | 39.5% | Yes | retention +0.19 Recall@20 |
| NetEase | 10.1% | No | no payoff |
| iFashion | 7.2% | No | no payoff (refuted split-half R²) |

**Threshold:** somewhere in the open interval (10.1%, 39.5%). No third data point yet pins it
further.

**Use as a cheap pre-GPU screen:** before spending any GPU time on a new dataset, compute the
top-64 spectral-energy concentration of `W_ub`. If it falls below ~10%, `C_u` almost certainly
won't help on that dataset. See `scripts/diagnose_cu_spectrum.py` in the ranking project for the
reference implementation.

---

## 5. Forced-Global Load-Bearing Refuted 3× — Drift-Selective Gated Robustness CONFIRMED

**FACT:** Forcing the identified/causal term globally load-bearing displaced clean accuracy in
three independent ranking-project experiments:

1. **Plan-5 drift-consistency** — forced `C_ℓ`/diffusion term load-bearing hurt absolute Recall@20.
2. **MDSD routing campaign** — forced routing through the identified blend; `concat` won every time.
3. **Drift-constraint Approach A** — newly trained `E_b` encoder, different mechanism, same outcome.

In all three cases the `concat` baseline — free to down-weight the causal term — won on absolute
Recall@20. Slogan: **"Load-bearing ≠ better."**

**FACT:** Drift-SELECTIVE GATED identified robustness is CONFIRMED. gate-g (single model,
observable drift ρ̂, monotone gate `g(ρ̂)`) beats tuned-RDiffBR at EVERY drift level
(Δ +0.09→+0.13, mealrec_h, 3 seeds), recovers full clean recall, Pareto-dominates `concat` and
the 2-model PoC, and generalizes to mealrec_l (3rd dataset, 3 seeds). See §1 for the full
gate-g record.

**The original failure mode:** the earlier "drift-as-intervention" framing relabeled an existing
generative knob without changing the estimand or objective. The model had no reason to behave
differently.

**Lesson for PIC-Diff:** a causal interpretation must do at least one of these:

1. **Change the estimand** — the model must output a different quantity under the causal
   interpretation than under the purely predictive one (e.g., completions conditioned on
   `do(seed)` rather than `P(completion | seed)`).
2. **Change the objective** — the loss must penalize confounded completions (e.g., via an
   identification moment condition or an adversarial deconfounding term).
3. **Carry a falsifiable win condition defined up front** — a specific metric, dataset, and
   threshold where the causal interpretation predicts a gain that a purely predictive model
   cannot match (e.g., OOD/cold-item completion accuracy).

If none of these three conditions hold, the causal framing is cosmetic. Do not submit a paper
without at least conditions 1 and 3.

---

## 6. Dataset Intelligence

### Two Degeneracy Axes

Bundle datasets have two orthogonal failure modes:

1. **Low bundles/user** — Clothing (~1–2 bundles/user), Steam (~1–2) — too little per-user
   history for personalization.
2. **Low users/bundle** — personal-playlist data (AotM-2011, 30Music) — ~1 user/bundle, so the
   user×bundle co-adoption matrix `W_ub` is essentially diagonal. C_u (SVD of `W_ub`) provides
   no collaborative signal; all identification must come from user history and C_o text.

Shared-bundle data (Youshu / NetEase / iFashion, 10–60 users/bundle) is what makes `C_u`
informative. Personal playlists are the mirror image.

### MPD / Spotify

Playlist-completion benchmark. **No user–bundle interaction matrix** — tracks belong to playlists
but there is no adoption/purchase signal that would constitute `W_ub`. This means:

- Unusable for personalized bundle construction (no user signal).
- Unusable for `C_u` computation (no user–bundle graph).
- **Was DDBC's home turf** — DDBC's Spotify-k results (k ∈ {30, 60, 90}) are the natural
  control-track benchmark.

**CRITICAL CONTROL-TRACK RISK (FACT):** Spotify-MPD **has no user ids and is no longer
downloadable** from its original source. The control track "reproduce DDBC on Spotify" may be
infeasible. AotM-2011 and 30Music are the downloadable personal-playlist substitutes for the
construction control track. Confirmed negatives (do not re-research): Goodreads released data
has no real bundles (shelves = status labels; Listopia not released); Amazon/Steam bundles are
degenerate for construction.

### AotM-2011 (personal-playlist dataset — FACT: built and registered)

- 15,361 users / 95,262 bundles / 119,459 items; 6.2 bundles/user.
- Built, converted, and registered in the ranking project: `name: aotm`, config
  `configs/dataset/aotm.yaml`, builder `scripts/build_aotm2011_dataset.py`, ρ-drift masks at
  seed 42.
- Personal-playlist regime: ~1 user/bundle. `C_u` will not help (low shared-adoption signal).
  Useful as a construction control track (personalization via user history, not collaborative SVD).
- Download: public, scriptable (no manual browser step required).

### 30Music (personal-playlist dataset — FACT: registered, GPU-heavy)

- 14,883 users / 46,990 bundles / **465,015 items** (≈4× NetEase); 3.2 bundles/user.
- `name: thirtymusic`. Raw data lives on an external drive; item-side memory footprint requires
  **≥24GB GPU + chunked-InfoNCE** (same fix as NetEase — port from ranking project). Training on
  Vast.ai only; do not attempt locally.
- Personal-playlist regime: same ~1 user/bundle caveat as AotM.
- Download: requires manual browser step (not scriptable from a headless server).

### Personalization tension (playlist datasets)

Personal-playlist data is ~1 user/bundle → little/no cross-user adoption signal and a
**sparse-collaborative stress test for C_u**. This is exactly the "construction without
shared-user signal" regime: the collaborative proxy carries no information and all personalization
must come from user history. This sharpens the existing tension noted in §3 (proxy substitutability):
do not treat C_u as informative on AotM or 30Music without first running the spectral-energy screen.

### NetEase

- **~0.04 recall@20 ceiling**, architectural / catalog-scale, robust across 6 experiments.
- Not fixable by DDCL regime, neg-hardness tuning, or learning-rate search.
- The ceiling is caused by catalog scale and sparse adoption — a structural problem, not a
  training instability.
- Top-64 spectral energy = 10.1% → `C_u` won't help here.
- **Compute-bound** on the ranking project's hardware; use `NJOBS=1` to avoid OOM.

### iFashion

- Feature-rich and multimodal (item images + text), **with users**.
- Has a user–bundle interaction graph, so `C_u` is computable.
- Top-64 spectral energy = 7.2% → `C_u` won't help (refuted the split-half R² criterion here).
- Valuable for the primary personalization track despite the low spectral energy; the
  personalization gain must come from user history rather than collaborative structure.

### Steam

- Degenerate: **~460 bundles** total. Positive-control only — useful for sanity-checking that
  a model can learn anything at all, not for publishable personalization results.

### Youshu

- Top-64 spectral energy = 39.5% → `C_u` helps (retention +0.19 Recall@20 confirmed).
- The most reliable user-bearing dataset for the primary track.

### MealRec+

- Distributed via Git LFS. Pull via `git lfs pull` after cloning.
- Has user interaction data; suitable for the primary personalization track.

### Data transport

- **Laptop ↔ Vast.ai resets files >1MB** — do not SCP large raw datasets directly.
- Pull data via git from established repos:
  - CrossCBR-style data + Fold masks: `mysbupt/CrossCBR` and `WUT-IDEA/RDiffBR`.
  - MealRec: clone with `git lfs`.
- DDBC's Spotify data: follow DDBC's own data download instructions; do not assume it is
  pre-processed in the CrossCBR format.

---

## 7. Engineering and Infra

**Chunked InfoNCE for DDCL.** When running the DDCL contrastive term on large catalogs, compute
InfoNCE in blocks rather than over the full batch. Memory complexity drops from O(B²) to O(block·B).
This is already implemented in the ranking project — port it.

**In-batch DDCL negatives for large datasets.** On iFashion-scale datasets, switch from
full-catalog negatives to in-batch negatives (`ddcl_inbatch_negatives=true`). This reduced training
time from 54 min/epoch to 2.5 min/epoch on iFashion in the ranking project — a 20× speedup with
negligible metric impact.

**ARPACK `svds` for `C_u`.** Use `scipy.sparse.linalg.svds` (ARPACK backend) for the sparse SVD
of `W_ub`. PROPACK was tested and is approximately 100× slower for the matrix sizes in this
project. **Do not reintroduce PROPACK.**

**Capacity-matched proxy comparisons.** When comparing two proxies or ablation variants, match
parameter counts. Giving one variant more capacity and declaring it better is not a valid ablation.

**Verdict gates.** A result is accepted as a win only if it satisfies BOTH:
1. The metric improvement is within the noise band of repeated seeds (clean-within-noise).
2. Retention beats the baseline (the new model does not degrade any existing metric while gaining
   on the target).

**PyTorch compatibility.** `torch.load` with `weights_only` default changed in torch≥2.6 —
explicitly pass `weights_only=False` when loading older checkpoints that contain non-tensor objects.

**Vast.ai infra.** Target GPU: RTX 3090 (24GB VRAM). Python 3.12 + venv (not conda). The ranking
project used a venv at `~/venv`; port this pattern. When local disk fills, push GPU artifacts to
external drive before the instance is destroyed.

---

## 8. Process

**Payoff-gated feasibility tiers (GREEN / AMBER / RED).** Before any GPU spend, run a cheap CPU
screen on the candidate dataset or idea. Classify:
- **GREEN** — clear expected payoff, proceed to GPU.
- **AMBER** — ambiguous; design a minimal GPU probe (one seed, one dataset) before full run.
- **RED** — no expected payoff; do not proceed.

The spectral-energy screen (§4) is the canonical example: a 30-second CPU computation that
predicts whether GPU spend on `C_u` is worthwhile.

**Adversarial multi-agent panels.** When a design decision has real trade-offs (e.g., proximal-CI
vs multi-proxy backdoor for identification; RVQ-level vs full-sequence masking), convene an
adversarial panel: one agent argues each side with citations, a judge agent rules. This settled
several ranking-project design forks efficiently.

**Lead-line + characterized-boundary reframe pattern.** When a universal claim is refuted (e.g.,
"`C_u` always helps"), do not discard the idea — reframe it to a characterized operating regime
with a predictive diagnostic (e.g., "spectral-energy concentration predicts when `C_u` helps").
The reframe must be falsifiable and must predict on held-out data.

**Subagent-driven TDD.** Write the test first (specifying the load-bearing CI gate), then write
the implementation. This catches inert-term issues at commit time rather than at paper-review time.

---

## 9. What DDBC Gives Us to Build On

DDBC (Tu et al., ICLR 2026) is the primary baseline and the structural foundation for PIC-Diff.
Key assets:

**RVQ tokenization.** Items are tokenized into L=4 levels, C=128 codes per level, via residual
vector quantization. This compresses large catalogs into a discrete token sequence of manageable
length. The tokenizer is pre-trained (or fine-tuned) separately from the diffusion model — port
the tokenization pipeline as-is before modifying anything.

**Masked-diffusion clamp/generate split.** DDBC's core is an absorbing-mask discrete diffusion
model (building on MDLM, Sahoo et al., 2024). At inference, *observed* seed items are clamped
(their tokens are held fixed) and *missing* items are generated (their tokens start masked and are
iteratively unmasked). This clamp/generate split is **the natural structural carrier for the
`do(item j in bundle)` / `do(seed)` intervention**: the clamped items are treated as an exogenous intervention on the
bundle, not merely as conditioning. PIC-Diff makes this causal interpretation explicit and
exploits it for identification.

**Per-level token-validation constraints.** DDBC enforces that generated token sequences satisfy
per-level validity constraints (each generated token must be a valid code at that RVQ level). These
constraints must be preserved in PIC-Diff's modified diffusion process.

**Assumptions to challenge (open design questions for PIC-Diff):**

- *Fixed bundle length k at inference.* DDBC fixes k when generating; it cannot adapt the number
  of completions per user or per seed composition. Personalization may require variable k.
- *Approximate, not exact, order-insensitivity.* DDBC's permutation-invariance is approximate
  (achieved via masking order randomization, not a theoretically exact symmetry). If exact
  order-insensitivity matters for the causal estimand, this needs to be addressed.
- *RVQ hierarchical residual assumption.* The RVQ tokenizer assumes that item embeddings have a
  hierarchical residual structure (level 1 captures coarse semantics, level 2 captures finer
  residuals, etc.). This assumption may not hold for all item modalities (e.g., food in MealRec,
  clothing in iFashion). Validate it before the final method commit.
