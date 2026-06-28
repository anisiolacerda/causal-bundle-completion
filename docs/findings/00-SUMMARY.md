# Findings carried over from Bundle_construction (the PIC-Diff recovery)

Synthesized handoff. Full detail in the sibling docs in this folder and the `memory-snapshots/`.
Evidence labels: **FACT** = held-out, multi-seed, full-catalog; **HYPOTHESIS** = plausible, unverified.

## The honest arc (what happened, in order)

1. **PIC-Diff causal/personalization thesis — REFUTED (FACT).** Reinterpreting masked-diffusion
   completion as `do(seed)` over a deconfounded, user-conditioned bundle theme added **no** held-out
   gain. Refuted three independent ways:
   - **do(T) intervention is inert at decode** — flipping the theme moves the output set ~2% vs a
     sampler-noise null of ~99% (Phase-6 verdict).
   - **Stage-1 oracle ceiling** — even an oracle deconfounded θ adds ~0 at the decode argmax.
   - **Free user embedding degrades** the reranker rather than helping.
   - The deconfounded user-theme carries no held-out completion signal on MealRec; θ was clean
     (HSIC deconfounding p=0.156) but uninformative.
2. **DDBC's generative decode is weak full-catalog (FACT).** The cosine-NN decode gets ~0.025–0.07
   hit@1 on MealRec and ~0.06 on Spotify-90 (its OWN benchmark) once evaluated full-catalog. It
   reproduced at F1≈0.298 only under the rho-pool **shortlist** that already contains the gt.
3. **Content embeddings ≠ complementarity (FACT).** MiniLM/PCA content reranking wins SHORTLIST hit@1
   (a negative-sampling artifact) but only ties co-occurrence on the honest full-catalog metric.
4. **Winner: retrieve-then-rerank (FACT, but scope-limited).** co-occurrence candidates reordered by a
   **PPMI-SVD CF** reranker beat raw co-occ on MealRec+H/+L (hit@1 0.164 vs 0.132/0.149, p=0.031,
   5 seeds, full-catalog). **Does NOT transfer** to the 254k sparse Spotify catalog (RR ties/loses).

## The two solid contributions (predecessor stop-here state)

- **C1 — full-catalog evaluation critique.** Shortlist/sampled-neg eval over-claims for bundle
  completion; demonstrated twice (content-reranker reversal; DDBC shortlist→full-catalog collapse).
- **C2 — simple non-generative baseline ≫ SOTA generative decode** on 3 datasets incl. DDBC's own
  Spotify-90 (co-occ recall@100 0.096 vs DDBC 0.065, per-example paired Wilcoxon p=1.5e-42, n=779).
- (C3 — the reranker — is dense-catalog-only; not a standalone method contribution.)

## What this means for the NEW method

- The causal signal must come from **somewhere with real held-out predictive value** — a deconfounded
  *user-theme* did not have it on these datasets. (HYPOTHESIS: the leverage is in **item-item /
  exposure / popularity confounding**, not user-theme — the survey research must adjudicate this.)
- Place the causal term where it **cannot be bypassed at decode** (decode-path / estimand-level) OR
  gate it on an **observable** (cold-item flag, exposure imbalance), per the 3× inertness lesson in
  `../learnings/LEARNINGS-from-bundle-ranking-project.md`.
- Beating DDBC is **already done by co-occ** full-catalog — so the new method's bar is not "beat
  DDBC" (cheap) but "**beat co-occ / retrieve-rerank AND show a real, intervention-verified causal
  gain**". That is the hard, publishable target.

## Transferable assets (in `code/reference/`)

| file | use |
|---|---|
| `ranking_metrics.py` | full-catalog multi-gt recall@k / hit@1 / MRR + per-example variant (paired tests). |
| `significance.py` | exact/normal paired Wilcoxon, bootstrap CI, Bonferroni. |
| `ddbc_decode.py` | full-catalog scoring of DDBC's generative decode (no shortlist). |
| `reranker.py`, `fusion.py` | the CF reranker + learned-fusion combiner. |

Reusable **diagnostics** (re-implement in the new harness): do(T) sampler-noise null; oracle re-rank
ceiling; shortlist-vs-full-catalog artifact; popularity-trivial random-negatives check.
