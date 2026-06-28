# Stage-1 Probe A — Oracle Re-rank Ceiling (findings)

> Single-backbone directional read (the 8-seed box was turned off; D2 `backbone_theta_mealrec.pt`
> on the laptop). n=2000 TEST examples, n_test=1000, ρ=100, optimistic β-on-test ceiling. CPU.
> Code: `src/causalbr/eval/oracle_ceiling.py` (tested), `scripts/stage1_probe_a.py`.

## Numbers

| metric | random negatives (DDBC protocol) | popularity-matched hard negatives |
|---|--:|--:|
| baseline — theme-blind cosine-NN decode (`evaluate.py` argmax) | 0.069 | 0.035 |
| θ-free oracle ceiling (best global additive bias) | 0.196 | 0.059 |
| θ-cond oracle ceiling (best linear-in-θ bias) | 0.199 | 0.060 |
| **θ-cond − θ-free  (the theme's marginal value)** | **+0.003** | **+0.001** |
| θ-free recon-free (bias only, recon ignored) | 0.210 | 0.059 |
| **popularity (rank pool by train frequency)** | **0.616** | 0.059 |

(best_beta saturates the grid → at the ceiling the recon is nearly ignored; the "win" is a
recon-independent prior, not an improvement of the backbone's decode.)

## Reads (robust across both negative regimes)

1. **θ is genuinely uninformative.** The best *linear-in-θ* additive bias at the decode argmax adds
   **+0.001 to +0.003** hit@1 over the θ-free oracle — i.e. nothing. W-theme (the causal/
   personalization theme) is dead at the source: the deconfounded 16-d θ carries no held-out
   item-discriminative signal. This is consistent with the Phase-6 verdict (theme inert) and the
   panel quick-probe (θ redundant with popularity), now confirmed at the argmax with an oracle ceiling.
2. **The DDBC random-negative protocol is popularity-trivial.** Ranking the ρ=100 pool by raw train
   frequency gets **0.616** hit@1 — 9× the backbone's cosine-NN decode (0.069) and 3× the embedding
   oracle (0.2). Random negatives from a long-tail catalog reduce the task to "spot the real menu
   item among random noise," which popularity solves. Under popularity-matched hard negatives,
   popularity collapses to **0.059** — confirming the 0.616 was a negative-sampling artifact.
3. **On the genuinely-hard task, everything is signal-poor.** With hard negatives, baseline 0.035,
   the *oracle ceiling* itself only 0.059 (~6× the 1/100 floor). The held-out 3rd course is
   **near-unpredictable from content / user / θ beyond popularity** on MealRec+H |gt|=1. The
   embeddings (recon + best linear bias) carry weak complementarity signal at best.

## Stage-1 verdict (pre-registered §4)

- **W-theme: NO-GO** — θ-cond − θ-free < +0.003 in both regimes. Do not relocate the current θ again.
- **W-rerank: real but weak/vacuous** — a theme-free bias beats the backbone, but under random
  negatives that is just popularity (scientifically vacuous), and under hard negatives the ceiling
  is only +0.024 over baseline and recon-independent. A popularity-aware decode reranker would beat
  DDBC's decode on MealRec+H, but it is a narrow protocol-sensitive win, not the causal/personalization thesis.
- **Deeper finding:** the benchmark/task (|gt|=1 over a long-tail catalog) lacks the signal the
  PIC-Diff thesis needs. The honest winner search must move the *signal*, not the placement.

## Caveats

- Single backbone (8-seed significance not available — box off). The θ-adds-~0 read is robust
  (consistent across both regimes); the absolute numbers are one-seed directional.
- Optimistic β-on-test (ceiling, upper bound). The θ-free/θ-cond are ceilings; a real learned model
  gets less. This makes the θ NO-GO *stronger* (even the best case adds nothing).
- Probe C (re-sampled-recon survival) not separately run — moot here since the ceiling win is
  recon-independent (recon ignored at best β), so there is no fixed-recon artifact to guard against.

## Signal-hunt update (seed-pair conditioning) — a WINNER handle found

Re-probed the oracle with a **seed-pair** feature (concat of the 2 observed course embeddings, 128-d,
PCA-64; no retrain) instead of the iVAE theta. n=1500, n_test=750.

| probe | baseline | theta-free | seed-pair (theta-cond) | popularity |
|---|--:|--:|--:|--:|
| hard negs, seed-pair | 0.025 | 0.057 | **0.189** | 0.059 |
| hard negs, seed-pair + iVAE-theta | 0.025 | 0.057 | **0.193** (theta adds +0.004) | 0.059 |
| random negs, seed-pair | 0.071 | 0.175 | **0.347** | 0.611 |

**Findings:**
1. **Complementarity signal EXISTS and is strong.** The seed-pair -> completion oracle hits 0.189
   (hard) / 0.347 (random) held-out — 7.5x the backbone's cosine-NN decode and 3x popularity (hard).
   The iVAE theta had destroyed this signal (mean-pooling + HSIC deconfounding); the raw seed pair
   preserves it. The signal was never absent from the data — it was discarded by the theme estimator.
2. **Personalization is dead, robustly.** Adding the iVAE theta on top of the seed-pair adds +0.004.
   Since the iVAE theta already conditions on (user, content), the user carries ~nothing beyond the
   content pair. A re-derived theta is not worth chasing.
3. **Popularity dominates random negs (0.611)** — the DDBC random-neg protocol is popularity-trivial;
   hard (popularity-matched) negatives are the meaningful eval.

**Route: build the complementarity-aware decode reranker (Stage 2).** A learned seed-pair -> completion
scorer (bilinear/MLP over item embeddings) that replaces/augments DDBC's generative cosine-NN decode.
Oracle ceiling ~0.19 hard / ~0.35 random vs DDBC decode 0.025/0.071. Validate vs DDBC + popularity +
a raw item-item co-occurrence baseline, multi-seed, hard+random negs, with the honest do(T)/recon
guards. Honest framing: a real beats-DDBC winner via complementarity; the causal/personalization
thesis is refuted (theta inert by oracle + do(T)).
