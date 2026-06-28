# Stage-2 — Complementarity Reranker: a validated winner (PIC-Diff recovery)

> Held-out TEST, hard (popularity-matched) + random negatives, ρ=100, |gt|=1. CPU, no GPU.
> Code: `src/causalbr/recovery/reranker.py` (tested), `scripts/stage2_reranker.py`,
> `scripts/stage2_multiseed.py`. Reuses the Phase-6 `eval/significance.paired_wilcoxon`.

## The method

A learned **seed-pair → completion** reranker: `g(concat(emb[obs0], emb[obs1])) -> query`, scored by
cosine against the ρ-pool item embeddings, trained with listwise CE over popularity-matched pools.
It replaces DDBC's theme-blind, generative cosine-NN decode argmax. Item embeddings + backbone frozen;
the reranker is ~tens of k params, CPU-trained. Motivated by the Stage-1 oracle (the completion signal
lives in the seed pair, not the deconfounded theme).

## Results

Single-config (n=2000):

| method | random negs | hard negs (meaningful) |
|---|--:|--:|
| DDBC decode (cosine-NN) | ~0.071 | ~0.025 |
| popularity | 0.588 | 0.068 |
| co-occurrence (CF) | 0.770 | 0.149 |
| **complementarity reranker** | 0.241 | **0.187** |

Multi-seed (5 seeds, hard negs, held-out n=2000), median hit@1:

| method | median | note |
|---|--:|---|
| **complementarity reranker** | **0.193** | stable 0.1875–0.1975 |
| co-occurrence (CF) | 0.148 | |
| reranker + free user embedding | 0.142 | personalization last-shot |
| popularity | 0.068 | |

Paired Wilcoxon (one-sided, across 5 seeds):
- **reranker > co-occurrence: p = 0.03125** (the n=5 floor — all 5 seeds), median ΔF1 **+0.045**.
  Statistically real win over CF.
- **reranker + user vs reranker: p = 1.0, median ΔF1 −0.0545** — a free per-user embedding **degrades**
  the reranker (overfits; ~10 examples/user). Personalization adds nothing and hurts.

## Verdict

1. **WINNER (condition 1 — beats DDBC):** the complementarity reranker is the best method on the
   meaningful hard-negative task — it beats DDBC's generative decode (~7.5×), popularity, and, crucially,
   **raw co-occurrence CF (p=0.031 across seeds)** — so it is a real learned-complementarity method, not
   just CF beating diffusion. It realizes the Stage-1 oracle ceiling (~0.19).
2. **NEGATIVE (condition 2 — personalization/causal):** decisively refuted, three independent ways —
   Phase-6 do(T) (theme inert vs sampler noise), the Stage-1 oracle (θ-cond − θ-free ≈ 0), and now a
   free user embedding that *degrades* the reranker. The PIC-Diff personalization/causal premise has no
   support on MealRec+H |gt|=1.
3. **Protocol finding:** DDBC's random-negative protocol is popularity-trivial (popularity 0.59–0.77);
   popularity-matched hard negatives are the meaningful eval, where the reranker leads.

## Honest paper shape

*A learned complementarity reranker substantially outperforms generative masked-diffusion decoding and
collaborative-filtering baselines for bundle completion under hard negatives; the personalization/causal
premise is unsupported — a correctly-deconfounded user theme is inert (do(T) + oracle), and an explicit
user embedding degrades held-out accuracy.* A winner on the benchmark axis + a rigorous causal negative,
with the do(T) sampler-noise null and the oracle-ceiling probe as reusable falsification tools.

## Caveats / next for rigor

- Single backbone for the recon/DDBC baseline (8-seed box off); the reranker multi-seed (init +
  shortlist) is solid. The DDBC-decode number is from the Stage-1 probe (same protocol).
- n=5 one-sided Wilcoxon floor is 0.03125 — significant for the single confirmatory reranker-vs-CF
  test; a larger seed count would tighten it. The effect is stable (tight per-seed spread).
- Next if pursued: scale seeds, add recall@k / full-catalog eval, and a proper DDBC end-to-end
  comparison on its reported metric; consider |gt|≥2 (denser) where complementarity may be richer.
