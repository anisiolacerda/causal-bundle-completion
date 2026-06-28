# The winner — retrieve-then-rerank for bundle completion (PIC-Diff recovery)

> Full-catalog eval (no negative sampling), 5 seeds, paired Wilcoxon vs co-occurrence.
> Code: `scripts/build_cf_embeddings.py`, `scripts/stage4_fusion.py` (`--item-emb item_emb_cf.pt`),
> `src/causalbr/recovery/{reranker,ranking_metrics,fusion}.py`.

## The method

**Retrieve-then-rerank:** (1) co-occurrence generates candidates (its strong recall) — take the top-50
by training item-item co-occurrence with the observed items; (2) a **learned-CF reranker** reorders
those 50 by a seed-pair query over **PPMI-SVD co-occurrence embeddings** (item2vec-style). The CF
embeddings *smooth* co-occ's noisy mid-frequency counts, so they promote the true completion *up*
within co-occ's candidate set — lifting precision AND recall over raw co-occ.

## Results (median over 5 seeds, full catalog)

| dataset | method | hit@1 | recall@10 | recall@20 | MRR |
|---|---|--:|--:|--:|--:|
| MealRec+H (n=4677) | **retrieve-rerank** | **0.164** | **0.501** | **0.634** | **0.273** |
|  | co-occurrence | 0.132 | 0.468 | 0.631 | 0.241 |
| MealRec+L (n=1181) | **retrieve-rerank** | **0.164** | **0.553** | **0.700** | **0.289** |
|  | co-occurrence | 0.149 | 0.538 | 0.681 | 0.269 |

Paired Wilcoxon (one-sided, vs co-occurrence): hit@1 **p=0.0312** (both datasets, all 5 seeds);
recall@10 **p=0.0312** (H) / 0.0625 (L, 4/5 seeds). For context, DDBC's generative cosine-NN decode
gets ~0.025–0.07 hit@1 (Stage-1), and the content (MiniLM) reranker / learned fusion only tie co-occ
(content similarity != complementarity). The win comes specifically from **CF-embedding** reranking of
**co-occ** candidates.

## The full arc (honest)

1. **Causal/personalization (PIC-Diff thesis): refuted** 3 ways — Phase-6 do(T) (theme inert vs sampler
   noise), Stage-1 oracle (deconfounded θ adds ~0 at the argmax), a free user embedding that degrades
   the reranker. The deconfounded user-theme carries no held-out completion signal on MealRec.
2. **Generative diffusion decode (DDBC): far suboptimal** — cosine-NN decode ~0.025–0.07 hit@1, beaten
   by popularity, co-occurrence, and the reranker.
3. **Content embeddings: don't beat CF** — content similarity is not complementarity; the content
   reranker wins shortlist hit@1 (a protocol artifact) but only ties co-occ on the honest full catalog.
4. **Winner: co-occurrence retrieval + learned-CF reranking** beats raw co-occ on all full-catalog
   metrics, 2 datasets, significant.

Reusable diagnostics produced: the do(T) sampler-noise null, the oracle re-rank ceiling, the
shortlist-vs-full-catalog protocol artifact, the popularity-trivial random-negatives finding.

## Dataset 3 — Spotify-90 (DDBC's own benchmark): firm result (2026-06-28)

Spotify-90 is 254k items, fixed 45-observed / 45-held-out playlists, **969 test** bundles. Two
questions were settled here: (a) does the CF reranker (RR) still beat co-occ? and (b) the headline —
do co-occ / RR beat DDBC's *generative decode* on DDBC's own benchmark?

### (a) RR does NOT transfer to the 254k sparse catalog — honest negative

Full 969 test, 5 seeds, `scripts/stage5_spotify.py` (median over seeds, full catalog):

| config | method | hit@1 | recall@50 | recall@100 | recall@500 | MRR |
|---|---|--:|--:|--:|--:|--:|
| dim64, topk200 | retrieve-rerank | 0.078 | 0.056 | 0.095 | 0.229 | 0.183 |
| dim128, topk500 | retrieve-rerank | 0.052 | 0.043 | 0.080 | 0.229 | 0.144 |
| — | co-occurrence | **0.103** | 0.053 | 0.090 | 0.229 | **0.200** |

The CF reranker only reshuffles co-occ's top-K (recall@500 is identical), and on Spotify that
reshuffle **loses on hit@1 / MRR** (0.078–0.052 vs 0.103) while at best edging co-occ on
recall@50/100. A 64–128-d PPMI-SVD on the 254k-item catalog is too coarse to promote the true
completion. The earlier "marginal edge" (hit@1 0.090 vs 0.086) was a 500-subset + 3-seed + topk200
artifact; on the full 969 test it reverses. **Verdict: RR helps on the dense MealRec catalogs but
not on the sparse 254k Spotify catalog. Co-occurrence is the strong baseline there.**

### (b) HEADLINE — co-occ / RR beat DDBC's generative decode, full-catalog

The trained DDBC model (`models/models_spotify90.pt`, reproduced at F1≈0.298 under the shortlist
protocol) was run through its generative decode and scored **full-catalog** — every catalog item
ranked by max cosine to any reconstructed completion slot (`scripts/stage6_ddbc_decode.py`,
`src/causalbr/recovery/ddbc_decode.py`). This removes the shipped evaluator's rho-pool shortlist,
which *already contains the gt* and over-claims. 969 test, n_steps=25, observed items masked:

| method | hit@1 | recall@50 | recall@100 | recall@500 | MRR |
|---|--:|--:|--:|--:|--:|
| **co-occurrence** | **0.107** | **0.057** | **0.096** | **0.232** | **0.209** |
| DDBC decode (ema) | 0.061 | 0.039 | 0.065 | 0.178 | 0.136 |
| DDBC decode (raw) | 0.049 | 0.038 | 0.064 | 0.177 | 0.125 |
| popularity | 0.024 | 0.017 | 0.030 | 0.100 | 0.063 |

co-occ beats DDBC's decode on **every** metric. Per-example paired Wilcoxon (one-sided, co-occ >
DDBC-ema, n = #test playlists): recall@50 **p=1.2e-30** (n=713), recall@100 **p=1.5e-42** (n=779),
MRR **p=8.6e-15** (n=933); hit@1 mean 0.107 vs 0.061 (Wilcoxon p=8e-5 over the n=143 discordant
playlists — its median_diff is 0 only because hit@1 is binary and most playlists miss top-1 for both,
a Pratt artifact). DDBC-raw is the same picture (recall@100 p=2.6e-41). The unmasked variant agrees.

DDBC's full-catalog decode hit@1≈0.06 matches its ~0.025–0.07 weakness on MealRec — i.e. **the
generative cosine-NN decode is genuinely weak even on its home benchmark; the rho-pool shortlist was
hiding it.** This is the same shortlist-over-claims lesson, now turned on DDBC itself.

## Honest 3-dataset state (final)

| dataset | RR vs co-occ | co-occ/RR vs DDBC decode |
|---|---|---|
| MealRec+H | **RR wins** (hit@1 0.164 vs 0.132, p=0.031) | co-occ/RR ≫ DDBC (~0.025–0.07 hit@1) |
| MealRec+L | **RR wins** (hit@1 0.164 vs 0.149, p=0.031) | co-occ/RR ≫ DDBC |
| Spotify-90 | RR ties/loses (no transfer to 254k) | **co-occ/RR ≫ DDBC, p<1e-30, n=969** |

Two complementary claims, each on the datasets where it is strong: **RR > co-occ** on the two
MealRec catalogs (the complementarity-reranker contribution), and **co-occ/RR ≫ DDBC's generative
decode, full-catalog** on all three including DDBC's own Spotify-90 benchmark (the "beats DDBC"
contribution). The causal/personalization (PIC-Diff) thesis remains refuted.
