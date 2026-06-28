---
name: recovery-complementarity-winner
description: Validated winner = retrieve-then-rerank (co-occ candidates + learned-CF PPMI-SVD rerank) beats raw co-occ on ALL full-catalog metrics on MealRec+H & +L; causal/personalization refuted.
metadata: 
  node_type: memory
  type: project
  originSessionId: dbf70449-0dce-4226-b882-36c1ddf37645
---

Post-Phase-6-REJECT recovery. All CPU on the laptop (GPU box off). Per [[push-to-winner-method]] +
[[winner-needs-three-datasets]]. Results: `docs/superpowers/2026-06-28-winner-retrieve-rerank.md`.

**THE WINNER (validated on 2 datasets, full-catalog, 5 seeds):** retrieve-then-rerank —
(1) co-occurrence generates top-50 candidates (recall); (2) a seed-pair reranker over **PPMI-SVD
co-occurrence embeddings** (item2vec, `scripts/build_cf_embeddings.py`) reorders them (precision). The
CF embeddings smooth co-occ's noisy counts → promote the true completion within co-occ's candidates →
beat raw co-occ on EVERY metric. MealRec+H: hit@1 0.164 vs 0.132 (p=0.031), recall@10 0.501 vs 0.468
(p=0.031). MealRec+L: hit@1 0.164 vs 0.149 (p=0.031), recall@10 0.553 vs 0.538 (p=0.0625, 4/5).
Code: `scripts/stage4_fusion.py --item-emb item_emb_cf.pt` (retrieve-rerank built in);
`src/causalbr/recovery/{reranker,ranking_metrics,fusion}.py` (tested).

**Honest arc (what failed, in order):**
- Causal/personalization (PIC-Diff thesis) REFUTED 3 ways: Phase-6 do(T) (theme inert vs sampler
  noise), Stage-1 oracle (deconfounded θ adds ~0 at the decode argmax), free user embedding DEGRADES
  the reranker. The deconfounded user-theme has no held-out completion signal on MealRec.
- DDBC generative cosine-NN decode is far suboptimal (~0.025–0.07 hit@1; beaten by popularity/co-occ).
- Content (MiniLM PCA-64) embeddings DON'T beat CF: content similarity != complementarity. The content
  reranker wins SHORTLIST hit@1 (a negative-sampling protocol artifact) but only ties co-occ on the
  honest FULL-CATALOG metric. Learned fusion + content retrieve-rerank also only tie co-occ.
- Only **CF-embedding** (not content) reranking of **co-occ** candidates wins outright.

**KEY METHODOLOGICAL LESSON:** shortlist eval (ρ=100 sampled negs) is misleading — random negs are
popularity-trivial (popularity hit@1 0.6), and the content reranker's shortlist "win" reversed on the
honest full-catalog eval. Always eval full-catalog (no neg sampling) for bundle completion. Reusable
diagnostics: do(T) sampler-noise null, oracle re-rank ceiling, shortlist-vs-full-catalog artifact.

**3-DATASET STATUS: FIRMED UP (2026-06-28, commit 6c6945b).** Two complementary claims, each on the
datasets where it's strong:
- **RR > co-occ** on MealRec+H (hit@1 0.164 vs 0.132, p=0.031) and MealRec+L (0.164 vs 0.149,
  p=0.031) — the complementarity-reranker contribution.
- **co-occ/RR ≫ DDBC's generative decode (full-catalog) on ALL three datasets incl. DDBC's own
  Spotify-90** — the "beats DDBC on its own benchmark" contribution. Spotify-90 (969 test, n_steps=25,
  masked): co-occ hit@1 0.107 / recall@100 0.096 / MRR 0.209 vs DDBC-ema 0.061 / 0.065 / 0.136.
  Per-example paired Wilcoxon co-occ>DDBC: recall@100 **p=1.5e-42** (n=779), recall@50 p=1.2e-30, MRR
  p=8.6e-15. DDBC-raw same. DDBC full-catalog hit@1≈0.06 matches its MealRec weakness — the shipped
  rho-pool shortlist (which already contains the gt) was hiding it. Same shortlist-over-claims lesson,
  now turned on DDBC itself. Code: `scripts/stage6_ddbc_decode.py` +
  `src/causalbr/recovery/ddbc_decode.py` (full-catalog max-pool slot scoring, NO shortlist) +
  `rank_metrics_multigt_per_example` (n=969 paired test). Model load: guarded `strict=False`
  (`models_spotify90.pt` predates learnable-gain LayerNorm 4a41477; ones-init reproduces it exactly).

**Spotify RR does NOT beat co-occ (honest negative).** Full 969 test, 5 seeds: RR loses on hit@1/MRR
(dim64/topk200 0.078, dim128/topk500 0.052, vs co-occ 0.103), recall@500 ties (RR only reshuffles
co-occ's top-K). The CF reranker doesn't transfer to the 254k sparse catalog — a 64–128-d PPMI-SVD is
too coarse. The earlier "marginal edge" (0.090 vs 0.086) was a 500-subset/3-seed/topk200 artifact. So
the Spotify claim rides on the headline (co-occ/RR ≫ DDBC), NOT on RR>co-occ. Spotify-90 data on
laptop `data/spotify90/` (clhe.pt 254155x64, train_90.txt 188k bundles avg 90 items, fixed 45-obs/45-gt,
969 test, tokens.pt). Laptop = 8.6GB RAM, ~0.6GB free → co-occ at full 188k bundles OOMs; train-limit
20000 is the safe profile. Repro spec `docs/baselines/ddbc-repro-spec.md`.

MealRec H/L construction data built via `build_mealrec_construction_splits.py` +
`build_mealrec_item_embeddings.py` (MealRecH loader works for H and L; raw roots
`~/code/Bundle_recsys/baselines/MealRecPlus/MealRec+/{MealRec+H,MealRec+L}`). CF embeddings per dataset:
`scripts/build_cf_embeddings.py --data-dir data/<ds>` → item_emb_cf.pt. The winner run:
`scripts/stage4_fusion.py --data-dir data/<ds> --item-emb item_emb_cf.pt --seeds 5` (retrieve-rerank
built in). Latest commit on main: 6c6945b (Spotify-90 headline + RR firm-up).
