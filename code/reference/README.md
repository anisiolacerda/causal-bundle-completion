# Reference code (from Bundle_construction)

**Reference only.** Copied verbatim from the predecessor's `src/causalbr/`. They import from the
`causalbr` package and will NOT run standalone here — adapt imports / lift the functions you need
when the new harness is built. Kept so the new method reuses proven, tested implementations.

| file | what it provides |
|---|---|
| `ranking_metrics.py` | full-catalog multi-gt recall@k / hit@1 / MRR + `rank_metrics_multigt_per_example` (per-example paired tests). |
| `significance.py` | exact/normal paired Wilcoxon, bootstrap CI, Bonferroni. |
| `ddbc_decode.py` | full-catalog scoring of DDBC's generative decode (max-pool slot cosine; NO shortlist). |
| `reranker.py` | the seed-pair CF reranker (the dense-catalog positive result). |
| `fusion.py` | learned logistic fusion of complementarity signals (co-occ / content / popularity). |

The matching tests live in the predecessor repo under `tests/recovery/` — port them with the code.
