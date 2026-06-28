# causal-bundle-ranking

A **Method paper** project: design a new **causal-inference-based method** for **ranking and bundle
construction** that (a) beats DDBC (Tu et al., ICLR 2026) on its own benchmark and (b) adds a
statistically real, intervention-verified causal/personalization gain.

This repository is the spun-off successor to `Bundle_construction` (the PIC-Diff recovery work).
It carries forward that project's **findings, learnings, reusable assets, and reference baselines**,
and stages the **literature research** that will seed the new method's design.

> Status: **scaffold only.** The actual research + method-design tasks run in a later session.
> See [research/00-RESEARCH-KICKOFF.md](research/00-RESEARCH-KICKOFF.md).

## Why a fresh project

The predecessor established a rigorous **negative** result for the original causal thesis (PIC-Diff:
deconfounded user-theme + `do(seed)`), and a **positive** non-causal result (a complementarity
reranker + the full-catalog evaluation critique). See [docs/findings/00-SUMMARY.md](docs/findings/00-SUMMARY.md).
The new method must be **causal by construction** in a way the predecessor's placement was not —
which requires grounding in the causal-recsys literature first.

## Layout

| path | contents |
|---|---|
| `docs/findings/` | What worked / failed / was refuted in the predecessor (transferred + synthesized). |
| `docs/findings/memory-snapshots/` | Project-memory snapshots (the durable context). |
| `docs/learnings/` | The bundle-ranking-project learnings (3× inertness history, gate-g lesson). |
| `docs/baselines/` | DDBC reproduction spec. |
| `refs/` | Domain PDFs (bundle construction + recommendation). Causal-recsys papers added during research. |
| `research/` | Literature-review seed + plan + the **promising-research-lines** deliverable. |
| `code/reference/` | Reusable modules from the predecessor (metrics, significance, ddbc decode, reranker). **Reference only** — adapt imports before use. |

## Non-negotiables carried over (from predecessor CLAUDE.md)

- **Full-catalog eval** for bundle completion (no negative sampling — shortlist eval over-claims, proven twice).
- **Inertness diagnostic** on any causal term: ablation delta, tau@0 / set-correlation under ablation,
  gradient-flow magnitude. "It's in the architecture" ≠ "it's exercised at inference."
- **Honesty guards:** multi-seed, paired Wilcoxon, pre-registered thresholds, report negatives plainly.
- **Causal term placement:** prefer conditionally load-bearing (observable gate) or decode-path /
  estimand-level placement; do NOT force globally load-bearing (it displaced clean accuracy 3× in the
  sibling ranking project).
- Commits human-authored only (no AI attribution).
