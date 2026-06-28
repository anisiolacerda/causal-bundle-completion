---
name: winner-needs-three-datasets
description: "The winner bar is ~3 datasets — more than one, not necessarily all. Validate the method on 3."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dbf70449-0dce-4226-b882-36c1ddf37645
---

Stated 2026-06-27 (superseding an earlier "one dataset suffices"): **"We need more than one dataset,
3 is a good number."** A convincing winner must be demonstrated on **about three** datasets/settings —
not just one, but not required to win on *all*.

**Why:** one result is too thin to be publishable; ~3 establishes the method generalizes without
demanding universal coverage.

**How to apply:** validate the complementarity reranker ([[recovery-complementarity-winner]]) on ~3
bundle-completion datasets. MealRec+H is one (winner shown). Candidates for the other two: MealRec+L
and BundleRec (sibling repo `~/code/Bundle_recsys/baselines/`), and/or DDBC's own benchmark
(Spotify-90, where DDBC was reproduced at F1 0.298 — the strongest "beats DDBC on its own benchmark"
claim). Reuse the same harness (full-catalog hit@1 / recall@k / MRR, hard negatives, multi-seed,
reranker vs DDBC-decode + co-occurrence + popularity). Keep the honesty guards on each dataset.
