# Promising research lines (SCAFFOLD — fill in the next session)

> Empty template. The deep sweep ([00-RESEARCH-KICKOFF.md](00-RESEARCH-KICKOFF.md)) populates and
> ranks these. Pre-seeded with HYPOTHESES from the predecessor's findings — treat as priors to
> confirm/refute against the literature, NOT as conclusions.

## Scoring rubric (per line)

| criterion | meaning |
|---|---|
| **Novelty** | not already done in the causal-recsys / bundle literature. |
| **Fit-to-data** | leverages signal we KNOW exists (co-occ / exposure / popularity), not user-theme (refuted). |
| **Verifiability** | the causal term can pass the inertness diagnostic (ablation / tau@0 / gradient-flow). |
| **Feasibility** | tractable vs DDBC + co-occ/retrieve-rerank baselines on our 3 datasets. |
| **Bar-clearing** | plausibly beats co-occ/retrieve-rerank full-catalog (not just DDBC). |

## Candidate lines (pre-seeded hypotheses — to validate against the survey)

### L1 — Exposure/popularity deconfounding at the SET level (HYPOTHESIS)
Treatment = item exposure; confounder = popularity/exposure imbalance that inflates co-occurrence.
Estimand defined over the **constructed set**, not a single item. Co-occ won because it captures
real complementarity but is *confounded by popularity* (popularity is a non-trivial baseline) —
deconfounding co-occ at the set level could beat raw co-occ. *Why promising:* attacks the exact
signal that worked, where a confounder demonstrably exists.

### L2 — Front-door / decode-path causal objective for generative construction (HYPOTHESIS)
Place the causal term on the **decode path** (set-level front-door objective) so it cannot be bypassed
structurally — the predecessor's lesson for avoiding inertness. Pairs a generative decoder (DDBC-style)
with a causal decode criterion. *Why promising:* fixes DDBC's weak full-catalog decode AND embeds an
unavoidable causal term.

### L3 — Counterfactual/uplift reranking (treatment = adding item j to the partial bundle) (HYPOTHESIS)
Rerank candidates by the **uplift** of adding j to the seed set (counterfactual "completion lift"),
not raw co-occurrence. *Why promising:* a set-conditional causal effect; directly a reranker (our
working positive result) but with a causal estimand.

### L4 — Observable-gated causal term (cold-item / exposure-imbalance gate) (HYPOTHESIS)
Activate the causal correction only where it helps (cold items, high exposure imbalance), per the
gate-g resolution. *Why promising:* sidesteps the "globally forced term hurts clean accuracy 3×" trap.

### L5 — [open] from the survey's Future Directions
To be filled from the seed survey's open-problems section.

---

## Ranked shortlist (fill after the sweep)

| rank | line | novelty | fit | verifiability | feasibility | bar | notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | (populate next session) |
