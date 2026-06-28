# Research kickoff — deep causal-recsys literature sweep (next session)

> This project is **scaffold only**. This doc is the executable plan for the NEXT session: a deep
> multi-agent literature sweep that produces [promising-research-lines.md](promising-research-lines.md).
> Confirmed scope: **deep multi-agent sweep** (user choice). Inputs in [seed-sources.md](seed-sources.md).

## Objective

Find and rank **promising research lines** for a new **causal-inference-based method** for ranking +
bundle construction — one that clears the bar in [../docs/findings/00-SUMMARY.md](../docs/findings/00-SUMMARY.md):
beat co-occ / retrieve-rerank **and** show a real, intervention-verified causal gain (the predecessor's
user-theme `do(seed)` placement failed the inertness bar — do not repeat it).

## How to run it (literature-review skill)

Invoke `/lit` (the `literature-review` skill) with the topic below. It runs:
Plan → Gather (`researcher` agents) → Synthesize → Cite (`verifier`) → Verify (`reviewer`) → Deliver
(`outputs/<slug>.md` + `.provenance.md`). Use **worktree/parallel researcher agents** given the scope.

Topic string:
> Causal inference methods/algorithms for recommender ranking and bundle/set construction. Seed:
> Gao et al. "Causal Inference in Recommender Systems: A Survey and Future Directions" + the two
> curated repos in seed-sources.md. Goal: rank promising research lines for a NEW causal method whose
> causal term is intervention-verifiable (passes ablation / tau@0 / gradient-flow) and beats
> co-occurrence/retrieve-rerank full-catalog, not just DDBC.

## Sweep structure (suggested agent fan-out)

1. **Seed decomposition** — deep-read the survey; extract its taxonomy + every "Future Directions" item.
2. **Repo mining (parallel)** — one agent per curated repo (FIB-lab, Chrissie-Law): extract all linked
   papers, dedup, tag by category, flag set/bundle/generative ones. Coverage diff = signal.
3. **Category deep-dives (parallel)** — one agent per causal family:
   (a) debiasing (IPW / doubly-robust / exposure / popularity / position),
   (b) deconfounding (backdoor / front-door / deconfounder / IV),
   (c) counterfactual & uplift (treatment = exposure/recommendation),
   (d) causal **generative / diffusion** recommenders,
   (e) **set/slate/bundle**-level causal estimands (likely sparse — the gap).
4. **Bundle-construction cross-map** — map each family onto bundle construction: what is the
   treatment, the outcome, the confounder, the estimand at the **set** level?
5. **Synthesize → rank lines** — fill [promising-research-lines.md](promising-research-lines.md):
   each line scored on novelty, fit-to-our-data (co-occ/exposure signal exists; user-theme does not),
   intervention-verifiability, and feasibility vs DDBC/co-occ baselines.

## Constraints (inherited — see ../README.md)

Full-catalog eval; inertness diagnostic mandatory; honesty guards; causal term decode-path or
observable-gated (not globally forced); human-authored commits. Reuse `code/reference/` assets.

## Deliverables of the next session

- `research/outputs/<slug>.md` + `.provenance.md` (the cited landscape).
- `research/promising-research-lines.md` filled and ranked.
- A short-list (2–3) of candidate method designs → hand to a `brainstorming` → `writing-plans` pass.
