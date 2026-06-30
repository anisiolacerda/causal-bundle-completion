# Plan — causal-recsys-bundle-construction lit sweep

Slug: `causal-recsys-bundle-construction`. Run date: 2026-06-28.
Workflow: `/lit` (Plan → Gather → Synthesize → Cite → Verify → Deliver). Orchestrated via Workflow tool
(ultracode on) — parallel researcher agents, file-based handoff, adversarial verify.

## Objective

Rank **promising research lines** for a NEW causal method for ranking + bundle/set construction that
clears the bar in `docs/findings/00-SUMMARY.md`: **beat co-occ / retrieve-rerank full-catalog AND show a
real, intervention-verified causal gain** (predecessor's user-theme `do(seed)` failed the inertness bar —
do not repeat). Special focus: **set/slate/bundle-level causal estimands** and **causal generative/diffusion**
recommenders (the hypothesized gap).

## Key questions

- Q1. What is the survey's (Gao 2208.12397) full taxonomy, and every "Future Directions" open problem?
- Q2. Full deduped citation graph of both curated repos, tagged by family; coverage diff = signal.
- Q3. Per family — debiasing (IPW/DR/exposure/popularity/position), deconfounding (backdoor/front-door/
  deconfounder/IV), counterfactual+uplift (treatment = exposure/recommendation) — what is the estimand,
  where is the causal term placed, is it intervention-verifiable, does it act pointwise or set-level?
- Q4. Causal **generative / diffusion** recommenders: do any exist? Where is the causal term in the decoder?
- Q5. **Set/slate/bundle-level** causal estimands (incl. slate off-policy / list-wise bandits, PBM/cascade):
  confirm sparsity; capture the few that exist. THE GAP.
- Q6. Cross-map: for each family, define treatment / outcome / confounder / estimand at the **set** level for
  bundle construction. What is verifiable + fits our data (co-occ/exposure signal exists; user-theme doesn't)?
- Q7. Rank lines on novelty / fit-to-data / verifiability / feasibility / bar-clearing.

## Source types

- Primary: arXiv (2208.12397 + chased citations), ACM DL / proceedings, the two curated repos.
- Web search for: causal generative/diffusion recsys, slate/list-wise off-policy, causal bundle/set.
- Local: `docs/findings/`, `docs/learnings/`, `refs/` (DDBC + bundle PDFs), `code/reference/` assets.
- Tooling note: `alpha` CLI NOT installed → paper discovery via WebSearch + WebFetch (arXiv/ACM abstracts,
  raw GitHub READMEs). Researcher agents read primary sources, write to disk, return structured digests.

## Agent fan-out (Gather)

Stage A (parallel researchers, file-based output):
- R1 **Seed decomposition** — deep-read Gao survey → taxonomy + every Future-Directions item → `...-research-survey.md`.
- R2 **Repo mining** — dedup+tag both repos, flag set/bundle/generative, compute coverage diff → `...-research-repos.md`.
- R3 **Debiasing family** (IPW/DR/exposure/popularity/position) → `...-research-debiasing.md`.
- R4 **Deconfounding family** (backdoor/front-door/deconfounder/IV) → `...-research-deconfounding.md`.
- R5 **Counterfactual + uplift** (treatment = exposure/recommendation) → `...-research-uplift.md`.
- R6 **Causal generative / diffusion** recommenders → `...-research-generative.md`.
- R7 **Set/slate/bundle-level** causal estimands + slate off-policy/bandits → `...-research-bundle.md`.

Stage B: **Bundle cross-map** synthesizer reads R1–R7 + local findings → treatment/outcome/confounder/
estimand table at set level per family → `...-research-crossmap.md`.

## Synthesize / Cite / Verify

- Lead synthesizes consensus / disagreements / open questions; Mermaid taxonomy; rank lines vs rubric.
- Cite pass (verifier-role agent): inline citations + verify every source URL resolves.
- Verify pass (adversarial reviewer-role agent): unsupported claims, single-source critical findings,
  zombie sections, "gap" claims that are actually covered. Fix FATAL before deliver; MAJOR → Open Questions.
- Adversarial check specific to this project: any proposed line that repeats the inertness trap (causal term
  bypassable at decode / globally forced) is flagged.

## Deliverables

- `research/outputs/causal-recsys-bundle-construction.md` — cited landscape + ranked lines.
- `research/outputs/causal-recsys-bundle-construction.provenance.md` — sources consulted/accepted/rejected,
  verification status, intermediate files.
- `research/promising-research-lines.md` — filled + ranked (validate/refute the 4 pre-seeded hypotheses L1–L4;
  add L5+ from survey Future Directions and the bundle gap).
- Short-list 2–3 candidate method designs → hand to brainstorming → writing-plans.

## Constraints (inherited)

Full-catalog eval; inertness diagnostic mandatory; honesty guards (multi-seed, paired Wilcoxon, report
negatives); causal term decode-path or observable-gated (not globally forced); reuse `code/reference/`.

## Task ledger

| id | task | status |
|---|---|---|
| R1 | seed decomposition (survey) | pending |
| R2 | repo mining + dedup + tag | pending (raw data already pulled) |
| R3 | debiasing family deep-dive | pending |
| R4 | deconfounding family deep-dive | pending |
| R5 | counterfactual+uplift deep-dive | pending |
| R6 | causal generative/diffusion deep-dive | pending |
| R7 | set/slate/bundle deep-dive (the gap) | pending |
| B  | bundle cross-map | pending |
| S  | synthesize + rank | pending |
| C  | cite | pending |
| V  | adversarial verify | pending |
| D  | deliver + fill research-lines | pending |

## Verification log

(empty — populated during the run)
