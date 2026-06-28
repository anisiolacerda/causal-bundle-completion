# Seed sources for the causal-recsys literature research

The next session's deep multi-agent sweep starts here. Chain outward from these via citations.

## Primary seed (deep-read first)

- **Causal Inference in Recommender Systems: A Survey and Future Directions** — Gao, Chen et al.
  (Tsinghua FIB Lab). The taxonomy backbone. Fetch the PDF/arXiv version into `refs/` in the next
  session (not bundle-specific, so not in the inherited `refs/`).
  - Read for: the survey's split of causal-recsys into **(i) causal debiasing** (IPW/doubly-robust,
    exposure/selection/popularity/position bias), **(ii) causal effect estimation / uplift**
    (treatment = recommendation/exposure), **(iii) counterfactual / deconfounding** (backdoor,
    front-door, instrumental, deconfounder), and the **Future Directions** section (the richest vein
    for novel method ideas).

## Curated paper-list repos (mine + chase the linked papers)

- **github.com/tsinghua-fib-lab/Causal-Recommender-Systems** — the survey authors' own curated list.
  Treat as the authoritative citation graph for the seed.
- **github.com/Chrissie-Law/Causal-Inference-for-Recommendation** — second curated awesome-list.
  Cross-reference; capture papers NOT in the FIB list (coverage diff is signal).

> For both repos: extract every linked paper (title, venue, year, link), dedup across the two lists,
> tag by survey category, and flag the few that touch **set/slate/bundle** construction or
> **generative** recommenders — those are the closest to our target and likely sparse (a gap = an
> opportunity).

## Domain refs already in `refs/` (bundle side)

- `Discrete Diffusion for Bundle Construction.pdf` — **DDBC**, the SOTA we must beat. Anchor baseline.
- `A Survey on Bundle Recommendation- Methods, Applications, and Challenges.pdf`
- `Context-aware Sequential Bundle Recommendation via User-specific Representations.pdf`
- `Leveraging Multimodal Features and Item-level User Feedback for Bundle Construction.pdf`
- `Text2Bundle`, `Adaptive In-Context Learning for Bundle Generation`,
  `Fine-tuning multimodal LLMs for product bundling`, `Does KD matter for LLM bundle generation` — LLM/bundle context.

## The gap to probe (HYPOTHESIS to confirm in the review)

Causal-recsys is mature for **pointwise ranking debiasing** (exposure/popularity confounders) but
thin for **set/bundle construction** and for **generative/diffusion** decoders. The novel method
likely lives at that intersection: a causal estimand defined over the **constructed set** (not a
single item), with the confounder being **exposure/popularity/co-occurrence** structure (which DID
carry signal — co-occ wins) rather than a user-theme (which did NOT).
