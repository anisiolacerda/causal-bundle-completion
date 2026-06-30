# Provenance — causal-recsys-bundle-construction

- **Date:** 2026-06-28
- **Slug:** `causal-recsys-bundle-construction`
- **Workflow:** `/lit` (literature-review skill): Plan → Gather → Synthesize → Cite → Verify → Deliver,
  orchestrated with the Workflow tool (ultracode).
- **Final output:** [causal-recsys-bundle-construction.md](causal-recsys-bundle-construction.md)
- **Plan artifact:** [.plans/causal-recsys-bundle-construction.md](.plans/causal-recsys-bundle-construction.md)
- **Draft:** [.drafts/causal-recsys-bundle-construction-draft.md](.drafts/causal-recsys-bundle-construction-draft.md)
  (pre-adversarial-review; superseded by the final).
- **Downstream:** [../promising-research-lines.md](../promising-research-lines.md) (filled + ranked).

## Tooling note

`alpha` CLI (alphaXiv) was **not installed** in this environment. All paper discovery and verification
ran via **WebSearch + WebFetch** (arXiv abstract/HTML, ACM DL, PMLR, OpenReview, raw GitHub READMEs).
ACM DL returns HTTP 403 to the automated fetcher (anti-bot) for several DOIs; those were confirmed via
WebSearch DOI→title resolution instead (marked below).

## Method / agent accounting

- **Gather workflow** (`wf_a3497251-500`): 8 agents, 465,720 output tokens, 242 tool calls, ~18 min.
  - 7 parallel `researcher` agents (general-purpose): R1 survey decomposition, R2 repo mining + coverage
    diff, R3 debiasing, R4 deconfounding, R5 counterfactual+uplift, R6 causal generative/diffusion,
    R7 set/slate/bundle. + 1 cross-map synthesizer.
  - Raw structured digests: `tasks/w5jjl902t.output` (1746 lines; per-paper tables, key findings,
    open problems, gap assessments, candidate lines, cross-map, ranked lines, shortlist).
- **Cite + Verify workflow** (`wf_995ac903-5d8`, resumed as run `w3si1zmul`): 7 agents.
  - 6 parallel citation-verifier agents fetched all 56 cited URLs.
  - 1 adversarial reviewer (effort=high) stress-tested the ranked lines for novelty collision,
    inertness trap, bar-clearing realism, identifiability, single-source risk.
  - Raw output: `tasks/w3si1zmul.output`.
  - (First attempt `wf_995ac903-5d8` failed at the review step on an `args` plumbing bug; fixed by
    inlining the ranked-lines data and resuming — cite agents returned cached.)

## Sources consulted → accepted / rejected

- **Primary seeds (accepted):** Gao et al. survey (arXiv 2208.12397); the two curated repos
  (tsinghua-fib-lab/Causal-Recommender-Systems; Chrissie-Law/Causal-Inference-for-Recommendation) —
  deduped union ~70 papers, ~90% pointwise, 0 set-construction, 0 causal-generative.
- **Local corpus (accepted):** `docs/findings/00-SUMMARY.md`, `docs/learnings/LEARNINGS-*.md`,
  `docs/baselines/ddbc-repro-spec.md`, `refs/` (DDBC + bundle PDFs), `code/reference/`.
- **Rejected / down-weighted:** counterfactual-as-augmentation lines (CauseRec, CASR, CLBR,
  counterfactual-session-aug) — bypassable at decode, not causal estimands; IV4Rec/iDCF — require
  instruments/proxies our datasets lack.

## Citation verification (56 URLs fetched)

**54 RESOLVE** to the claimed paper. **2 ERRORS found and corrected in the final doc:**

| Claim | Bad URL (rejected) | Status | Corrected reference |
|---|---|---|---|
| COR — Causal Representation Learning for OOD Recommendation | arXiv 2204.00800 | **MISMATCH** (that ID = an unrelated cs.NI networking paper) | DOI 10.1145/3485447.3512251 (WWW 2022, Wang/Lin/Feng/He/Lin/Chua) |
| NCoRE | arXiv 2103.11589 | **MISMATCH** (that ID = "Adversarially Optimized Mixup") + wrong title | arXiv 2103.11175, "Neural Counterfactual **Representation Learning** for Combinations of Treatments" (Parbhoo/Bauer/Schwab) |

**Confirmed REAL (not hallucinated) — the pivotal post-cutoff papers the verdict depends on:**
Cadence (2512.17733, Dec'25), A2G-DiffRec (2602.14706, SIGIR'26), Su et al. orthogonal uplift
(2602.19851, Feb'26), CNSDiff (2508.07243), DCFG (2506.14399), control-function position bias
(2506.06989), DMSG (2408.06883), CausalDiffRec (2408.00490), UpliftRec (2405.08582), OPCB (2408.11202).

**Verified via WebSearch DOI→title (ACM 403 to fetcher, link valid):** CPR (10.1145/3459637.3482305),
Goldenberg retrospective uplift (10.1145/3383313.3412215), ULRMF/ULBPR (10.1145/3298689.3347018),
CountER (10.1145/3459637.3482420), CVID (10.1145/3745023), BALF
(10.1007/978-981-92-1462-4_8). **PDF binaries (loaded, parsed locally/by search):** Top-K off-policy
correction (alexbeutel.com), Joachims unbiased-LTR (Cornell), Wang/Blei Deconfounded Recommender (Columbia).

**Minor venue/title notes (paper identity confirmed):** MACR full title adds "for Eliminating … in
Recommender System" (KDD 2021); CR adds "for Mitigating Clickbait Issue"; HCR is IEEE TKDE 2023/24;
DifFaiRec venue "ICDE 2025" not confirmable from arXiv (identity matches); DCFG canonical title is
"**Decoupled** Classifier-Free Guidance" (an early fetch showed a stale "Factored" title).

## Verification status of strong claims

- **GAP A (no set/bundle causal estimand for construction):** VERIFIED — survey scan (0 hits) + deduped
  repo union (0 set-construction) + spot-checks (CauseRec/CLBR confirmed augmentation/ranking, not
  construction).
- **GAP B (no causal generative/diffusion decoder):** VERIFIED — generative decoders (DiffRec/DreamRec/
  TIGER/GeMS/DMSG/DDBC) confirmed non-causal; causal×generative works (CausalDiffRec/CNSDiff/CVID/
  DifFaiRec/A2G-DiffRec) confirmed off-decode-path or fairness/representation, none a set-construction
  causal estimand.
- **NOVELTY (downgraded by adversarial review):** "deconfound co-occurrence vs popularity" is **PRIOR
  ART** (Cadence). Defensible novelty re-scoped to set-level + generative-decode-path +
  intervention-verified + within-set-interaction. Cadence-UACR added as a mandatory baseline.
- **IDENTIFIABILITY (FATAL flag applied):** uplift/propensity/IV/slate-OPE lines (#3,#5,#6,#7,#9) are
  unidentifiable-as-stated on MealRec/Spotify-MPD (no exposure logs / no user ids); re-stated or moved
  to Open Questions.
- **INERTNESS (FATAL flag applied):** scalar popularity reweights (Set-PDA, substitute-confounder) can
  be monotone → PPMI-equivalent → inert in ranking; final adds a mandatory **rank-inversion** test and
  per-slot observable-conditioned γ.

## Intermediate artifacts

- `tasks/w5jjl902t.output` — full gather digest (researchers + cross-map), structured JSON.
- `tasks/w3si1zmul.output` — full cite + adversarial-review digest, structured JSON.
- Workflow scripts under `…/workflows/scripts/causal-recsys-bundle-gather-*.js` and
  `…-cite-verify-*.js`.

## Honesty caveats

- The survey's cutoff (2022–23) means the gap claim is established against the field's canon; newer
  causal-diffusion works (CausalDiffRec, CNSDiff, CVID, A2G-DiffRec, Cadence) were found by directed
  web search and are accounted for above — they do **not** close the set-construction gap but they do
  **narrow the deconfounding-novelty claim** (Cadence).
- No empirical result is asserted here; all method-design claims are HYPOTHESES with pre-registered
  win conditions and falsification gates, to be tested under the inherited honesty guards (full-catalog,
  multi-seed, paired Wilcoxon, inertness battery).
