---
name: phase6-cpu-done-gpu-pending
description: Phase 6 (D3) DONE 2026-06-27 — verdict on held-out MealRec+H TEST = REJECT (theme inert at decode; tune +57% did not replicate). Pre-registered protocol + runbook.
metadata: 
  node_type: memory
  type: project
  originSessionId: dbf70449-0dce-4226-b882-36c1ddf37645
---

Phase 6 (D3) — the falsifiable-bar VERDICT. **CPU pipeline merged to main @ a0276f3** (2026-06-27, branch feat/phase6-falsifiable-verdict, 16 commits, subagent-driven). Supersedes [[phase5-readiness-handoff]] (Phase 5 axes + D1/D2 all merged). Spec: `docs/superpowers/specs/2026-06-26-phase6-falsifiable-verdict-design.md`; plan: `docs/superpowers/plans/2026-06-26-phase6-falsifiable-verdict.md`.

**What's built (CPU, all tested + reviewed, full suite 474 passed/0 fail):**
- `eval/significance.py` — paired_wilcoxon (exact small-n + scipy delegate), bootstrap_ci, bonferroni.
- `diagnostics/hsic_permutation.py` — HSIC permutation-null p-value.
- `diffusion/decode.py` — per-row `generators=` → EXACT batched sampler (bit-identical to per-example on CPU; CUDA is value-equiv not bit-guaranteed — float-reduction order).
- `training/evaluate.py` — `evaluate_construction_batched` (batched model forward + per-example theta; shared `_finish_example`).
- `eval/theme_contrast.py` — do(T): flip_tau_df1 (tau0=1-jaccard, signed dF1), mahalanobis_under_prior, theme_flip_contrast (real/flip/null calls).
- `eval/loadbearing_driver.py` — device-aware _grad_flow (GPU CI gate) + `compute_theme_blind` leg.
- `eval/verdict.py` — build_verdict (pure; the pre-registered decision logic).
- Runners: `train_backbone_mealrec.py --seed`; `sweep_eval_mealrec.py --split/--seed`; `do_t_contrast_mealrec.py`; `phase6_verdict.py`.

**Pre-registered protocol (§5/§6, locked BEFORE GPU — do not tune post-hoc):**
- **N=8 seeds** (n=5 was impossible: two-sided Wilcoxon p-floor 2/2^5=0.0625 can't reach 0.05). One-sided `alternative='greater'` primary, **Bonferroni m=2** {causal gain, personalization lift}.
- Causal gain = CONFIRMATORY; personalization lift = EXPLORATORY (null → "inconclusive" → PARTIAL, never REJECT).
- do(T): two flips — user-flip (sensitivity) + content-swap (theme-level causal); per-seed (tau0_flip − tau0_null) one-sided>0 AND median dF1>0.
- CI gate: load_bearing ≥5/8 seeds AND **median(delta_theme_blind)>0** (NOT the vacuous `<ablation_delta`).
- Verdict: ACCEPT iff gain_ok+gate_ok+dot_ok AND lift_sig; PARTIAL if those 3 but lift inconclusive; else REJECT.
- Lift null = **permuted-user** (primary; breaks user↔bundle alignment) + fixed-user-0 (secondary).
- Eval **protocol A-prime**: each example seeded seed+i for BOTH sampler AND `default_rng(seed+i)` pool → per-example-independent decode noise + negative shortlist (fixes D2's identical-shortlist-for-all-examples weakness; paired gains unaffected).

**GPU phase G1-G5 DONE 2026-06-27 (RTX 3090, 8 seeds, ~2.5 GPU-hr). VERDICT = REJECT.** Doc:
`docs/superpowers/2026-06-27-phase6-verdict.md` (committed caffcb2). Per-seed JSONs on laptop
`data/mealrec_h/{test_metrics,do_t}_seed{1..8}.json` (git-ignored). 16 backbones left on the box.
- **Causal gain did NOT replicate:** held-out median ΔF1 −0.0004, Bonferroni one-sided p=1.0, CI
  straddles 0. The D2 tune +57% was a single-split sampler-variance artifact — the multi-seed
  held-out protocol caught it. Personalization lift inconclusive (p_corrected 0.367, tiny +0.0005).
- **ROOT CAUSE — theme inert at decode** (the do(T) sampler-noise null is decisive): flipping the
  theme (content-swap or different-user) moves ~1.5–5% of the completion (tau0_flip ≈ 0.015–0.05)
  while a different DECODE seed alone moves ~99% (tau0_null ≈ 0.99). A 16-d theme on the DiT adaLN
  cond is a tiny perturbation vs the MDLM sampler's variance on the |gt|=1 / ρ=100 task. "In the
  architecture ≠ exercised at inference" — exactly what CLAUDE.md warns about; do(T) surfaced it.
- HSIC(θ,exposure) p=0.156 → θ IS deconfounded (machinery works; failure is downstream). Gate
  fires 0% on TEST (selector dormant; cold-slice CI load-bearing in only 1/8).
- **KL-collapse mitigation would NOT help** (theme already deconfounded; problem is decode inertness).
  Forward options (see verdict doc §Implications): (a) write up the negative result + the do(T)
  sampler-noise-null diagnostic; (b) NEW SNR regime (|gt|≥2, smaller shortlist, fewer/cooler decode
  steps) then re-test; (c) stronger theme injection (cross-attn/per-layer) under the conditional-gate
  discipline. NOT a tweak to the current setting.

**Two GPU-surfaced bug fixes landed on main** (CPU smoke missed both): e37f8fc + **aecb8db** — the CI
gate's `_grad_flow` moves the selector to cuda IN-PLACE (nn.Module.to is in-place), so the later
theme-blind leg got a cuda selector with CPU inputs → device mismatch only when the gate activates at
TEST scale. Fix: `_finish_example` builds selector inputs on the selector's own device. Lesson:
limit-4 smoke didn't activate the gate, so it missed the selector-path device bug.

Box ssh: `ssh -p 26705 root@ssh2.vast.ai` (this run). Guard launches with FULLY bracketed patterns
AND no literal script-name elsewhere in the same command — `pkill -f run_seeds` self-matched the ssh
shell's own argv (the `~/run_seeds.sh` reference); use `pkill python` + `pkill -f "run_[s]eeds"`.
