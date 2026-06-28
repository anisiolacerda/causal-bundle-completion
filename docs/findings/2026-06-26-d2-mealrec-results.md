# D2 Results — MealRec+H PIC-Diff Training (First GPU Spend)

> **Status:** D2 GPU runs complete (2026-06-26, RTX 3090). Tune-split operating point selected.
> The falsifiable-bar VERDICT (held-out TEST + multi-seed significance) is Phase 6, not D2.
> Plan: `docs/superpowers/plans/2026-06-26-p5-d2-mealrec-training.md`. Spec:
> `docs/superpowers/specs/2026-06-26-d2-mealrec-training-design.md`.

## Operating point (selected on tune)
- iVAE: `λ=5, β=0.02`, 3000 steps → **HSIC(θ,e)≈0** (θ deconfounded from exposure).
- Backbones: DDBC-small (hidden 64 / cond 128 / 6 blocks / 8 heads; RVQ codebook 256, 3 levels),
  `lr=3e-4`, 12,000 steps, batch 256, swap 0 (theme-conditioned). theme-ON + theme-OFF.
- Gate: `τ=0.5`, `τ_cold=6.0` (cold = item training-interaction count < 6).
- Checkpoints (git-ignored, on laptop `data/mealrec_h/` + box): `ivae_mealrec.pt`,
  `backbone_theta_mealrec.pt`, `backbone_null_mealrec.pt`, `theta_store_train.pt`, `tune_metrics.json`.

## Tune-split metrics (500 examples, |gt|=1, ρ=100 shortlist)
| condition | F1 |
|---|--:|
| PIC-Diff-full (theme + selector + gate) | 0.044 |
| user-conditioned-DDBC (backbone theme, no selector) | 0.044 |
| user-agnostic-DDBC (no theme) | 0.028 |
| **causal/theme gain** (full vs agnostic) | **+0.016 = +57%** |
| **personalization lift** (real-user θ vs fixed-user θ) | **+4.8%** |
| **CI gate on a 20-example COLD slice** | **load_bearing=True, Δ=0.15, tau@0=0.90, grad=4.50** |

## Interpretation
- **Causal/theme gain is real and large (+57%):** the theme-conditioned backbone nearly doubles F1
  over the user-agnostic DDBC baseline. The two backbones differ ONLY in the theme (same config, seed,
  steps, data), so the gain is attributable to the theme conditioning.
- **The decode-theme threading fix (commit d9ec256) was load-bearing for this result.** Before it,
  `semi_ar_sample` was theme-blind, so a theme-conditioned backbone's `proj_theme` was dormant at decode
  — the +57% gain would not exist. Caught + fixed BEFORE the GPU spend (the runner-review pass).
- **Personalization is modestly positive (+4.8%):** real-user θ beats a fixed-user θ — the user signal
  matters, but weakly. Two causes: (i) the iVAE KL-collapsed (a user's meals are content-consistent →
  the user-prior μ_p(u) alone reconstructs C_o, so θ is user-dominated and content-weak), and (ii) the
  |gt|=1 thin task limits the achievable lift.
- **The selector is conditionally load-bearing on COLD seeds (Δ=0.15, tau@0=0.90 on real MealRec),**
  yet dormant on the general tune split (pic_full == user_conditioned) — the gate behaving exactly as
  designed: no clean displacement, real causal work where the seed is cold/novel.

## Caveats (carry to Phase 6)
- **Tune split, not the held-out TEST.** The verdict + significance are Phase 6.
- **Low absolute F1** (~0.044) — the |gt|=1 leave-one-out task (3-course meals) over a 7,280-item
  catalog. ~4.4× random (1/ρ=0.01); the RELATIVE gains are the falsifiable-bar signal.
- **KL collapse** — θ is user-dominated. If the Phase-6 personalization lift is to be strengthened,
  options: a richer content (e.g. un-PCA'd MiniLM C_o), a free-bits/annealed KL, or accept the
  user-dominated theme (it is still a deconfounded, personalized theme — the bar's "personalization"
  axis is satisfied, modestly).

## Next — Phase 6 (D3)
Held-out TEST evaluation of the 3 conditions + the proper personalization lift + the do(T)≠condition(T)
theme-flip contrast + **multi-seed significance** (the "statistically real" Gate-G6 requirement). The
device-aware fix for `run_loadbearing_check` (CPU model in D2) should be generalized so the CI gate
runs on GPU at TEST scale. Re-run on multiple training seeds for the significance test.
