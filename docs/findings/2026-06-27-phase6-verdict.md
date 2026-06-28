# Phase 6 — Falsifiable-Bar Verdict (MealRec+H, held-out TEST, 8 seeds)

> **VERDICT: REJECT.** PIC-Diff does not clear the pre-registered falsifiable bar on the
> held-out MealRec+H TEST split. The tune-set causal gain (+57%) does **not** replicate across
> 8 re-trained seeds, and the theme is **inert at decode** (flipping it changes the completion far
> less than sampler noise). The deconfounding machinery works (HSIC p=0.156); the failure is
> downstream — the deconfounded theme does not influence the MDLM decode. Run 2026-06-27,
> RTX 3090, 8 seeds, frozen D2 θ, eval protocol A-prime (per-example-independent decode noise +
> negative shortlist). hsic_p = 0.156 (3000-subsample permutation null).

## Summary of the pre-registered tests

| Test | Result | Pre-registered bar | Pass? |
|---|---|---|---|
| **Causal gain** (pic_full − user_agnostic), 8-seed one-sided paired Wilcoxon, Bonferroni m=2 | median ΔF1 **−0.0004**, p_corrected **1.000**, 95% CI [−0.0026, +0.0021] | p<0.05 AND CI excludes 0 | **NO** |
| **Personalization lift** vs permuted-user (exploratory) | median ΔF1 **+0.0005**, p_corrected **0.367**, CI [−0.0003, +0.0009] | significant ⇒ strengthen; null ⇒ inconclusive | inconclusive |
| **do(T) content-swap** (theme-level causal): per-seed (tau0_flip − tau0_null) one-sided>0 AND median ΔF1>0 | median (tau0_flip − tau0_null) **−0.971**, p **1.000**, median ΔF1 **−0.0005** | both > 0 | **NO** |
| **CI gate** cold-slice load-bearing | load_bearing in **1/8** seeds; theme_carries **False** | ≥5/8 AND theme carries | **NO** |
| **HSIC(θ, exposure) permutation** | p **0.156** | report (not gated) | θ deconfounded |

**Condition 1 (control, DDBC parity on Spotify-90): met** earlier (F1 0.298 vs 0.287), cited not re-run.
**Condition 2 (primary, statistically-real causal + personalization gain): NOT met.** → REJECT.

## Per-seed values (8 seeds)

| seed | pic_full F1 | user_agnostic F1 | gain | lift_vs_permuted | gate_firing | do(T) content tau0_flip / null |
|---|--:|--:|--:|--:|--:|--:|
| 1 | 0.0455 | 0.0483 | −0.0028 | −0.0015 | 0.00 | 0.017 / 0.987 |
| 2 | 0.0438 | 0.0502 | −0.0064 | +0.0004 | 0.00 | 0.013 / 0.993 |
| 3 | 0.0487 | 0.0490 | −0.0002 | +0.0006 | 0.00 | 0.015 / 0.990 |
| 4 | 0.0505 | 0.0524 | −0.0019 | −0.0006 | 0.00 | 0.014 / 0.984 |
| 5 | 0.0502 | 0.0451 | +0.0051 | +0.0002 | 0.00 | 0.016 / 0.986 |
| 6 | 0.0490 | 0.0477 | +0.0013 | +0.0015 | 0.00 | 0.017 / 0.989 |
| 7 | 0.0453 | 0.0460 | −0.0006 | +0.0011 | 0.00 | 0.015 / 0.990 |
| 8 | 0.0483 | 0.0447 | +0.0036 | +0.0011 | 0.00 | 0.022 / 0.990 |

(pic_full == user_conditioned every seed — the SDPP gate never fires on TEST, so the selector is
dormant and only the theme-conditioning axis is live.)

## Interpretation (facts)

1. **The causal gain did not replicate.** On the held-out TEST across 8 re-trained seeds the
   theme-conditioned backbone does not beat the user-agnostic DDBC (median −0.0004, 5/8 seeds
   negative, p_corrected = 1.0). The D2 tune-set +57% was a single-split artifact, almost certainly
   sampler-variance on the ~300–500-example tune slice — exactly the failure the multi-seed held-out
   protocol was built to catch.

2. **The theme is inert at decode — the root cause.** The do(T) sampler-noise null is decisive:
   re-running the *same* θ with a different decode seed changes ~**99%** of the completed items
   (tau0_null ≈ 0.99), while *flipping* the theme (content-swap or different-user) changes only
   ~**1.5–5%** (tau0_flip ≈ 0.015–0.05). The theme signal is overwhelmed by the MDLM sampler's
   variance, so it does not survive to the output. This is precisely the "in the architecture ≠
   exercised at inference" gap CLAUDE.md warns about — and the do(T) test surfaced it.

3. **Personalization is inconclusive, not disproven** (pre-registered exploratory). The lift over a
   permuted-user null is tiny (+0.0005) and non-significant (p_corrected 0.367); the test is
   underpowered and, more fundamentally, rides on a theme that is itself inert.

4. **The deconfounding worked.** HSIC(θ, exposure) p=0.156 — the iVAE theme shows no detectable
   dependence on the exposure proxy. The machinery is sound; the problem is strictly downstream.

5. **The selector axis is dormant on TEST.** gate_firing_rate = 0.00 every seed — TEST seed-items
   are not cold under the τ_cold=6.0 calibration, so the SDPP selector never activates in production.
   The cold-slice CI gate is load-bearing in only 1/8 seeds (a forced-cold slice), confirming the
   selector contributes nothing to the actual TEST result.

## Diagnosis (hypotheses — labeled)

- **HYPOTHESIS (high confidence):** the |gt|=1 leave-one-out task over a 7,280-item catalog with a
  ρ=100 shortlist is intrinsically high-variance at decode (the model picks ~1 item from 100, and
  two decode seeds disagree 99% of the time → F1 ≈ 0.045 ≈ 4.5× the 1/100 random floor). A 16-d
  theme added to the DiT cond (adaLN) is a small perturbation relative to that noise, so it cannot
  move the metric — regardless of how good the theme is.
- **HYPOTHESIS:** the +57% tune result was sampler noise on one split; the held-out, multi-seed,
  protocol-A-prime evaluation removes that artifact (independent decode noise + independent negative
  shortlist per example), exposing the null.
- The KL-collapse mitigation (the previously-flagged conditional follow-up) would **not** rescue this:
  the theme is already deconfounded (HSIC p=0.156); the failure is that a deconfounded theme does not
  influence the decode, which improving the theme's user-specificity does not address.

## Implications & options (for decision — none oversold)

1. **Accept the negative result and write it up.** "A deconfounded bundle-theme, correctly
   estimated and verified independent of exposure, is inert under a high-variance masked-diffusion
   decoder on |gt|=1 MealRec+H" is a real, falsifiable, publishable *negative* finding — and the
   do(T) sampler-noise null is a clean diagnostic others can reuse.
2. **Change the signal-to-noise regime (new experiment, not a tweak):** a denser completion task
   (|gt| ≥ 2), a much smaller candidate shortlist, or fewer decode steps / lower decode temperature
   to cut sampler variance — then re-test whether the theme becomes load-bearing. This is a new
   benchmark setting, not a fix to the current one.
3. **Stronger theme injection** (cross-attention or per-layer θ instead of a single adaLN add) —
   but CLAUDE.md's 3× displacement lesson cautions against forcing it global; would need the
   conditional-gate discipline and its own do(T) check.
4. **Do NOT** pursue the KL-collapse mitigation for this bar (see Diagnosis).

The rigorous outcome stands: on this benchmark, with this backbone, PIC-Diff's causal/personalization
gain is not statistically real on held-out TEST. The falsifiable bar did its job.
