---
name: push-to-winner-method
description: "Standing directive — keep creatively pushing PIC-Diff to a method that actually WINS; don't settle for the negative result."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dbf70449-0dce-4226-b882-36c1ddf37645
---

Standing instruction (stated 2026-06-27, after the Phase-6 REJECT): **keep pushing to a winner
method, and be creative.** Do not treat the honest negative result as the stopping point — treat it
as the launchpad. Actively search for and BUILD a method that clears the bar (beats DDBC on its
benchmark AND adds a real, do(T)-passing causal/personalization gain), exploring many angles and
novel mechanisms, not just the obvious relocation of the same idea.

**Why:** the project's primary objective is a *publishable winner*, not a diagnosis. The user has
repeated this push twice. They want momentum toward a working method and creative breadth in the
search, while still respecting the rigor guards (fail-fast probes, pre-registered thresholds, the
do(T) / re-sampled-recon honesty checks — the [[phase6-cpu-done-gpu-pending]] artifact trap).

**How to apply:**
- Default to ACTION toward a winner: run the cheap decisive probe, then BUILD the most promising
  method (don't stall in analysis). The recovery program (`docs/superpowers/specs/2026-06-27-pic-diff-recovery-design.md`)
  is the current vehicle — Stage 1 probes route to Stage 2 (TAFR theme term) or Stage 3 (theme-free
  decode-path reranker that beats DDBC's cosine-NN argmax); both are winner candidates, pursue
  whichever the evidence supports rather than stopping.
- Be creative: when an angle dead-ends (e.g. θ uninformative), pivot the *estimand/mechanism*, not
  just the placement — a learned decode reranker, a seed-pair-conditioned theme re-derivation,
  test-time adaptation, etc. Generate and adversarially judge multiple options (the ideation-panel
  workflow worked well).
- Keep the honesty guards ON while pushing — a "winner" only counts if it survives the held-out,
  multi-seed, re-sampled-recon, do(T) bar. Creative ≠ cutting corners on the falsifiable bar.
