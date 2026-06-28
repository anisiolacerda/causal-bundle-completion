---
name: g3-control-parity-result
description: Gate G3 (DDBC control parity) is MET — the winning Spotify-90 config + the hard lesson
metadata: 
  node_type: memory
  type: project
  originSessionId: a08a5fb1-7016-40e4-b57e-30cdeb9471b9
---

**G3 (control parity) MET on 2026-06-26.** Clean-room DDBC reproduction on Spotify-90:
**F1 0.298 / Jaccard 0.186 / OAS 0.689** (EMA model) vs the paper's **0.287 / 0.177 / 0.684** — matched/beat on all three. Falsifiable-bar control-track condition (1) satisfied. Validated checkpoint: `models/models_aug_validated.pt`; main at `568429b`.

**Winning config:** un-normalized RVQ (3 levels, C=256, +dedup); DiT small (hidden 64 / cond 128 / 6 blocks / 8 heads, custom learnable-gain fp32 LayerNorm, zero-init output head, time_conditioning=False); AdamW lr3e-4 wd0, 500-step warmup, **annealed EMA** `min(0.9999,(1+n)/(10+n))`, batch 512, 20k steps, **swap_ratio=0.4** train augmentation; semi-AR DDPM-cache sampler 25 steps (gradual reveal + SUBS at inference); eval = ρ=100 shortlist with **without-replacement** retrieval (exactly |b_y| distinct items).

**The hard lesson (cost ~4 GPU runs):** the persistent 0.17-vs-0.287 gap was mostly a **scoring/measurement bug, not model quality.** Three real fixes mattered, found by spec-audit + a target-protocol investigation, not by guessing: (1) EMA used fixed 0.9999 from step 0 → 13.5% init contamination → [[decode-vs-rvq-nlevels-convention]]-style careful spec read fixed it; (2) the 25-step sampler collapsed to one-shot decode (masked_fill killed the stay-mask weight); (3) **decisive** — our eval retrieved WITH replacement over the full catalog, but DDBC retrieves WITHOUT replacement from the ρ=100 shortlist (forces |pred|=|b_y| distinct → precision==recall==F1). The without-replacement fix alone moved F1 0.17→0.298, **eval-only, no retrain.** RVQ-normalize, swap-augmentation, and DiT-init were each tested and did NOT move F1 under the wrong scoring — so attribute cautiously. **Lesson: before blaming the model for a reproduction gap, verify the eval protocol matches the paper exactly (candidate set, with/without replacement, |pred| size, metric definition).**
