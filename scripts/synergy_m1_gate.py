"""Synergy falsification gate (Design 1, slice-1): measure whether within-set synergy
tau_syn(j|{i,k}) carries non-additive completion signal co-occ cannot represent. CPU-only,
full-catalog, no training. KILL-or-PROCEED per the pre-registered thresholds in `decide`.

Run (predecessor venv + its src on PYTHONPATH):
  PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
  /Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
  scripts/synergy_m1_gate.py \
    --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/mealrec_l \
    --dataset mealrec_l --topk 200 --seed 0 --out results/synergy/m1/mealrec_l.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synergy.gate import run_gate
from causalbr.eval.significance import paired_wilcoxon


# ---- PRE-REGISTERED THRESHOLDS (frozen; see plan Task 6) ----
TAU_PERM_P_MAX = 0.05        # residual must beat the additivity-null permutation (one-sided)
RANK_SPEARMAN_MAX = 0.90     # >= this AND jaccard >= JACCARD_MAX => rank-equivalent (inert)
RANK_JACCARD_MAX = 0.90
HIT_WILCOXON_P_MAX = 0.05    # joint-conditional > co-occ on gt hit@20 (one-sided)
MASKED_FRAC_MIN = 0.01       # >=1% masked-complement instances
PLACEBO_TAU_RATIO_MAX = 0.5  # placebo tau must be <= half the observed tau (calibration)


def decide(m: dict) -> dict:
    """Apply the frozen thresholds. KILL if inert/miscalibrated; else GO (STRONG/WEAK)."""
    failed = []
    if m["tau_perm_p"] > TAU_PERM_P_MAX:
        failed.append("residual_at_null")
    if (m["rank"]["spearman_mean"] >= RANK_SPEARMAN_MAX
            and m["rank"]["top20_jaccard_mean"] >= RANK_JACCARD_MAX):
        failed.append("rank_equivalent_to_cooc")
    if not m.get("placebo_ok", False):
        failed.append("placebo_miscalibrated")
    if failed:
        return {"verdict": "KILL", "failed": failed}
    strong = (m["hit"][20]["wilcoxon_p"] <= HIT_WILCOXON_P_MAX
              and m["hit"][20]["joint_mean"] > m["hit"][20]["cooc_mean"]
              and m["masked_frac"] >= MASKED_FRAC_MIN)
    return {"verdict": "GO", "strength": "STRONG" if strong else "WEAK", "failed": []}


def _load_mealrec(data_dir):
    tr = np.load(os.path.join(data_dir, "train_examples.npy"), allow_pickle=True)
    te = np.load(os.path.join(data_dir, "test_examples.npy"), allow_pickle=True)
    train = [list(row[1]) for row in tr]
    obs = [list(row[1]) for row in te]
    gt = [list(row[2]) for row in te]
    n_items = 1 + max(max((max(b) for b in train if b), default=0),
                      max((max(b) for b in obs if b), default=0),
                      max((max(g) for g in gt if g), default=0))
    return train, obs, gt, int(n_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, obs, gt, n_items = _load_mealrec(args.data_dir)
    rng = np.random.default_rng(args.seed)
    g = run_gate(train, obs, gt, n_items, ks=tuple(args.ks), topk=args.topk, rng=rng)

    # residual vs null: one-sided paired Wilcoxon over per-instance tau_abs vs null
    # (recompute paired arrays through run_gate is overkill; use the summary + a permutation p)
    # permutation p: fraction of null instances whose tau_abs >= observed mean.
    tau_perm_p = 1.0
    if g["n_instances"]:
        # conservative: compare observed mean to null distribution mean via Wilcoxon on the
        # per-instance arrays returned below; run_gate exposes them via re-run in strict mode.
        tau_perm_p = _tau_perm_p(train, obs, gt, n_items, args.topk, args.seed, g["tau_abs_mean"])

    # completion lift: paired Wilcoxon joint > cooc on hit@20
    h20 = g["hit"][20]
    wp = paired_wilcoxon(h20["joint"], h20["cooc"], alternative="greater")["p_value"] \
        if h20["cooc"] else 1.0
    g["hit"][20]["wilcoxon_p"] = float(wp)

    placebo_ok = (g["tau_abs_mean_null"] <= PLACEBO_TAU_RATIO_MAX * max(g["tau_abs_mean"], 1e-12))

    metrics = {
        "tau_abs_mean": g["tau_abs_mean"], "tau_abs_mean_null": g["tau_abs_mean_null"],
        "tau_perm_p": float(tau_perm_p), "rank": g["rank"],
        "hit": {20: {"cooc_mean": h20["cooc_mean"], "joint_mean": h20["joint_mean"],
                     "wilcoxon_p": float(wp)}},
        "masked_frac": g["masked_frac"], "placebo_ok": bool(placebo_ok),
    }
    verdict = decide(metrics)
    out = {"dataset": args.dataset, "n_items": n_items, "n_instances": g["n_instances"],
           "topk": args.topk, "seed": args.seed, "metrics": metrics,
           "hit_full": {k: {"cooc_mean": g["hit"][k]["cooc_mean"],
                            "joint_mean": g["hit"][k]["joint_mean"]} for k in args.ks},
           "verdict": verdict}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"dataset": args.dataset, "n_instances": g["n_instances"],
                      "metrics": metrics, "verdict": verdict}, indent=2))


def _tau_perm_p(train, obs, gt, n_items, topk, seed, observed_mean):
    """One-sided permutation p: build R additivity-null datasets, compute the tau_abs_mean
    statistic on each; p = (#{null_mean >= observed_mean} + 1)/(R+1)."""
    from synergy.gate import run_gate as _rg
    R = 19
    ge = 0
    for r in range(R):
        rng = np.random.default_rng(seed * 1000 + r + 1)
        gr = _rg(train, obs, gt, n_items, ks=(20,), topk=topk, rng=rng)
        # run_gate already compares to its own null; here we want the null statistic itself:
        if gr["tau_abs_mean_null"] >= observed_mean:
            ge += 1
    return (ge + 1) / (R + 1)


if __name__ == "__main__":
    main()
