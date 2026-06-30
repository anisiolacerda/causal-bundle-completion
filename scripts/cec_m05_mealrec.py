# scripts/cec_m05_mealrec.py
"""CEC M0.5 confirmation on MealRec+ (dense regime): content (SetCompleter) vs co-occ,
per-gt-item ranks stratified by ITEM-exposure percentile. Mirrors cec_m05_exposure_premise.py
but reads the MealRec construction .npy examples and RETRAINS SetCompleter (no checkpoint exists).

Run (predecessor venv + its src on PYTHONPATH):
  PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
  /Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
  scripts/cec_m05_mealrec.py \
    --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/mealrec_l \
    --dataset mealrec_l --n-bins 5 --ks 1 20 --epochs 20 --seed 0 \
    --out results/cec/m05/mealrec_l.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.sparse import csr_matrix
import torch

from causalbr.recovery.set_completer import setcompleter_full_scores, train_set_completer
from causalbr.eval.significance import paired_wilcoxon

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cec.exposure import item_exposure, exposure_bin_edges, assign_bin
from cec.stratify import item_rank, per_bin_paired_stats


def item_bundle_incidence(bundles, n_items):
    """Item x bundle incidence csr (same as the Spotify runner; memory-safe co-occ)."""
    rows, cols = [], []
    for b_idx, b in enumerate(bundles):
        for it in b:
            if 0 <= it < n_items:
                rows.append(it); cols.append(b_idx)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_items, len(bundles)))


def cooc_scores(B, observed, n_items):
    """cs[j] = sum_{o in observed} cooc(o,j) for non-observed j, via B @ (B[observed].sum(0))."""
    if not observed:
        return np.full(n_items, -np.inf)
    overlap = np.asarray(B[observed].sum(axis=0)).ravel()
    cs = np.asarray(B @ overlap).ravel().astype(float)
    cs[observed] = -np.inf
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="mealrec_l")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = args.data_dir
    train_ex = np.load(os.path.join(d, "train_examples.npy"), allow_pickle=True)
    test_ex = np.load(os.path.join(d, "test_examples.npy"), allow_pickle=True)
    item_emb = torch.load(os.path.join(d, "item_emb.pt"), map_location="cpu", weights_only=False).float().numpy()
    n_items = int(item_emb.shape[0])

    train = [list(row[1]) for row in train_ex]            # full meals (item lists)
    obs = [list(row[1]) for row in test_ex]               # observed courses
    gts = [list(row[2]) for row in test_ex]               # held-out course(s)

    exposure = item_exposure(train, n_items)              # per-ITEM train frequency
    edges = exposure_bin_edges(exposure, n_bins=args.n_bins)
    B = item_bundle_incidence(train, n_items)
    model = train_set_completer(train, item_emb, hidden=128, epochs=args.epochs, seed=args.seed)
    model.eval()

    rec_bin, rec = [], {f"content_hit{k}": [] for k in args.ks}
    rec.update({f"cooc_hit{k}": [] for k in args.ks})
    with torch.no_grad():
        for i in range(len(obs)):
            seed = obs[i]; gt = gts[i]
            if not seed or not gt:
                continue
            sc = setcompleter_full_scores(model, seed, n_items, mask_observed=True)
            cc = cooc_scores(B, seed, n_items)
            for g in gt:
                if g < 0 or g >= n_items:
                    continue
                rc = item_rank(sc, g); rk = item_rank(cc, g)
                rec_bin.append(int(assign_bin(np.array([exposure[g]]), edges)[0]))
                for k in args.ks:
                    rec[f"content_hit{k}"].append(1.0 if rc <= k else 0.0)
                    rec[f"cooc_hit{k}"].append(1.0 if rk <= k else 0.0)

    rec_bin = np.array(rec_bin)
    sig = lambda a, b: paired_wilcoxon(a, b, alternative="greater")
    per_bin = {}
    for k in args.ks:
        per_bin[f"hit{k}"] = per_bin_paired_stats(
            np.array(rec[f"content_hit{k}"]), np.array(rec[f"cooc_hit{k}"]),
            rec_bin, n_bins=args.n_bins, sig_fn=sig)

    d1 = [b["delta"] for b in per_bin["hit1"]]
    monotone = all((d1[j] + 1e-9) >= d1[j + 1] for j in range(len(d1) - 1))
    top = per_bin["hit1"][-1]
    overall = {
        f"hit{k}_delta": float(np.nanmean(rec[f"content_hit{k}"]) - np.nanmean(rec[f"cooc_hit{k}"]))
        for k in args.ks
    }
    out = {
        "dataset": args.dataset, "n_items": n_items, "n_test": len(obs), "n_records": int(rec_bin.size),
        "n_bins": args.n_bins, "edges": [float(x) for x in edges], "ks": args.ks,
        "epochs": args.epochs, "seed": args.seed,
        "overall": overall, "per_bin": per_bin,
        "verdict_inputs": {"hit1_delta_monotone_decreasing": bool(monotone),
                           "top_bin_hit1_delta": top["delta"], "top_bin_hit1_p": top["p_value"]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"dataset": args.dataset, "n_items": n_items, "n_records": out["n_records"],
                      "overall": overall, "verdict_inputs": out["verdict_inputs"]}, indent=2))


if __name__ == "__main__":
    main()
