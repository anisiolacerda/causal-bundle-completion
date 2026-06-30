# scripts/cec_m05_exposure_premise.py
"""CEC M0.5: re-score Spotify-90 (content vs co-occ), stratify per-gt-item ranks by item exposure.

Run (predecessor venv + its src on PYTHONPATH):
  PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
  /Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
  scripts/cec_m05_exposure_premise.py \
    --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/spotify90 \
    --ckpt /Users/anisiomlacerda/code/Bundle_construction/models/setcompleter_spotify90.pt \
    --n-items 254155 --n-bins 5 --out results/cec/m05/spotify90.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.sparse import csr_matrix
import torch

from causalbr.diffusion.bundle_dataset import load_bundles
from causalbr.recovery.set_completer import setcompleter_full_scores, train_set_completer
from causalbr.eval.significance import paired_wilcoxon

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cec.exposure import item_exposure, exposure_bin_edges, assign_bin
from cec.stratify import item_rank, per_bin_paired_stats


def item_bundle_incidence(bundles, n_items):
    """Item x bundle incidence matrix B (csr): B[i, b] = 1 iff item i in bundle b.

    Memory-safe replacement for materializing the n_items x n_items co-occurrence
    matrix: B has only sum(|bundle|) nonzeros (~1.27M for Spotify-90), vs ~78M
    item-pairs for the dense double-loop (which OOMs on the 254k catalog).
    """
    rows, cols = [], []
    for b_idx, b in enumerate(bundles):
        for it in b:
            if 0 <= it < n_items:
                rows.append(it); cols.append(b_idx)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_items, len(bundles)))


def cooc_scores(B, observed, n_items):
    """Co-occurrence completion score per item, via a sparse mat-vec (no n_items^2 matrix).

    cs[j] = sum_{o in observed} cooc(o, j) for non-observed j, where
    cooc(o, j) = #bundles containing both o and j. Equivalent to
    B @ (B[observed].sum(axis=0)): for each bundle, overlap = #observed items in it,
    propagated to every item in that bundle. For non-observed j this equals the
    diagonal-zeroed co-occurrence sum exactly (the only self-term, o == j, requires
    j observed, which is masked below). Observed items masked to -inf.
    """
    if not observed:
        return np.full(n_items, -np.inf)
    overlap = np.asarray(B[observed].sum(axis=0)).ravel()  # (n_bundles,)
    cs = np.asarray(B @ overlap).ravel().astype(float)      # (n_items,)
    cs[observed] = -np.inf
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_items = args.n_items
    train = load_bundles(os.path.join(args.data_dir, "train_90.txt"))
    obs = load_bundles(os.path.join(args.data_dir, "test_input_90.txt"))
    gts = load_bundles(os.path.join(args.data_dir, "test_gt_90.txt"))
    assert len(obs) == len(gts), f"obs/gt mismatch {len(obs)} vs {len(gts)}"

    exposure = item_exposure(train, n_items)
    edges = exposure_bin_edges(exposure, n_bins=args.n_bins)
    B = item_bundle_incidence(train, n_items)

    model = torch.load(args.ckpt, weights_only=False)
    model.eval()

    # records keyed by gt item: exposure-bin + per-method hit@k / rank
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

    # verdict inputs: monotone-decreasing delta over bins (low->high exposure) on hit@1
    d1 = [b["delta"] for b in per_bin["hit1"]]
    monotone = all((d1[j] + 1e-9) >= d1[j + 1] for j in range(len(d1) - 1))
    top = per_bin["hit1"][-1]
    overall = {
        f"hit{k}_delta": float(np.nanmean(rec[f"content_hit{k}"]) - np.nanmean(rec[f"cooc_hit{k}"]))
        for k in args.ks
    }
    out = {
        "dataset": "spotify90", "n_test": len(obs), "n_records": int(rec_bin.size),
        "n_bins": args.n_bins, "edges": [float(x) for x in edges], "ks": args.ks,
        "overall": overall, "per_bin": per_bin,
        "verdict_inputs": {"hit1_delta_monotone_decreasing": bool(monotone),
                           "top_bin_hit1_delta": top["delta"], "top_bin_hit1_p": top["p_value"]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"overall": overall, "verdict_inputs": out["verdict_inputs"]}, indent=2))


if __name__ == "__main__":
    main()
