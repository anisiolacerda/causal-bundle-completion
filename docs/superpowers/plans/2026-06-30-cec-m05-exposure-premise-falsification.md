# CEC M0.5 — Exposure-Stratified Premise Falsification (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Decide, at near-zero cost, whether the predecessor's existing *content (SetCompleter) beats
co-occurrence* full-catalog win on Spotify-90 **concentrates on low-exposure (cold) items** — the central
CEC premise — by re-scoring with the existing checkpoint + data and stratifying per-target-item ranks by
item-exposure percentile. Ends at a pre-registered go/no-go verdict.

**Architecture:** A small, unit-tested analysis package `cec` in *this* repo (pure functions for exposure
binning + per-bin paired deltas), plus one runner script that imports the predecessor package `causalbr`
(metrics, co-occ, SetCompleter, gate) to load Spotify-90 and the `setcompleter_spotify90.pt` checkpoint,
computes full-catalog scores for co-occ and SetCompleter, builds per-(example, gt-item) records tagged with
the gt item's exposure, runs the analysis, and writes a verdict table + warm/cold curve. **No retrain, no
port, no modification of the predecessor repo.**

**Tech Stack:** Python ≥3.10, numpy, scipy, torch (from the predecessor venv), pytest. Predecessor package
`causalbr` at `/Users/anisiomlacerda/code/Bundle_construction/src` (import via `PYTHONPATH`).

## Global Constraints

- **Reuse, don't re-derive:** import metrics/co-occ/SetCompleter/significance/exposure from `causalbr`; only
  the *binning + per-bin stratified-delta* logic is new code in this repo.
- **Full-catalog only** — score over all `n_items` (254,155 for spotify90); observed seed items masked to −∞.
- **Per-example paired tests** — paired Wilcoxon (`causalbr.eval.significance.paired_wilcoxon`, alternative
  `"greater"`), per exposure bin.
- **Predecessor is UNPUBLISHED — do not cite it.** Its code/data/checkpoints are inputs only.
- **Pre-registered verdict (before looking at numbers):** CEC's premise is CONFIRMED iff the content−co-occ
  hit@1 (and recall@20) **delta is positive and largest in the lowest exposure bins and ≈0 (within paired
  significance) in the highest exposure bin** — i.e. a monotone-decreasing delta-vs-exposure curve. A flat or
  uniform delta REFUTES the premise (content's win is not a cold-item effect) → CEC does not proceed to M1.
- **Run command prefix** (every runner/test invocation that needs `causalbr`):
  `PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src /Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python`
  (the predecessor venv has torch+scipy+causalbr; this repo has no venv). Refer to this as `$PYBC` below.

## File Structure

- `pyproject.toml` — minimal package config for `cec` (this repo), pytest settings.
- `src/cec/__init__.py` — package marker.
- `src/cec/exposure.py` — NEW pure functions: `item_exposure`, `exposure_bin_edges`, `assign_bin`.
- `src/cec/stratify.py` — NEW pure functions: `item_rank`, `per_bin_paired_stats`.
- `scripts/cec_m05_exposure_premise.py` — runner (imports `causalbr`; real-data integration).
- `tests/test_exposure.py`, `tests/test_stratify.py` — unit tests (pure, no `causalbr` needed).
- `results/cec/m05/spotify90.json` — emitted verdict table + curve (runner output).
- `docs/findings/2026-06-30-cec-m05-premise.md` — the written go/no-go verdict.

---

### Task 1: Scaffold the `cec` package + pytest

**Files:**
- Create: `pyproject.toml`, `src/cec/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: an importable `cec` package and a working `pytest` so later pure-function tasks are testable
  without the predecessor venv.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_smoke.py
import cec

def test_package_imports():
    assert cec.__name__ == "cec"
```

- [ ] **Step 2: Create the package marker**

```python
# src/cec/__init__.py
"""CEC — counterfactual exposure-balanced completion (analysis package)."""
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cec"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "scipy>=1.11"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]
addopts = "-v --tb=short"
```

- [ ] **Step 4: Run the smoke test (this repo's python; numpy/scipy/pytest required)**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS. (If pytest/numpy are missing in this repo's interpreter, run with the predecessor venv:
`$PYBC -m pytest tests/test_smoke.py -v` — it has both.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cec/__init__.py tests/test_smoke.py
git commit -m "chore(cec): scaffold cec analysis package + pytest"
```

---

### Task 2: Exposure binning (pure)

**Files:**
- Create: `src/cec/exposure.py`
- Test: `tests/test_exposure.py`

**Interfaces:**
- Produces:
  - `item_exposure(train_bundles: list[list[int]], n_items: int) -> np.ndarray` — per-item train count
    (matches `causalbr.causal.gate.item_frequency`; reimplemented here so unit tests need no predecessor).
  - `exposure_bin_edges(exposure: np.ndarray, n_bins: int = 5) -> np.ndarray` — `n_bins+1` percentile edges
    over **present** items (exposure>0); lowest edge forced to 0 so zero-exposure items fall in bin 0.
  - `assign_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray` — integer bin in `[0, n_bins-1]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_exposure.py
import numpy as np
from cec.exposure import item_exposure, exposure_bin_edges, assign_bin

def test_item_exposure_counts():
    bundles = [[0, 1, 2], [1, 2], [2]]
    exp = item_exposure(bundles, n_items=4)
    assert exp.tolist() == [1, 2, 3, 0]

def test_bin_edges_monotone_and_zero_floor():
    exp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    edges = exposure_bin_edges(exp, n_bins=5)
    assert len(edges) == 6
    assert edges[0] == 0.0
    assert np.all(np.diff(edges) >= 0)

def test_assign_bin_low_high():
    exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    edges = exposure_bin_edges(exp, n_bins=5)
    bins = assign_bin(exp, edges)
    assert bins.min() == 0 and bins.max() == 4
    # higher exposure -> not-lower bin
    assert assign_bin(np.array([1]), edges)[0] <= assign_bin(np.array([10]), edges)[0]

def test_zero_exposure_in_bin_zero():
    exp = np.array([0, 0, 5, 10, 20])
    edges = exposure_bin_edges(exp[exp > 0], n_bins=2)
    assert assign_bin(np.array([0]), edges)[0] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_exposure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cec.exposure'`

- [ ] **Step 3: Implement**

```python
# src/cec/exposure.py
from __future__ import annotations
import numpy as np


def item_exposure(train_bundles: list[list[int]], n_items: int) -> np.ndarray:
    """Per-item train occurrence count (exposure proxy). Matches causalbr item_frequency."""
    freq = np.zeros(n_items, dtype=np.int64)
    for bundle in train_bundles:
        for item in bundle:
            if 0 <= item < n_items:
                freq[item] += 1
    return freq


def exposure_bin_edges(exposure: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """n_bins+1 percentile edges over present (exposure>0) items; lowest edge forced to 0."""
    present = np.asarray(exposure)[np.asarray(exposure) > 0]
    if present.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(present, qs).astype(float)
    edges[0] = 0.0
    return edges


def assign_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, n_bins-1] via np.digitize on interior edges."""
    interior = edges[1:-1]
    return np.clip(np.digitize(np.asarray(values), interior, right=False), 0, len(edges) - 2)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_exposure.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cec/exposure.py tests/test_exposure.py
git commit -m "feat(cec): exposure percentile binning (pure)"
```

---

### Task 3: Per-record rank + per-bin paired stats (pure)

**Files:**
- Create: `src/cec/stratify.py`
- Test: `tests/test_stratify.py`

**Interfaces:**
- Consumes: nothing from `causalbr` at import time (the significance function is **injected** so unit tests
  stay self-contained).
- Produces:
  - `item_rank(scores: np.ndarray, item: int) -> float` — 1-based average-tie rank of `item` in `scores`
    (higher score = better). Matches the convention of `causalbr.recovery.ranking_metrics.gt_ranks`.
  - `per_bin_paired_stats(metric_a, metric_b, bin_idx, n_bins, sig_fn=None) -> list[dict]` — per bin:
    `{"bin","n","mean_a","mean_b","delta","p_value"}`. `metric_a/_b` are per-record metrics (e.g. hit@1 of
    the gt item under content vs co-occ); `delta = mean_a − mean_b`; `p_value` from `sig_fn(a, b)` if given.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stratify.py
import numpy as np
from cec.stratify import item_rank, per_bin_paired_stats

def test_item_rank_top_and_ties():
    scores = np.array([0.9, 0.5, 0.9, 0.1])
    assert item_rank(scores, 3) == 4.0          # lowest score -> rank 4
    # two tied top scores (idx 0 and 2): average-tie rank = 1.5
    assert item_rank(scores, 0) == 1.5
    assert item_rank(scores, 2) == 1.5

def test_per_bin_delta_and_counts():
    a = np.array([1.0, 1.0, 0.0, 0.0])   # content hit@1
    b = np.array([0.0, 0.0, 0.0, 0.0])   # cooc hit@1
    bins = np.array([0, 0, 1, 1])
    stats = per_bin_paired_stats(a, b, bins, n_bins=2)
    assert stats[0]["n"] == 2 and stats[0]["delta"] == 1.0
    assert stats[1]["n"] == 2 and stats[1]["delta"] == 0.0

def test_per_bin_uses_sig_fn():
    a = np.array([1.0, 1.0]); b = np.array([0.0, 0.0]); bins = np.array([0, 0])
    called = {}
    def fake_sig(x, y):
        called["seen"] = (list(x), list(y))
        return {"p_value": 0.03}
    stats = per_bin_paired_stats(a, b, bins, n_bins=1, sig_fn=fake_sig)
    assert stats[0]["p_value"] == 0.03
    assert called["seen"] == ([1.0, 1.0], [0.0, 0.0])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_stratify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cec.stratify'`

- [ ] **Step 3: Implement**

```python
# src/cec/stratify.py
from __future__ import annotations
from typing import Callable
import numpy as np


def item_rank(scores: np.ndarray, item: int) -> float:
    """1-based average-tie rank of `item` (higher score = better)."""
    s = np.asarray(scores, dtype=float)
    sv = s[item]
    greater = int(np.sum(s > sv))
    ties = int(np.sum(s == sv))  # includes item itself
    return float(greater + (ties + 1) / 2.0)


def per_bin_paired_stats(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
    sig_fn: Callable[[list, list], dict] | None = None,
) -> list[dict]:
    """Per-bin mean(a), mean(b), delta=mean_a-mean_b, n, and p_value via injected sig_fn."""
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    bi = np.asarray(bin_idx)
    out = []
    for bn in range(n_bins):
        mask = bi == bn
        n = int(mask.sum())
        if n == 0:
            out.append({"bin": bn, "n": 0, "mean_a": float("nan"),
                        "mean_b": float("nan"), "delta": float("nan"), "p_value": float("nan")})
            continue
        ma = float(np.nanmean(a[mask]))
        mb = float(np.nanmean(b[mask]))
        p = float("nan")
        if sig_fn is not None:
            p = float(sig_fn(list(a[mask]), list(b[mask])).get("p_value", float("nan")))
        out.append({"bin": bn, "n": n, "mean_a": ma, "mean_b": mb, "delta": ma - mb, "p_value": p})
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_stratify.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cec/stratify.py tests/test_stratify.py
git commit -m "feat(cec): per-record rank + per-bin paired stratified stats (pure)"
```

---

### Task 4: The runner — real-data exposure-stratified re-score (integration)

**Files:**
- Create: `scripts/cec_m05_exposure_premise.py`

**Interfaces:**
- Consumes (from `causalbr`, predecessor):
  - `causalbr.diffusion.bundle_dataset.load_bundles(path: str) -> list[list[int]]`
  - co-occ: build with `causalbr.recovery.reranker.cooccurrence_counts` is the dict form; for full-catalog
    use the sparse matrix pattern — replicate the stage-script `cooc_sparse` inline (below) and score
    `cs = np.asarray(co[obs].sum(axis=0)).ravel()`.
  - `causalbr.recovery.set_completer.setcompleter_full_scores(model, seed_items, n_items, mask_observed=True)`
    and checkpoint load `torch.load(path, weights_only=False) -> SetCompleter`.
  - `causalbr.recovery.ranking_metrics.gt_ranks` is available, but we use our own `cec.stratify.item_rank`
    per gt item.
  - `causalbr.eval.significance.paired_wilcoxon(a, b, alternative="greater")` injected as `sig_fn`.
  - `causalbr.causal.gate.item_frequency` (equivalent to `cec.exposure.item_exposure`; use ours).
- Produces: `results/cec/m05/spotify90.json` with `{n_test, n_records, n_bins, edges, overall:{hit1_delta,
  recall20_delta, wilcoxon_p}, per_bin:[...], verdict_inputs:{monotone_decreasing, top_bin_delta_sig}}`.

- [ ] **Step 1: Write the runner**

```python
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


def cooc_sparse(bundles, n_items):
    rows, cols = [], []
    for b in bundles:
        for x in b:
            for y in b:
                if x != y:
                    rows.append(x); cols.append(y)
    data = np.ones(len(rows), dtype=np.float32)
    co = csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
    co.setdiag(0); co.eliminate_zeros()
    return co


def cooc_scores(co, observed, n_items):
    cs = np.asarray(co[observed].sum(axis=0)).ravel().astype(float)
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
    co = cooc_sparse(train, n_items)

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
            cc = cooc_scores(co, seed, n_items)
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
```

- [ ] **Step 2: Dry-run argument parsing (no heavy compute)**

Run: `$PYBC scripts/cec_m05_exposure_premise.py --help`
Expected: argparse help prints (confirms `causalbr` imports resolve under `$PYBC`). If an import fails, fix
the import path before proceeding (do NOT stub it out).

- [ ] **Step 3: Commit the runner**

```bash
git add scripts/cec_m05_exposure_premise.py
git commit -m "feat(cec): M0.5 exposure-stratified re-score runner"
```

---

### Task 5: Execute on Spotify-90 + write the go/no-go verdict

**Files:**
- Create: `results/cec/m05/spotify90.json` (runner output), `docs/findings/2026-06-30-cec-m05-premise.md`

**Interfaces:**
- Consumes: Task 4 runner + Task 2/3 pure functions; predecessor data `data/spotify90/{train_90,
  test_input_90,test_gt_90}.txt` and checkpoint `models/setcompleter_spotify90.pt`.
- Produces: the verdict doc with the per-bin table and the GO / NO-GO decision per the pre-registered rule.

- [ ] **Step 1: Run the re-score on Spotify-90**

Run:
```bash
PYTHONPATH=/Users/anisiomlacerda/code/Bundle_construction/src \
/Users/anisiomlacerda/code/Bundle_construction/.venv/bin/python \
scripts/cec_m05_exposure_premise.py \
  --data-dir /Users/anisiomlacerda/code/Bundle_construction/data/spotify90 \
  --ckpt /Users/anisiomlacerda/code/Bundle_construction/models/setcompleter_spotify90.pt \
  --n-items 254155 --n-bins 5 --ks 1 20 \
  --out results/cec/m05/spotify90.json
```
Expected: prints `overall` (positive `hit1_delta` / `hit20_delta` — content beats co-occ overall, reproducing
F3a) and `verdict_inputs`; writes `results/cec/m05/spotify90.json`. Runtime = minutes (full-catalog scoring
over the test set; CPU ok).
**Fallback if the checkpoint fails to load/align** (class-path or shape error): retrain in-process by
replacing the `torch.load` line with
`feats = torch.load(os.path.join(args.data_dir,"clhe.pt"), weights_only=False).float().numpy();
model = train_set_completer(train, feats, hidden=128, epochs=20, seed=0)` — deterministic, ~minutes.

- [ ] **Step 2: Sanity-check the output**

Run: `python -c "import json; d=json.load(open('results/cec/m05/spotify90.json')); print(d['overall']); [print(b) for b in d['per_bin']['hit1']]"`
Expected: `overall['hit1_delta'] > 0` (content beats co-occ overall). Inspect the per-bin `delta` column:
the premise predicts the largest positive delta in **bin 0–1** (low exposure) and the smallest near **bin
n-1** (high exposure).

- [ ] **Step 3: Write the verdict doc**

Create `docs/findings/2026-06-30-cec-m05-premise.md` with: (a) the per-bin hit@1 / recall@20 delta table
copied from the JSON; (b) the warm/cold delta curve described in words; (c) the pre-registered decision —
**GO** iff `hit1_delta_monotone_decreasing` is true AND the lowest-bin delta is positive with paired
Wilcoxon `p < 0.05` AND the top (highest-exposure) bin delta is ≈0 / not significant; otherwise **NO-GO**
(content's win is not a cold-item effect → CEC's premise is refuted; do not proceed to M1). State the exact
numbers and the decision plainly (honesty guard: report a refutation as such).

- [ ] **Step 4: Commit results + verdict**

```bash
git add results/cec/m05/spotify90.json docs/findings/2026-06-30-cec-m05-premise.md
git commit -m "feat(cec): M0.5 Spotify-90 exposure-stratified premise verdict"
```

---

### Task 6 (optional robustness): repeat on POG

**Files:**
- Create: `results/cec/m05/pog.json`

- [ ] **Step 1: Run on POG** (smaller catalog; confirms the premise is not Spotify-only)

Run: same command as Task 5 Step 1 with `--data-dir .../data/pog` (confirm the POG construction file names
and `--n-items` from its `count.json` first via
`python -c "import json;print(json.load(open('/Users/anisiomlacerda/code/Bundle_construction/data/pog/count.json')))"`),
`--ckpt .../models/setcompleter_pog.pt`, `--out results/cec/m05/pog.json`.
Expected: same shape of output; note in the verdict doc whether the cold-concentration replicates on POG.

- [ ] **Step 2: Append the POG result to the verdict doc and commit**

```bash
git add results/cec/m05/pog.json docs/findings/2026-06-30-cec-m05-premise.md
git commit -m "feat(cec): M0.5 POG robustness for the exposure-premise verdict"
```

---

## Next plan (gated on this verdict)
If **GO**: the next plan is **M0 (port the `causalbr` harness/baselines/SetCompleter/gate/exposure tooling
into this repo) + M1 (the exposure-gated doubly-robust counterfactual completer)** per the CEC spec §3/§8.
If **NO-GO**: stop; the premise (content's win = a cold-item effect) is refuted — revisit the core bet
(the spec's risk table) rather than building M1.
