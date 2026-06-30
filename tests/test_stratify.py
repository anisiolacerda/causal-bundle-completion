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
