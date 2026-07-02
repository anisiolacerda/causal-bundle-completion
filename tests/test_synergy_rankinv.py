import numpy as np
from synergy.rankinv import item_rank, hit_at_k, rank_agreement

def test_item_rank_and_hit():
    s = np.array([0.9, 0.5, 0.9, 0.1])
    assert item_rank(s, 3) == 4.0
    assert item_rank(s, 0) == 1.5       # tied top -> average-tie rank 1.5
    assert hit_at_k(s, 0, 1) == 0.0     # avg-tie rank 1.5 is NOT <= 1 (conservative, matches cec.stratify)
    assert hit_at_k(s, 0, 2) == 1.0     # 1.5 <= 2
    assert hit_at_k(s, 3, 1) == 0.0

def test_rank_agreement_identical_is_one():
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([6.0, 4.0, 2.0, 0.0])   # monotone reweight of a -> same ranking
    cand = np.array([0, 1, 2, 3])
    r = rank_agreement(a, b, cand, topk=2)
    assert abs(r["spearman"] - 1.0) < 1e-9
    assert abs(r["kendall"] - 1.0) < 1e-9
    assert r["topk_jaccard"] == 1.0      # same top-2 => inert reweight

def test_rank_agreement_reversed_is_negative():
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])   # reversed
    cand = np.array([0, 1, 2, 3])
    r = rank_agreement(a, b, cand, topk=2)
    assert r["spearman"] < -0.9
    assert r["topk_jaccard"] == 0.0      # disjoint top-2 => strong rank inversion
