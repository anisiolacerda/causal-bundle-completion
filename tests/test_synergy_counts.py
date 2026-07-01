import numpy as np
from synergy.counts import (
    item_bundle_incidence, bundles_with, counts_over_j, cond_prob, cooc_score,
    ppmi_score,
)

# 4 items, 4 bundles. item 3 appears ONLY with {0,1} together.
BUNDLES = [[0, 1, 3], [0, 1, 3], [0, 2], [1, 2]]
N_ITEMS = 4

def test_incidence_shape_and_entries():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    assert B.shape == (4, 4)
    assert np.array_equal(np.asarray(B[3].todense()).ravel(), [1, 1, 0, 0])  # item 3 in bundles 0,1

def test_bundles_with_intersection():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    assert list(bundles_with(B, [0, 1])) == [0, 1]     # bundles containing BOTH 0 and 1
    assert list(bundles_with(B, [2])) == [2, 3]
    assert list(bundles_with(B, [])) == [0, 1, 2, 3]   # empty -> all bundles

def test_counts_over_j():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    # over bundles {0,1} (which contain items 0,1,3), item counts:
    c = counts_over_j(B, np.array([0, 1]))
    assert list(c) == [2, 2, 0, 2]

def test_cond_prob_laplace():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    # P(3 | {0,1}) : bundles with {0,1} = 2, both contain 3 -> (2+1)/(2+4*1)? use +1/+n_items smoothing
    p = cond_prob(B, [0, 1], n_bundles=4)
    # smoothing: (count_j + 1) / (n_cond + n_items)
    assert abs(p[3] - (2 + 1) / (2 + 4)) < 1e-9
    # marginal P(j): item 0 in 3 of 4 bundles -> (3+1)/(4+4)
    pm = cond_prob(B, [], n_bundles=4)
    assert abs(pm[0] - (3 + 1) / (4 + 4)) < 1e-9

def test_cooc_score_masks_observed():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    s = cooc_score(B, [0, 1], N_ITEMS)
    assert s[0] == -np.inf and s[1] == -np.inf   # observed masked
    assert s[3] > s[2]                            # 3 co-occurs with {0,1} more than 2 does

def test_ppmi_score_masks_observed_and_orders():
    B = item_bundle_incidence(BUNDLES, N_ITEMS)
    s = ppmi_score(B, [0, 1], N_ITEMS)
    assert s[0] == -np.inf                        # observed masked
    assert s[3] > s[2]                            # item 3 co-occurs with {0,1} more than item 2
