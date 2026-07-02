# tests/test_synergy_interaction.py
import numpy as np
from synergy.counts import item_bundle_incidence
from synergy.interaction import synergy_vectors, null_bundles_product_of_marginals

def test_pure_interaction_gives_positive_tau():
    # item 3 appears IFF both 0 and 1 present; never with only one of them.
    bundles = [[0, 1, 3]] * 20 + [[0, 2]] * 20 + [[1, 2]] * 20
    n_items = 4
    B = item_bundle_incidence(bundles, n_items)
    v = synergy_vectors(B, (0, 1), n_bundles=len(bundles))
    # synergy for item 3 is strongly positive: joint high, but P(3|0)=P(3|1)~1/2, additive underestimates
    assert v["tau"][3] > 0.15
    # joint conditional ranks item 3 top among non-observed candidates (2 and 3)
    assert v["joint"][3] > v["joint"][2]

def test_additive_data_gives_near_zero_tau():
    # items 2 and 3 each appear independently of the (0,1) co-presence -> tau ~ 0
    rng = np.random.default_rng(0)
    bundles = []
    for _ in range(400):
        b = [0, 1]
        if rng.random() < 0.5: b.append(2)
        if rng.random() < 0.5: b.append(3)
        bundles.append(b)
    n_items = 4
    B = item_bundle_incidence(bundles, n_items)
    v = synergy_vectors(B, (0, 1), n_bundles=len(bundles))
    assert abs(v["tau"][2]) < 0.08 and abs(v["tau"][3]) < 0.08

def test_null_bundles_preserve_sizes_and_are_additive():
    rng = np.random.default_rng(1)
    bundles = [[0, 1, 3]] * 20 + [[0, 2]] * 20 + [[1, 2]] * 20
    null = null_bundles_product_of_marginals(bundles, n_items=4, sizes=[len(b) for b in bundles], rng=rng)
    assert [len(b) for b in null] == [len(b) for b in bundles]     # sizes preserved
    B = item_bundle_incidence(null, 4)
    v = synergy_vectors(B, (0, 1), n_bundles=len(null))
    # on additive/independent null data the interaction estimate is small
    assert abs(v["tau"][3]) < 0.12
