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
