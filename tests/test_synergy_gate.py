import numpy as np
from synergy.gate import run_gate

def test_gate_detects_synergy_on_interaction_data():
    # gt item 3 appears IFF both observed items {0,1}; co-occ alone under-ranks it vs item 2 noise
    train = [[0, 1, 3]] * 40 + [[0, 2]] * 40 + [[1, 2]] * 40 + [[2, 3]] * 5
    test_obs = [[0, 1]] * 10
    test_gt = [[3]] * 10
    rng = np.random.default_rng(0)
    out = run_gate(train, test_obs, test_gt, n_items=4, ks=(1, 2), topk=3, rng=rng)
    assert out["n_instances"] == 10
    assert out["tau_abs_mean"] > out["tau_abs_mean_null"]       # residual above additivity null
    assert set(out["hit"].keys()) == {1, 2}
    assert len(out["hit"][1]["cooc"]) == 10                     # per-instance paired arrays present

def test_gate_skips_instances_with_fewer_than_two_observed():
    train = [[0, 1, 3]] * 10
    out = run_gate(train, [[0]], [[3]], n_items=4, ks=(1,), topk=3, rng=np.random.default_rng(0))
    assert out["n_instances"] == 0
