import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "synergy_m1_gate",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "synergy_m1_gate.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _metrics(**over):
    m = {"tau_abs_mean": 0.10, "tau_abs_mean_null": 0.02, "tau_perm_p": 0.001,
         "rank": {"spearman_mean": 0.60, "top20_jaccard_mean": 0.50},
         "hit": {20: {"cooc_mean": 0.30, "joint_mean": 0.38, "wilcoxon_p": 0.001}},
         "masked_frac": 0.05, "placebo_ok": True}
    m.update(over); return m

def test_decide_go_when_all_pass():
    d = mod.decide(_metrics())
    assert d["verdict"] == "GO"

def test_decide_kill_when_rank_equivalent():
    d = mod.decide(_metrics(rank={"spearman_mean": 0.95, "top20_jaccard_mean": 0.95}))
    assert d["verdict"] == "KILL"
    assert "rank_equivalent_to_cooc" in d["failed"]

def test_decide_kill_when_residual_at_null():
    d = mod.decide(_metrics(tau_perm_p=0.9))
    assert d["verdict"] == "KILL"
    assert "residual_at_null" in d["failed"]

def test_decide_kill_when_placebo_miscalibrated():
    d = mod.decide(_metrics(placebo_ok=False))
    assert d["verdict"] == "KILL"
    assert "placebo_miscalibrated" in d["failed"]
