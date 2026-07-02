from __future__ import annotations
import numpy as np
from synergy.counts import item_bundle_incidence, cooc_score
from synergy.interaction import synergy_vectors, null_bundles_product_of_marginals
from synergy.rankinv import item_rank, hit_at_k, rank_agreement


def _instances(test_obs, test_gt):
    for obs, gt in zip(test_obs, test_gt):
        if len(obs) >= 2 and gt:
            yield obs[0], obs[1], obs, gt


def _tau_abs_over_candidates(B, obs_pair, n_bundles, topk):
    """mean_j |tau_syn(j)| over the top-`topk` co-occ candidates for this instance."""
    v = synergy_vectors(B, obs_pair, n_bundles)
    cs = cooc_score(B, list(obs_pair), B.shape[0])
    cand = np.argsort(-cs)[:topk]
    tau = v["tau"][cand]
    tau = tau[np.isfinite(tau)]
    return float(np.mean(np.abs(tau))) if tau.size else 0.0, v, cs, cand


def run_gate(train_bundles, test_obs, test_gt, n_items, *, ks=(1, 20), topk=200, rng):
    B = item_bundle_incidence(train_bundles, n_items)
    n_bundles = B.shape[1]
    inst = list(_instances(test_obs, test_gt))

    # placebo (additivity null) incidence, same sizes
    sizes = [len(b) for b in train_bundles]
    null_bundles = null_bundles_product_of_marginals(train_bundles, n_items, sizes, rng)
    Bn = item_bundle_incidence(null_bundles, n_items)

    tau_abs, tau_abs_null = [], []
    sp_list, jac_list, masked = [], [], []
    hit = {k: {"cooc": [], "joint": []} for k in ks}

    for i, k_, obs, gt in inst:
        ta, v, cs, cand = _tau_abs_over_candidates(B, (i, k_), n_bundles, topk)
        tau_abs.append(ta)
        tn, _, _, _ = _tau_abs_over_candidates(Bn, (i, k_), Bn.shape[1], topk)
        tau_abs_null.append(tn)

        joint = v["joint"].copy()          # additive + tau == joint conditional P(j|{i,k})
        ra = rank_agreement(joint, cs, cand, topk=20)
        if np.isfinite(ra["spearman"]):
            sp_list.append(ra["spearman"]); jac_list.append(ra["topk_jaccard"])

        for g in gt:
            for k in ks:
                hit[k]["cooc"].append(hit_at_k(cs, g, k))
                hit[k]["joint"].append(hit_at_k(joint, g, k))
            # masked-complement: gt has high tau but low co-occ rank
            tau_g = v["tau"][g]
            finite_tau = v["tau"][np.isfinite(v["tau"])]
            hi_tau = np.isfinite(tau_g) and tau_g >= np.percentile(finite_tau, 90)
            lo_cooc = item_rank(cs, g) > 100
            masked.append(1.0 if (hi_tau and lo_cooc) else 0.0)

    return {
        "n_instances": len(inst),
        "tau_abs_mean": float(np.mean(tau_abs)) if tau_abs else 0.0,
        "tau_abs_mean_null": float(np.mean(tau_abs_null)) if tau_abs_null else 0.0,
        "rank": {"spearman_mean": float(np.mean(sp_list)) if sp_list else float("nan"),
                 "top20_jaccard_mean": float(np.mean(jac_list)) if jac_list else float("nan")},
        "hit": {k: {"cooc": hit[k]["cooc"], "joint": hit[k]["joint"],
                    "cooc_mean": float(np.mean(hit[k]["cooc"])) if hit[k]["cooc"] else 0.0,
                    "joint_mean": float(np.mean(hit[k]["joint"])) if hit[k]["joint"] else 0.0}
                for k in ks},
        "masked_frac": float(np.mean(masked)) if masked else 0.0,
    }
