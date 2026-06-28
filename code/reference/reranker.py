"""Complementarity-aware decode reranker (PIC-Diff recovery, Stage 2).

The Stage-1 oracle showed the bundle-completion signal lives in the SEED PAIR (the observed
courses), not in the deconfounded theme. This module learns a seed-pair -> completion scorer
that replaces
DDBC's theme-blind cosine-NN decode argmax, and the co-occurrence baseline it must beat to count as
more than collaborative filtering.

- ComplementarityReranker: g(seedpair) -> query in item-embedding space; score_i = <g(sp), e_hat_i>.
  Trained with listwise cross-entropy over the rho-pool (target = gt). Item embeddings frozen.
- cooccurrence_scores: P(candidate | observed items) from train item-item co-occurrence (the CF
  baseline). The reranker must beat THIS to be a real contribution, not just CF.

Pure torch/numpy; no ranking-era imports.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class ComplementarityReranker(nn.Module):
    """seedpair (2*d) [+ optional free user embedding] -> query (d); score vs unit pool embeddings.

    A `num_users` > 0 adds a learned per-user embedding concatenated to the seed pair (the fairest
    last shot for personalization): if even a free user embedding does not beat the seedpair-only
    reranker on held-out, the user signal is exhausted.
    """

    def __init__(
        self, seedpair_dim: int, emb_dim: int, hidden: int = 128,
        num_users: int = 0, user_dim: int = 16,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_dim) if num_users > 0 else None
        in_dim = seedpair_dim + (user_dim if num_users > 0 else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
        )

    def forward(
        self, seedpair: Tensor, pool_unit: Tensor, user_ids: Tensor | None = None
    ) -> Tensor:
        """seedpair (B, 2d), pool_unit (B, rho, d) unit-norm -> scores (B, rho)."""
        feat = seedpair
        if self.user_emb is not None and user_ids is not None:
            feat = torch.cat([seedpair, self.user_emb(user_ids)], dim=-1)
        q = self.net(feat)  # (B, d)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.einsum("bd,brd->br", q, pool_unit)


def train_reranker(
    seedpair: np.ndarray,
    pool_unit: np.ndarray,
    gt_local: np.ndarray,
    *,
    emb_dim: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
    user_ids: np.ndarray | None = None,
    num_users: int = 0,
) -> ComplementarityReranker:
    """Train the reranker with listwise CE over the pool (target = gt slot)."""
    torch.manual_seed(seed)
    model = ComplementarityReranker(seedpair.shape[1], emb_dim, num_users=num_users).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sp = torch.as_tensor(seedpair, dtype=torch.float32, device=device)
    pu = torch.as_tensor(pool_unit, dtype=torch.float32, device=device)
    y = torch.as_tensor(gt_local, dtype=torch.long, device=device)
    uid = None if user_ids is None else torch.as_tensor(user_ids, dtype=torch.long, device=device)
    n = sp.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for s in range(0, n, batch_size):
            bi = perm[s: s + batch_size]
            logits = model(sp[bi], pu[bi], None if uid is None else uid[bi])
            loss = nn.functional.cross_entropy(logits, y[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def reranker_hit1(
    model: ComplementarityReranker, seedpair: np.ndarray, pool_unit: np.ndarray,
    gt_local: np.ndarray, *, device: str = "cpu", user_ids: np.ndarray | None = None,
) -> float:
    sp = torch.as_tensor(seedpair, dtype=torch.float32, device=device)
    pu = torch.as_tensor(pool_unit, dtype=torch.float32, device=device)
    uid = None if user_ids is None else torch.as_tensor(user_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(sp, pu, uid).argmax(dim=1).cpu().numpy()
    return float(np.mean(pred == gt_local))


def cooccurrence_counts(train_bundles: list[list[int]], n_items: int) -> dict[int, dict[int, int]]:
    """Sparse item-item co-occurrence counts over training bundles."""
    co: dict[int, dict[int, int]] = {}
    for items in train_bundles:
        for a in items:
            row = co.setdefault(int(a), {})
            for b in items:
                if a != b:
                    row[int(b)] = row.get(int(b), 0) + 1
    return co


def cooccurrence_hit1(
    co: dict[int, dict[int, int]], observed: list[list[int]], pool_ids: np.ndarray,
    gt_local: np.ndarray,
) -> float:
    """Score each pool candidate by summed co-occurrence with the observed items; hit@1."""
    hits = 0
    for i, obs in enumerate(observed):
        scores = np.zeros(pool_ids.shape[1])
        for j, cid in enumerate(pool_ids[i]):
            scores[j] = sum(co.get(int(o), {}).get(int(cid), 0) for o in obs)
        if int(scores.argmax()) == gt_local[i]:
            hits += 1
    return hits / len(observed)
