"""Learned fusion of complementarity signals for bundle completion (PIC-Diff recovery, Stage 4).

A small logistic combiner over per-candidate features [co-occurrence, content-reranker, log-popularity]
(standardized). It learns to weight CF (recall) + content (top-1 precision) + popularity, so it can
recover co-occurrence-only where content does not help and add content where it does — the path to a
single model that beats raw co-occurrence on the full catalog. Pure torch/numpy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def standardize_fit(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean/std for z-scoring (std floored at 1e-6)."""
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def fit_fusion(
    feat_pos: np.ndarray, feat_neg: np.ndarray, *, epochs: int = 200, lr: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Logistic combiner: gt candidates (feat_pos) vs pool negatives (feat_neg).

    Returns (w, b, mu, sd) — apply as sigmoid(((feat - mu)/sd) @ w + b). The learned w shows how much
    each signal (cooc / content / log-pop) is used.
    """
    feats = np.vstack([feat_pos, feat_neg])
    mu, sd = standardize_fit(feats)
    x = torch.as_tensor((feats - mu) / sd, dtype=torch.float32)
    y = torch.as_tensor(
        np.concatenate([np.ones(len(feat_pos)), np.zeros(len(feat_neg))]), dtype=torch.float32
    )
    torch.manual_seed(seed)
    lin = nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    # class weight: gt is rare (1 per pool) — upweight positives
    pos_weight = torch.tensor([len(feat_neg) / max(1, len(feat_pos))])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(lin(x).squeeze(1), y).backward()
        opt.step()
    w = lin.weight.detach().numpy().ravel()
    b = float(lin.bias.detach())
    return w, b, mu, sd


def fuse_scores(feats: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply the learned combiner to a (..., F) feature array -> scores (...,)."""
    return ((feats - mu) / sd) @ w
