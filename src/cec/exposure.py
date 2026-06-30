from __future__ import annotations
import numpy as np


def item_exposure(train_bundles: list[list[int]], n_items: int) -> np.ndarray:
    """Per-item train occurrence count (exposure proxy). Matches causalbr item_frequency."""
    freq = np.zeros(n_items, dtype=np.int64)
    for bundle in train_bundles:
        for item in bundle:
            if 0 <= item < n_items:
                freq[item] += 1
    return freq


def exposure_bin_edges(exposure: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """n_bins+1 percentile edges over present (exposure>0) items; lowest edge forced to 0."""
    present = np.asarray(exposure)[np.asarray(exposure) > 0]
    if present.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(present, qs).astype(float)
    edges[0] = 0.0
    return edges


def assign_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, n_bins-1] via np.digitize on interior edges."""
    interior = edges[1:-1]
    return np.clip(np.digitize(np.asarray(values), interior, right=False), 0, len(edges) - 2)
