"""Patient-level train/val/test splits (no patient appears in two folds)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import SplitConfig


@dataclass
class CohortSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    num_classes: int
    num_features: int


def make_split(ds, cfg: SplitConfig) -> CohortSplit:
    rng = np.random.default_rng(cfg.seed)
    n = len(ds.graphs)
    perm = rng.permutation(n)
    n_test = int(cfg.test_frac * n)
    n_val = int(cfg.val_frac * n)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return CohortSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                       num_classes=ds.num_classes, num_features=ds.num_features)
