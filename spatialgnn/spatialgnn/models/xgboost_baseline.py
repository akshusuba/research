"""XGBoost baseline -- the strong tabular, structure-blind competitor.

Trains gradient-boosted trees on each cell's expression vector (no graph). This
is the baseline the SRI winning formula explicitly asks for; beating it shows
the spatial graph adds value beyond what a powerful tabular learner extracts
from per-cell expression. Returns metrics in the same format as the torch path.
"""

from __future__ import annotations

import numpy as np

from ..metrics import classification_metrics


def train_xgboost(split, seed: int = 0, n_estimators: int = 400,
                  max_depth: int = 6, lr: float = 0.1) -> dict:
    from xgboost import XGBClassifier

    x = split.x.cpu().numpy()
    y = split.y.cpu().numpy()
    tr, va, te = split.train_mask, split.val_mask, split.test_mask

    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        n_jobs=0, random_state=seed,
    )
    clf.fit(x[tr.cpu().numpy()], y[tr.cpu().numpy()])

    def ev(mask):
        m = mask.cpu().numpy()
        pred = clf.predict(x[m])
        return classification_metrics(y[m], pred)

    return {
        "model": "xgboost", "best_epoch": -1,
        "val": ev(va), "test": ev(te), "history": {},
    }
