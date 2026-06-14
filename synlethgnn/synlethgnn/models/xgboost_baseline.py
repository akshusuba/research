"""XGBoost baseline -- the strong tabular, structure-blind competitor.

This is the baseline the SRI winning formula explicitly asks for. It is a
gradient-boosted tree ensemble trained on *pair features* built only from the
two genes' node features (the same symmetric representation the MLP decoder
sees). It never touches the graph, so beating it demonstrates that message
passing adds value beyond what a powerful tabular learner extracts from the
features alone. It does not fit the torch training loop, so it has its own
train/evaluate function returning the same metrics dict format.
"""

from __future__ import annotations

import numpy as np
import torch

from ..metrics import all_metrics


def _pair_features(x: torch.Tensor, pairs: torch.Tensor) -> np.ndarray:
    """Symmetric pair representation: [x_i + x_j , |x_i - x_j|]."""
    xi = x[pairs[0]]
    xj = x[pairs[1]]
    feat = torch.cat([xi + xj, (xi - xj).abs()], dim=-1)
    return feat.cpu().numpy()


def train_xgboost(split, seed: int = 0, n_estimators: int = 400,
                  max_depth: int = 6, lr: float = 0.1) -> dict:
    """Train XGBoost on pair features; return metrics in the standard format."""
    from xgboost import XGBClassifier

    x = split.x
    Xtr = _pair_features(x, split.train_pairs)
    ytr = split.train_labels.cpu().numpy()
    Xva = _pair_features(x, split.val_pairs)
    yva = split.val_labels.cpu().numpy()
    Xte = _pair_features(x, split.test_pairs)
    yte = split.test_labels.cpu().numpy()

    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        tree_method="hist", n_jobs=0, random_state=seed,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    def scores(X):
        return clf.predict_proba(X)[:, 1]

    return {
        "model": "xgboost",
        "best_epoch": -1,
        "val": all_metrics(yva, scores(Xva)),
        "test": all_metrics(yte, scores(Xte)),
        "history": {},
    }
