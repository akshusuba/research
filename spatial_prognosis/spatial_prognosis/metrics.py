"""Patient-level classification metrics: accuracy, macro-F1, AUROC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
    return out
