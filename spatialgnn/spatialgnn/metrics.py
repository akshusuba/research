"""Node-classification metrics: accuracy, macro-F1, and ARI.

Macro-F1 is reported because spatial domains are often imbalanced; ARI
(adjusted Rand index) measures agreement of the predicted partition with the
ground-truth domains the way the spatial-omics field evaluates domain methods.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }
