"""Link-prediction metrics: AUROC, AUPRC, plus ranking metrics Hits@K and MRR.

Ranking metrics treat each positive test pair against a set of sampled
negatives, mirroring how a wet-lab would triage a shortlist of candidate SL
partners.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def basic_scores(y_true, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {}
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    return out


def ranking_scores(pos_scores, neg_scores, ks=(1, 3, 10)) -> dict:
    """For each positive, rank it against ALL negatives; report Hits@K & MRR.

    This is the standard "1 positive vs N negatives" filtered-ranking style,
    approximated by ranking every positive against the shared negative pool.
    """
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {f"hits@{k}": float("nan") for k in ks} | {"mrr": float("nan")}

    neg_sorted = np.sort(neg_scores)[::-1]
    # rank = 1 + (#negatives scoring higher than the positive)
    ranks = len(neg_sorted) - np.searchsorted(neg_sorted[::-1], pos_scores,
                                              side="left")
    ranks = ranks + 1  # 1-indexed rank of the positive among negatives
    out = {f"hits@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    return out


def all_metrics(y_true, y_score, ks=(1, 3, 10)) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = basic_scores(y_true, y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    out.update(ranking_scores(pos, neg, ks=ks))
    return out
