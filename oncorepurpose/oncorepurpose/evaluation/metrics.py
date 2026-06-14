"""Link-prediction metrics: AUROC, AUPRC, Hits@K, MRR, F1 at optimal threshold."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = Union[np.ndarray, torch.Tensor, List]


def to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.array(x)
    return x


def hits_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    y_true, y_scores = to_numpy(y_true), to_numpy(y_scores)
    order = np.argsort(-y_scores)
    topk = order[:k]
    total_pos = float(np.sum(y_true))
    if total_pos == 0:
        return 0.0
    return float(np.sum(y_true[topk]) / min(k, total_pos))


def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true, y_scores = to_numpy(y_true), to_numpy(y_scores)
    order = np.argsort(-y_scores)
    ranked = y_true[order]
    pos_ranks = np.where(ranked == 1)[0] + 1
    if pos_ranks.size == 0:
        return 0.0
    return float(np.mean(1.0 / pos_ranks))


def optimal_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-10)
    if f1.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.5}
    i = int(np.argmax(f1))
    t = float(thr[i])
    pred = (y_scores >= t).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": t,
    }


def compute_all_metrics(y_true: ArrayLike, y_scores: ArrayLike, prefix: str = "") -> Dict[str, float]:
    y_true, y_scores = to_numpy(y_true), to_numpy(y_scores)
    keys = ["auroc", "auprc", "hits@1", "hits@3", "hits@10", "mrr", "precision", "recall", "f1"]
    if len(y_true) == 0 or np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return {f"{prefix}{k}": 0.0 for k in keys}
    tm = optimal_threshold_metrics(y_true, y_scores)
    return {
        f"{prefix}auroc": float(roc_auc_score(y_true, y_scores)),
        f"{prefix}auprc": float(average_precision_score(y_true, y_scores)),
        f"{prefix}hits@1": hits_at_k(y_true, y_scores, 1),
        f"{prefix}hits@3": hits_at_k(y_true, y_scores, 3),
        f"{prefix}hits@10": hits_at_k(y_true, y_scores, 10),
        f"{prefix}mrr": mean_reciprocal_rank(y_true, y_scores),
        f"{prefix}precision": tm["precision"],
        f"{prefix}recall": tm["recall"],
        f"{prefix}f1": tm["f1"],
    }
