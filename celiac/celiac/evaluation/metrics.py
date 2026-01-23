"""
Full metrics suite for knowledge graph link prediction.

Includes:
- Threshold-free metrics: AUROC, AUPRC
- Ranking metrics: Hits@K, MRR
- Threshold-based metrics: Precision, Recall, F1 (at optimal threshold)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from typing import Dict, Optional, Union, List
import torch


def to_numpy(x: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    return x


def hits_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Compute Hits@K metric for link prediction.

    For each positive sample, check if it ranks in the top-k among all samples.

    Args:
        y_true: Binary labels (1 for positive, 0 for negative)
        y_scores: Prediction scores (higher = more likely positive)
        k: Number of top predictions to consider

    Returns:
        Proportion of positive samples ranked in top-k
    """
    y_true = to_numpy(y_true)
    y_scores = to_numpy(y_scores)

    # Get indices that would sort scores in descending order
    sorted_indices = np.argsort(-y_scores)

    # Get the top-k predicted indices
    top_k_indices = sorted_indices[:k]

    # Count how many positives are in top-k
    positives_in_top_k = np.sum(y_true[top_k_indices])
    total_positives = np.sum(y_true)

    if total_positives == 0:
        return 0.0

    return float(positives_in_top_k / total_positives)


def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for link prediction.

    For each positive sample, compute 1/rank where rank is its position
    when all samples are sorted by score. Average across all positives.

    Args:
        y_true: Binary labels (1 for positive, 0 for negative)
        y_scores: Prediction scores (higher = more likely positive)

    Returns:
        Mean reciprocal rank score
    """
    y_true = to_numpy(y_true)
    y_scores = to_numpy(y_scores)

    # Get indices that would sort scores in descending order
    sorted_indices = np.argsort(-y_scores)

    # Get ranks (1-indexed)
    ranks = np.zeros(len(y_scores))
    ranks[sorted_indices] = np.arange(1, len(y_scores) + 1)

    # Get ranks of positive samples
    positive_mask = y_true == 1
    positive_ranks = ranks[positive_mask]

    if len(positive_ranks) == 0:
        return 0.0

    # Compute mean reciprocal rank
    return float(np.mean(1.0 / positive_ranks))


def optimal_threshold_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 at the optimal threshold.

    Optimal threshold is chosen to maximize F1 score.

    Args:
        y_true: Binary labels
        y_scores: Prediction scores

    Returns:
        Dict with precision, recall, f1, and optimal_threshold
    """
    y_true = to_numpy(y_true)
    y_scores = to_numpy(y_scores)

    # Get precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 for each threshold
    # Note: precision_recall_curve returns n+1 precision/recall values
    # but only n thresholds, so we skip the last precision/recall
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Compute metrics at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)

    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'optimal_threshold': float(optimal_threshold),
    }


def compute_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    prefix: str = '',
) -> Dict[str, float]:
    """
    Compute full metrics suite for link prediction.

    Args:
        y_true: Binary labels (1 for positive, 0 for negative)
        y_scores: Prediction scores (higher = more likely positive)
        prefix: Optional prefix for metric names (e.g., 'test_')

    Returns:
        Dict containing:
            - auroc: Area under ROC curve
            - auprc: Area under precision-recall curve
            - hits@1: Hits at 1
            - hits@3: Hits at 3
            - hits@10: Hits at 10
            - mrr: Mean reciprocal rank
            - precision: Precision at optimal threshold
            - recall: Recall at optimal threshold
            - f1: F1 score at optimal threshold
    """
    y_true = to_numpy(y_true)
    y_scores = to_numpy(y_scores)

    # Handle edge cases
    if len(y_true) == 0 or np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return {f'{prefix}auroc': 0.0, f'{prefix}auprc': 0.0,
                f'{prefix}hits@1': 0.0, f'{prefix}hits@3': 0.0, f'{prefix}hits@10': 0.0,
                f'{prefix}mrr': 0.0, f'{prefix}precision': 0.0,
                f'{prefix}recall': 0.0, f'{prefix}f1': 0.0}

    # Threshold-free metrics
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Ranking metrics
    hits_1 = hits_at_k(y_true, y_scores, k=1)
    hits_3 = hits_at_k(y_true, y_scores, k=3)
    hits_10 = hits_at_k(y_true, y_scores, k=10)
    mrr = mean_reciprocal_rank(y_true, y_scores)

    # Threshold-based metrics
    threshold_metrics = optimal_threshold_metrics(y_true, y_scores)

    return {
        f'{prefix}auroc': float(auroc),
        f'{prefix}auprc': float(auprc),
        f'{prefix}hits@1': float(hits_1),
        f'{prefix}hits@3': float(hits_3),
        f'{prefix}hits@10': float(hits_10),
        f'{prefix}mrr': float(mrr),
        f'{prefix}precision': threshold_metrics['precision'],
        f'{prefix}recall': threshold_metrics['recall'],
        f'{prefix}f1': threshold_metrics['f1'],
    }


def compute_ranking_metrics_per_query(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    query_indices: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ranking metrics per query (for filtered evaluation).

    In knowledge graph evaluation, we often want to compute metrics
    per source node (query) rather than globally.

    Args:
        y_true: Binary labels for all (query, candidate) pairs
        y_scores: Scores for all pairs
        query_indices: Index indicating which query each pair belongs to

    Returns:
        Dict with mean metrics across queries
    """
    y_true = to_numpy(y_true)
    y_scores = to_numpy(y_scores)
    query_indices = to_numpy(query_indices)

    unique_queries = np.unique(query_indices)

    mrr_list = []
    hits_1_list = []
    hits_3_list = []
    hits_10_list = []

    for q in unique_queries:
        mask = query_indices == q
        q_true = y_true[mask]
        q_scores = y_scores[mask]

        if np.sum(q_true) == 0:
            continue

        mrr_list.append(mean_reciprocal_rank(q_true, q_scores))
        hits_1_list.append(hits_at_k(q_true, q_scores, k=1))
        hits_3_list.append(hits_at_k(q_true, q_scores, k=3))
        hits_10_list.append(hits_at_k(q_true, q_scores, k=10))

    return {
        'mrr': float(np.mean(mrr_list)) if mrr_list else 0.0,
        'hits@1': float(np.mean(hits_1_list)) if hits_1_list else 0.0,
        'hits@3': float(np.mean(hits_3_list)) if hits_3_list else 0.0,
        'hits@10': float(np.mean(hits_10_list)) if hits_10_list else 0.0,
    }


# Metric names for reference
METRIC_NAMES = [
    'auroc', 'auprc', 'hits@1', 'hits@3', 'hits@10',
    'mrr', 'precision', 'recall', 'f1'
]

# Primary metrics for model selection
PRIMARY_METRICS = ['auroc', 'auprc', 'mrr', 'hits@10']

# Metrics to report in tables
TABLE_METRICS = ['auroc', 'auprc', 'hits@1', 'hits@10', 'mrr', 'f1']
