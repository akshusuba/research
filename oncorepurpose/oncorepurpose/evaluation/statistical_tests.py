"""Paired statistical tests for comparing models across seeds."""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
from scipy import stats


def paired_ttest(a: List[float], b: List[float], alternative: str = "two-sided") -> Dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired_ttest requires equal-length inputs")
    if a.size < 2:
        warnings.warn("Too few samples for a reliable paired t-test")
        return {"t_statistic": 0.0, "p_value": 1.0, "df": max(0, a.size - 1)}
    t, p = stats.ttest_rel(a, b, alternative=alternative)
    return {"t_statistic": float(t), "p_value": float(p), "df": int(a.size - 1)}


def cohens_d(a: List[float], b: List[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)
