"""
Statistical tests for comparing model performance.

Includes:
- Paired t-tests for comparing models across seeds
- Cohen's d effect size
- Multiple comparison correction (Bonferroni, Holm)
- Significance table generation
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings


def paired_ttest(
    values_a: List[float],
    values_b: List[float],
    alternative: str = 'two-sided',
) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of results.

    Args:
        values_a: Results from model A (one per seed)
        values_b: Results from model B (one per seed)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with t_statistic, p_value, and degrees of freedom
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length for paired test")

    if len(values_a) < 2:
        warnings.warn("Too few samples for reliable t-test")
        return {'t_statistic': 0.0, 'p_value': 1.0, 'df': 0}

    t_stat, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'df': len(values_a) - 1,
    }


def cohens_d(values_a: List[float], values_b: List[float]) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Cohen's d interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        values_a: Results from model A
        values_b: Results from model B

    Returns:
        Cohen's d effect size
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    diff = values_a - values_b
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    return float(d)


def pooled_std(values_a: List[float], values_b: List[float]) -> float:
    """Compute pooled standard deviation."""
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    n_a, n_b = len(values_a), len(values_b)
    var_a, var_b = np.var(values_a, ddof=1), np.var(values_b, ddof=1)

    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return float(pooled)


def cohens_d_pooled(values_a: List[float], values_b: List[float]) -> float:
    """
    Compute Cohen's d using pooled standard deviation.

    Alternative formulation sometimes used in literature.
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    mean_diff = np.mean(values_a) - np.mean(values_b)
    s_pooled = pooled_std(values_a, values_b)

    return float(mean_diff / (s_pooled + 1e-10))


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'negligible'
    elif d_abs < 0.5:
        return 'small'
    elif d_abs < 0.8:
        return 'medium'
    else:
        return 'large'


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        Dict with corrected alpha and significant flags
    """
    n = len(p_values)
    corrected_alpha = alpha / n

    return {
        'corrected_alpha': corrected_alpha,
        'significant': [p < corrected_alpha for p in p_values],
        'method': 'bonferroni',
    }


def holm_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply Holm-Bonferroni correction (step-down method).

    More powerful than Bonferroni while controlling FWER.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        Dict with rejection decisions
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]

    significant = [False] * n

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_pvals)):
        threshold = alpha / (n - i)
        if p < threshold:
            significant[idx] = True
        else:
            break  # Stop rejecting once we fail to reject

    return {
        'significant': significant,
        'method': 'holm',
    }


def compare_models(
    results_a: Dict[str, Any],
    results_b: Dict[str, Any],
    metric: str = 'auroc',
) -> Dict[str, Any]:
    """
    Compare two models using statistical tests.

    Args:
        results_a: Results dict with 'aggregated' containing metric values
        results_b: Results dict with 'aggregated' containing metric values
        metric: Metric to compare

    Returns:
        Dict with t-test results, effect size, and interpretation
    """
    values_a = results_a['aggregated'][metric]['values']
    values_b = results_b['aggregated'][metric]['values']

    ttest = paired_ttest(values_a, values_b)
    d = cohens_d(values_a, values_b)

    return {
        'metric': metric,
        'model_a_mean': float(np.mean(values_a)),
        'model_b_mean': float(np.mean(values_b)),
        'difference': float(np.mean(values_a) - np.mean(values_b)),
        't_statistic': ttest['t_statistic'],
        'p_value': ttest['p_value'],
        'cohens_d': d,
        'effect_interpretation': interpret_effect_size(d),
        'significant_0.05': ttest['p_value'] < 0.05,
        'significant_0.01': ttest['p_value'] < 0.01,
    }


def generate_significance_table(
    all_results: Dict[str, Any],
    baseline_model: str,
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Generate pairwise significance comparisons against a baseline.

    Args:
        all_results: Dict mapping model name -> ExperimentResults
        baseline_model: Name of baseline model to compare against
        metrics: Metrics to compare (default: auroc, auprc, mrr)
        alpha: Significance level

    Returns:
        Dict with comparison table and summary
    """
    if metrics is None:
        metrics = ['auroc', 'auprc', 'mrr']

    if baseline_model not in all_results:
        raise ValueError(f"Baseline model '{baseline_model}' not in results")

    baseline = all_results[baseline_model]
    comparisons = {}
    all_p_values = []

    for model_name, results in all_results.items():
        if model_name == baseline_model:
            continue

        model_comparisons = {}
        for metric in metrics:
            if metric in results.aggregated and metric in baseline.aggregated:
                comp = compare_models(
                    {'aggregated': results.aggregated},
                    {'aggregated': baseline.aggregated},
                    metric=metric,
                )
                model_comparisons[metric] = comp
                all_p_values.append(comp['p_value'])

        comparisons[model_name] = model_comparisons

    # Apply multiple comparison correction
    if all_p_values:
        holm_results = holm_correction(all_p_values, alpha)
        bonferroni_results = bonferroni_correction(all_p_values, alpha)
    else:
        holm_results = {'significant': [], 'method': 'holm'}
        bonferroni_results = {'significant': [], 'corrected_alpha': alpha, 'method': 'bonferroni'}

    return {
        'baseline': baseline_model,
        'comparisons': comparisons,
        'multiple_comparison_correction': {
            'holm': holm_results,
            'bonferroni': bonferroni_results,
        },
        'alpha': alpha,
    }


def format_significance_table(
    sig_table: Dict[str, Any],
    format: str = 'markdown',
) -> str:
    """
    Format significance table for display.

    Args:
        sig_table: Output from generate_significance_table
        format: 'markdown' or 'latex'

    Returns:
        Formatted table string
    """
    baseline = sig_table['baseline']
    comparisons = sig_table['comparisons']

    if not comparisons:
        return "No comparisons to display"

    # Get metrics from first comparison
    first_model = list(comparisons.keys())[0]
    metrics = list(comparisons[first_model].keys())

    rows = []
    for model_name, model_comps in comparisons.items():
        row = {'Model': f"{model_name} vs {baseline}"}
        for metric in metrics:
            if metric in model_comps:
                comp = model_comps[metric]
                p = comp['p_value']
                d = comp['cohens_d']
                diff = comp['difference']

                # Format with significance stars
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif p < 0.01:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'

                sign = '+' if diff > 0 else ''
                row[metric] = f"{sign}{diff:.3f} (d={d:.2f}){stars}"
            else:
                row[metric] = '-'
        rows.append(row)

    if format == 'markdown':
        header = '| Comparison | ' + ' | '.join(metrics) + ' |'
        separator = '|' + '|'.join(['---'] * (len(metrics) + 1)) + '|'
        body = '\n'.join([
            '| ' + row['Model'] + ' | ' + ' | '.join([row[m] for m in metrics]) + ' |'
            for row in rows
        ])
        footer = "\n*p<0.05, **p<0.01, ***p<0.001"
        return f"{header}\n{separator}\n{body}\n{footer}"

    elif format == 'latex':
        header = 'Comparison & ' + ' & '.join(metrics) + ' \\\\'
        body = '\n'.join([
            row['Model'].replace('_', '\\_') + ' & ' + ' & '.join([row[m] for m in metrics]) + ' \\\\'
            for row in rows
        ])
        return f"\\begin{{tabular}}{{{'l' + 'c' * len(metrics)}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\\\$^*p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$"

    else:
        raise ValueError(f"Unknown format: {format}")


def wilcoxon_test(
    values_a: List[float],
    values_b: List[float],
    alternative: str = 'two-sided',
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Useful when sample size is small or normality assumption is violated.

    Args:
        values_a: Results from model A
        values_b: Results from model B
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with statistic and p_value
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) < 6:
        warnings.warn("Wilcoxon test may be unreliable with fewer than 6 samples")

    try:
        stat, p_value = stats.wilcoxon(values_a, values_b, alternative=alternative)
        return {'statistic': float(stat), 'p_value': float(p_value)}
    except ValueError as e:
        warnings.warn(f"Wilcoxon test failed: {e}")
        return {'statistic': 0.0, 'p_value': 1.0}
