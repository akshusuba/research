"""Evaluation framework for celiac gut-brain GNN experiments."""

from .metrics import (
    compute_all_metrics,
    hits_at_k,
    mean_reciprocal_rank,
    optimal_threshold_metrics,
)
from .experiment_runner import (
    run_multi_seed_experiment,
    set_all_seeds,
    ExperimentConfig,
)
from .statistical_tests import (
    paired_ttest,
    cohens_d,
    compare_models,
    generate_significance_table,
)

__all__ = [
    # Metrics
    'compute_all_metrics',
    'hits_at_k',
    'mean_reciprocal_rank',
    'optimal_threshold_metrics',
    # Experiment runner
    'run_multi_seed_experiment',
    'set_all_seeds',
    'ExperimentConfig',
    # Statistical tests
    'paired_ttest',
    'cohens_d',
    'compare_models',
    'generate_significance_table',
]
