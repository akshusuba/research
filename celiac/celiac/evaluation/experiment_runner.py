"""
Multi-seed experiment runner for rigorous evaluation.

Provides:
- Reproducible seed setting across all libraries
- Multi-seed experiment execution
- Results aggregation with mean ± std
- Experiment configuration management
"""

import os
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Type
import numpy as np
import torch

from .metrics import compute_all_metrics, METRIC_NAMES


# Default seeds for reproducibility (commonly used in ML papers)
DEFAULT_SEEDS = [0, 1, 2, 42, 123]


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Model settings
    model_name: str = 'HeteroGNN'
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.3

    # Training settings
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 20
    batch_size: Optional[int] = None  # None = full batch

    # Data settings
    dataset: str = 'curated_ced'
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    neg_sampling_ratio: float = 1.0

    # Experiment settings
    seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ablation settings (optional)
    remove_node_types: List[str] = field(default_factory=list)
    remove_edge_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentResults:
    """Results from a multi-seed experiment."""

    config: ExperimentConfig
    seed_results: Dict[int, Dict[str, float]]  # seed -> metrics
    aggregated: Dict[str, Dict[str, float]]  # metric -> {mean, std, values}
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert results to dictionary for serialization."""
        return {
            'config': self.config.to_dict(),
            'seed_results': self.seed_results,
            'aggregated': self.aggregated,
            'timestamp': self.timestamp,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentResults':
        """Load results from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls(
            config=ExperimentConfig.from_dict(d['config']),
            seed_results=d['seed_results'],
            aggregated=d['aggregated'],
            timestamp=d['timestamp'],
        )

    def get_summary_string(self, metrics: Optional[List[str]] = None) -> str:
        """Get formatted summary string for printing."""
        if metrics is None:
            metrics = ['auroc', 'auprc', 'hits@10', 'mrr', 'f1']

        lines = [f"Results for {self.config.model_name}:"]
        for m in metrics:
            if m in self.aggregated:
                mean = self.aggregated[m]['mean']
                std = self.aggregated[m]['std']
                lines.append(f"  {m}: {mean:.4f} ± {std:.4f}")
        return '\n'.join(lines)


def aggregate_results(
    seed_results: Dict[int, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across seeds.

    Args:
        seed_results: Dict mapping seed -> metrics dict

    Returns:
        Dict mapping metric_name -> {mean, std, values}
    """
    # Collect all metric values
    all_metrics = {}
    for seed, metrics in seed_results.items():
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute mean and std
    aggregated = {}
    for metric_name, values in all_metrics.items():
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values,
        }

    return aggregated


def run_multi_seed_experiment(
    train_fn: Callable[[Any, ExperimentConfig, int], torch.nn.Module],
    evaluate_fn: Callable[[torch.nn.Module, Any], Dict[str, float]],
    data: Any,
    config: ExperimentConfig,
    verbose: bool = True,
) -> ExperimentResults:
    """
    Run experiment across multiple seeds.

    Args:
        train_fn: Function(data, config, seed) -> trained model
        evaluate_fn: Function(model, data) -> metrics dict
        data: Dataset (PyG HeteroData or similar)
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        ExperimentResults with aggregated metrics
    """
    seed_results = {}

    for i, seed in enumerate(config.seeds):
        if verbose:
            print(f"Running seed {seed} ({i+1}/{len(config.seeds)})...")

        # Set seed
        set_all_seeds(seed)

        # Train model
        model = train_fn(data, config, seed)

        # Evaluate
        metrics = evaluate_fn(model, data)
        seed_results[seed] = metrics

        if verbose:
            auroc = metrics.get('auroc', metrics.get('test_auroc', 0))
            print(f"  Seed {seed}: AUROC = {auroc:.4f}")

    # Aggregate results
    aggregated = aggregate_results(seed_results)

    results = ExperimentResults(
        config=config,
        seed_results=seed_results,
        aggregated=aggregated,
    )

    if verbose:
        print("\nAggregated Results:")
        print(results.get_summary_string())

    return results


def run_experiment_suite(
    model_configs: Dict[str, ExperimentConfig],
    train_fn: Callable,
    evaluate_fn: Callable,
    data: Any,
    results_dir: str = 'results',
    verbose: bool = True,
) -> Dict[str, ExperimentResults]:
    """
    Run experiments for multiple model configurations.

    Args:
        model_configs: Dict mapping model name -> config
        train_fn: Training function
        evaluate_fn: Evaluation function
        data: Dataset
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dict mapping model name -> ExperimentResults
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_name, config in model_configs.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiments for: {model_name}")
            print('='*60)

        results = run_multi_seed_experiment(
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
            data=data,
            config=config,
            verbose=verbose,
        )

        # Save individual results
        results_path = os.path.join(results_dir, f'{model_name}_results.json')
        results.save(results_path)

        all_results[model_name] = results

    return all_results


def generate_results_table(
    all_results: Dict[str, ExperimentResults],
    metrics: Optional[List[str]] = None,
    format: str = 'markdown',
) -> str:
    """
    Generate a results table from multiple experiments.

    Args:
        all_results: Dict mapping model name -> ExperimentResults
        metrics: List of metrics to include (default: standard set)
        format: Output format ('markdown', 'latex', 'csv')

    Returns:
        Formatted table string
    """
    if metrics is None:
        metrics = ['auroc', 'auprc', 'hits@10', 'mrr', 'f1']

    # Build table data
    rows = []
    for model_name, results in all_results.items():
        row = {'Model': model_name}
        for m in metrics:
            if m in results.aggregated:
                mean = results.aggregated[m]['mean']
                std = results.aggregated[m]['std']
                row[m] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[m] = '-'
        rows.append(row)

    # Format output
    if format == 'markdown':
        header = '| Model | ' + ' | '.join(metrics) + ' |'
        separator = '|' + '|'.join(['---'] * (len(metrics) + 1)) + '|'
        body = '\n'.join([
            '| ' + row['Model'] + ' | ' + ' | '.join([row[m] for m in metrics]) + ' |'
            for row in rows
        ])
        return f"{header}\n{separator}\n{body}"

    elif format == 'latex':
        header = 'Model & ' + ' & '.join(metrics) + ' \\\\'
        body = '\n'.join([
            row['Model'] + ' & ' + ' & '.join([row[m] for m in metrics]) + ' \\\\'
            for row in rows
        ])
        return f"\\begin{{tabular}}{{{'l' + 'c' * len(metrics)}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"

    elif format == 'csv':
        header = 'Model,' + ','.join(metrics)
        body = '\n'.join([
            row['Model'] + ',' + ','.join([row[m] for m in metrics])
            for row in rows
        ])
        return f"{header}\n{body}"

    else:
        raise ValueError(f"Unknown format: {format}")
